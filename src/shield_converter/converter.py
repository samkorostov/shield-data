"""
Binary parser and CSV converter for ESP32 DAQ data.

Parses fast_data.bin, medium_data.bin, and slow_data.bin files
and converts them to per-sensor CSV files.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    FirmwareMetadata,
    HealthLabel,
    SessionRecord,
    SensorType,
)

# ==================== Binary Record Formats ====================
# sensor_data_record_v2_t, packed little-endian, 20 bytes:
# uint32 timestamp_ms, uint8 sensor_id, uint8 kind, uint8 axis_count,
# uint8 flags, float data[3].

SENSOR_RECORD_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u4"),
        ("sensor_id", "u1"),
        ("kind", "u1"),
        ("axis_count", "u1"),
        ("flags", "u1"),
        ("data", "<f4", (3,)),
    ],
    align=False,
)
SENSOR_RECORD_SIZE = SENSOR_RECORD_DTYPE.itemsize  # 20 bytes

# The three SD-card files now share one binary record layout. Keep the old
# constants exported for callers/tests that import file-specific names.
FAST_RECORD_DTYPE = SENSOR_RECORD_DTYPE
FAST_RECORD_SIZE = SENSOR_RECORD_SIZE
MEDIUM_RECORD_DTYPE = SENSOR_RECORD_DTYPE
MEDIUM_RECORD_SIZE = SENSOR_RECORD_SIZE
SLOW_RECORD_DTYPE = SENSOR_RECORD_DTYPE
SLOW_RECORD_SIZE = SENSOR_RECORD_SIZE

RAW_KIND = 0
PROCESSED_KIND = 1
MS_PER_HOUR = 3_600_000

# ==================== Sensor ID Mapping ====================

SENSOR_ID_TO_NAME = {
    1: "vibration",
    2: "current",
    3: "pressure",
    4: "temperature",
    5: "microphone",
    6: "photodiode",
    7: "magnetometer",
    8: "gyroscope",
    9: "accelerometer",
}

SCALAR_SENSOR_IDS = {1, 2, 3, 4, 5, 6}
THREE_AXIS_SENSOR_IDS = {7, 8, 9}  # magnetometer, gyroscope, accelerometer

DATA_FILE_SENSOR_IDS = {
    "fast_data.bin": [1, 5, 7, 8, 9],
    "medium_data.bin": [2, 6],
    "slow_data.bin": [3, 4],
}

SENSOR_NAME_TO_INFO = {
    "vibration": {
        "id": 1,
        "type": SensorType.VIBRATION,
        "rate": 1000,
        "unit": "binary",
    },
    "current": {"id": 2, "type": SensorType.CURRENT, "rate": 200, "unit": "A"},
    "pressure": {"id": 3, "type": SensorType.PRESSURE, "rate": 50, "unit": "kPa"},
    "temperature": {"id": 4, "type": SensorType.TEMPERATURE, "rate": 50, "unit": "C"},
    "microphone": {
        "id": 5,
        "type": SensorType.MICROPHONE,
        "rate": 1000,
        "unit": "dBFS",
    },
    "photodiode": {"id": 6, "type": SensorType.PHOTODIODE, "rate": 200, "unit": "V"},
    "magnetometer": {
        "id": 7,
        "type": SensorType.MAGNETOMETER,
        "rate": 1000,
        "unit": "uT",
    },
    "gyroscope": {"id": 8, "type": SensorType.GYROSCOPE, "rate": 1000, "unit": "rad/s"},
    "accelerometer": {
        "id": 9,
        "type": SensorType.ACCELEROMETER,
        "rate": 1000,
        "unit": "m/s^2",
    },
}

# ==================== Binary Parsers ====================


def _read_structured_records(filepath: Path, dtype: np.dtype) -> np.ndarray:
    """Read only full records from a binary file into a NumPy structured array."""
    file_size = filepath.stat().st_size
    record_count = file_size // dtype.itemsize
    if record_count == 0:
        return np.empty(0, dtype=dtype)
    return np.fromfile(filepath, dtype=dtype, count=record_count)


def _read_record_view(filepath: Path, dtype: np.dtype) -> np.ndarray:
    """Return a read-only view of full records in a binary file."""
    file_size = filepath.stat().st_size
    record_count = file_size // dtype.itemsize
    if record_count == 0:
        return np.empty(0, dtype=dtype)
    return np.memmap(filepath, dtype=dtype, mode="r", shape=(record_count,))


def parse_fast_data(filepath: Path) -> np.ndarray:
    """
    Parse fast_data.bin file.

    Each v2 record is 20 bytes: uint32 timestamp_ms, uint8 sensor_id,
    uint8 kind, uint8 axis_count, uint8 flags, float data[3].

    Args:
        filepath: Path to fast_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, kind,
        axis_count, flags, and data[3]
    """
    return _read_structured_records(filepath, FAST_RECORD_DTYPE)


def parse_medium_data(filepath: Path) -> np.ndarray:
    """
    Parse medium_data.bin file.

    Args:
        filepath: Path to medium_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, kind,
        axis_count, flags, and data[3]
    """
    return _read_structured_records(filepath, MEDIUM_RECORD_DTYPE)


def parse_slow_data(filepath: Path) -> np.ndarray:
    """
    Parse slow_data.bin file.

    Args:
        filepath: Path to slow_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, kind,
        axis_count, flags, and data[3]
    """
    return _read_structured_records(filepath, SLOW_RECORD_DTYPE)


# ==================== Data Processing ====================


def _dataframe_for_sensor_records(
    sensor_id: int, timestamps: np.ndarray, values: np.ndarray
) -> pd.DataFrame:
    """Build a legacy-compatible per-sensor DataFrame from v2 record arrays."""
    if sensor_id in THREE_AXIS_SENSOR_IDS:
        return pd.DataFrame(
            {
                "timestamp_ms": timestamps,
                "x": values[:, 0],
                "y": values[:, 1],
                "z": values[:, 2],
            }
        )

    return pd.DataFrame(
        {
            "timestamp_ms": timestamps,
            "value": values[:, 0],
        }
    )


def _output_name_for_kind(sensor_name: str, kind: int) -> str:
    if kind == RAW_KIND:
        return sensor_name
    if kind == PROCESSED_KIND:
        return f"proc_{sensor_name}"
    return f"{sensor_name}_kind_{kind}"


def _base_sensor_name(output_name: str) -> str:
    if output_name.startswith("proc_"):
        return output_name[len("proc_") :]
    if output_name.endswith("_processed"):
        return output_name[: -len("_processed")]
    if "_kind_" in output_name:
        return output_name.split("_kind_", 1)[0]
    return output_name


def _normalize_time_window(
    start_hour: Optional[float], end_hour: Optional[float]
) -> Optional[Tuple[float, Optional[float]]]:
    if start_hour is None and end_hour is None:
        return None

    start = 0.0 if start_hour is None else float(start_hour)
    end = None if end_hour is None else float(end_hour)

    if start < 0:
        raise ValueError("start_hour must be >= 0")
    if end is not None and end <= start:
        raise ValueError("end_hour must be greater than start_hour")

    return start, end


def _hours_to_ms(hours: float) -> int:
    return int(round(hours * MS_PER_HOUR))


def _find_run_start_ms(run_dir: Path) -> Optional[int]:
    starts = []
    for file_name in DATA_FILE_SENSOR_IDS:
        data_path = run_dir / file_name
        if not data_path.exists():
            continue
        records = _read_record_view(data_path, SENSOR_RECORD_DTYPE)
        if records.size:
            starts.append(int(records["timestamp_ms"].min()))
    if not starts:
        return None
    return min(starts)


def _absolute_time_window_ms(
    run_start_ms: Optional[int],
    time_window: Optional[Tuple[float, Optional[float]]],
) -> Tuple[Optional[int], Optional[int]]:
    if time_window is None or run_start_ms is None:
        return None, None

    start_hour, end_hour = time_window
    start_ms = run_start_ms + _hours_to_ms(start_hour)
    end_ms = None if end_hour is None else run_start_ms + _hours_to_ms(end_hour)
    return start_ms, end_ms


def _filter_records_by_time(
    records: np.ndarray,
    start_ms: Optional[int],
    end_ms: Optional[int],
) -> np.ndarray:
    if start_ms is None and end_ms is None:
        return records

    mask = np.ones(records.size, dtype=bool)
    if start_ms is not None:
        mask &= records["timestamp_ms"] >= start_ms
    if end_ms is not None:
        mask &= records["timestamp_ms"] < end_ms
    return records[mask]


def _collect_record_coverage(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, Tuple[int, int]]:
    coverage: Dict[str, Tuple[int, int]] = {}
    if records.size == 0:
        return coverage

    timestamps = records["timestamp_ms"]
    sensor_ids_arr = records["sensor_id"]
    kinds = records["kind"]

    for sid in sensor_ids:
        sensor_name = SENSOR_ID_TO_NAME.get(sid)
        if sensor_name is None:
            continue

        sensor_mask = sensor_ids_arr == sid
        if not np.any(sensor_mask):
            continue

        for kind in np.unique(kinds[sensor_mask]):
            kind_int = int(kind)
            kind_mask = sensor_mask & (kinds == kind_int)
            output_name = _output_name_for_kind(sensor_name, kind_int)
            sensor_timestamps = timestamps[kind_mask]
            coverage[output_name] = (
                int(sensor_timestamps.min()),
                int(sensor_timestamps.max()),
            )

    return coverage


def _dataframe_duration_s(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    timestamps = df["timestamp_ms"].to_numpy(copy=False)
    return (int(timestamps.max()) - int(timestamps.min())) / 1000.0


def _format_relative_window(
    start_ms: int,
    end_ms: int,
    run_start_ms: int,
) -> str:
    start_h = (start_ms - run_start_ms) / MS_PER_HOUR
    end_h = (end_ms - run_start_ms) / MS_PER_HOUR
    return f"{start_h:.2f}h-{end_h:.2f}h"


def _format_requested_window(
    time_window: Tuple[float, Optional[float]],
) -> str:
    start_hour, end_hour = time_window
    if end_hour is None:
        return f"{start_hour:.2f}h-end"
    return f"{start_hour:.2f}h-{end_hour:.2f}h"


def _print_time_window_warnings(
    coverage: Dict[str, Tuple[int, int]],
    output_names: set[str],
    time_window: Optional[Tuple[float, Optional[float]]],
    run_start_ms: Optional[int],
    start_ms: Optional[int],
    end_ms: Optional[int],
    verbose: bool,
) -> None:
    if not verbose or time_window is None or run_start_ms is None or start_ms is None:
        return

    requested = _format_requested_window(time_window)
    for output_name in sorted(coverage):
        first_ms, last_ms = coverage[output_name]
        available = _format_relative_window(first_ms, last_ms, run_start_ms)

        if output_name not in output_names:
            print(
                f"    Warning: {output_name} has no data in requested "
                f"{requested}; available {available}"
            )
            continue

        misses_start = first_ms > start_ms
        misses_end = end_ms is not None and last_ms < end_ms
        if misses_start or misses_end:
            print(
                f"    Warning: {output_name} only covers {available}; "
                f"requested {requested}"
            )


def split_records_by_sensor_kind(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split v2 records by sensor ID and kind into legacy-compatible DataFrames.

    Raw records are written under the plain sensor name, e.g. ``vibration``.
    Processed records are written under ``proc_<sensor>``. CSV columns stay
    identical to the old converter: scalar sensors use [timestamp_ms, value],
    vector sensors use [timestamp_ms, x, y, z].

    Args:
        records: Structured NumPy array from a v2 binary file parser
        sensor_ids: List of expected sensor IDs in this data tier

    Returns:
        Dictionary mapping output file stems to DataFrames
    """
    if records.size == 0:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    timestamps = records["timestamp_ms"]
    sensor_ids_arr = records["sensor_id"]
    kinds = records["kind"]
    values = records["data"]

    for sid in sensor_ids:
        name = SENSOR_ID_TO_NAME.get(sid)
        if name is None:
            continue

        sensor_mask = sensor_ids_arr == sid
        if not np.any(sensor_mask):
            continue

        for kind in np.unique(kinds[sensor_mask]):
            kind_int = int(kind)
            kind_mask = sensor_mask & (kinds == kind_int)
            output_name = _output_name_for_kind(name, kind_int)
            result[output_name] = _dataframe_for_sensor_records(
                sid,
                timestamps[kind_mask],
                values[kind_mask],
            )

    return result


def split_fast_by_sensor(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split fast data records by sensor ID and kind into separate DataFrames.

    Args:
        records: Structured NumPy array from parse_fast_data
        sensor_ids: List of expected sensor IDs in fast data tier

    Returns:
        Dictionary mapping output file stems to DataFrames
    """
    return split_records_by_sensor_kind(records, sensor_ids)


def split_by_sensor(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split records by sensor ID and kind into separate DataFrames.

    Kept for compatibility with older callers. The v2 medium/slow files use
    the same record layout as fast data, so this delegates to the generic
    splitter.
    """
    return split_records_by_sensor_kind(records, sensor_ids)


# ==================== Metadata Handling ====================


def load_firmware_metadata(filepath: Path) -> FirmwareMetadata:
    """
    Load and parse firmware meta.json file.

    Args:
        filepath: Path to meta.json

    Returns:
        Parsed FirmwareMetadata object
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return FirmwareMetadata.model_validate(data)


def parse_folder_name(folder_name: str) -> Tuple[str, str]:
    """
    Extract unit_id and run_id from folder naming convention.

    Expected format: UNIT_XXX_RUN_YYY or variations

    Args:
        folder_name: Folder name string

    Returns:
        Tuple of (unit_id, run_id)

    Examples:
        "UNIT_001_RUN_001" -> ("unit_0001", "RUN_001")
        "RUN_001" -> ("unit_0001", "RUN_001")
    """
    # Try to match UNIT_XXX_RUN_YYY pattern
    match = re.match(r"UNIT_(\d+)_RUN_(\d+)", folder_name, re.IGNORECASE)
    if match:
        unit_num = int(match.group(1))
        run_num = int(match.group(2))
        return (f"unit_{unit_num:04d}", f"RUN_{run_num:03d}")

    # Try to match just RUN_XXX pattern
    match = re.match(r"RUN_(\d+)", folder_name, re.IGNORECASE)
    if match:
        run_num = int(match.group(1))
        return ("unit_0001", f"RUN_{run_num:03d}")

    # Default fallback
    return ("unit_0001", folder_name)


def generate_session_metadata(
    firmware_meta: Optional[FirmwareMetadata],
    unit_id: str,
    run_id: str,
    output_files: Dict[str, Path],
    health_label: HealthLabel = HealthLabel.UNKNOWN,
    output_durations_s: Optional[Dict[str, float]] = None,
) -> List[SessionRecord]:
    """
    Generate session metadata records for all output files.

    Args:
        firmware_meta: Parsed firmware metadata (or None if unavailable)
        unit_id: Physical unit identifier
        run_id: Run/session identifier
        output_files: Dictionary mapping sensor name to output CSV path
        health_label: Health label to apply to all sensors
        output_durations_s: Actual duration in seconds for each output CSV

    Returns:
        List of SessionRecord objects
    """
    records = []
    output_durations_s = output_durations_s or {}

    # Determine start time and fallback duration
    if firmware_meta:
        try:
            start_timestamp = int(firmware_meta.start_time)
            start_time = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
        except (ValueError, TypeError):
            start_time = datetime.now(tz=timezone.utc)
        default_duration_s = firmware_meta.statistics.duration_ms / 1000.0
    else:
        start_time = datetime.now(tz=timezone.utc)
        default_duration_s = 0.0

    for sensor_name, csv_path in output_files.items():
        base_sensor_name = _base_sensor_name(sensor_name)
        info = SENSOR_NAME_TO_INFO.get(base_sensor_name, {})
        duration_s = output_durations_s.get(sensor_name, default_duration_s)
        record = SessionRecord(
            session_id=run_id,
            unit_id=unit_id,
            sensor_name=sensor_name,
            file_name=csv_path.name,
            file_format="csv",
            start_time_utc=start_time,
            duration_s=duration_s,
            sampling_rate_hz=info.get("rate", 0),
            units=info.get("unit", "unknown"),
            health_label=health_label,
        )
        records.append(record)

    return records


# ==================== Main Conversion Function ====================


def convert_run(
    run_dir: Path,
    output_dir: Path,
    unit_id: Optional[str] = None,
    health_label: HealthLabel = HealthLabel.UNKNOWN,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Path], List[SessionRecord]]:
    """
    Convert a single run directory from binary to CSV format.

    Args:
        run_dir: Path to run directory containing binary files
        output_dir: Base output directory
        unit_id: Override unit ID (default: extracted from folder name)
        health_label: Health label to apply to all sensors
        start_hour: Optional start time relative to run start, in hours
        end_hour: Optional end time relative to run start, in hours
        verbose: Print progress messages

    Returns:
        Tuple of (output_files dict, session_records list)
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    time_window = _normalize_time_window(start_hour, end_hour)
    run_start_ms = _find_run_start_ms(run_dir) if time_window else None
    start_ms, end_ms = _absolute_time_window_ms(run_start_ms, time_window)

    # Parse folder name for unit_id and run_id
    folder_unit_id, run_id = parse_folder_name(run_dir.name)
    if unit_id is None:
        unit_id = folder_unit_id

    if verbose:
        print(f"Converting run: {run_dir.name}")
        print(f"  Unit ID: {unit_id}")
        print(f"  Run ID: {run_id}")
        if time_window:
            print(f"  Time window: {_format_requested_window(time_window)}")
            if run_start_ms is None:
                print("  Warning: no binary records found; time window cannot be applied")

    # Load firmware metadata if available
    meta_path = run_dir / "meta.json"
    firmware_meta = None
    if meta_path.exists():
        try:
            firmware_meta = load_firmware_metadata(meta_path)
            if verbose:
                print(f"  Loaded metadata: {firmware_meta.run_id}")
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load meta.json: {e}")

    # Create output directory
    run_output_dir = output_dir / "data" / f"{unit_id.upper()}_{run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in run_output_dir.glob("*.csv"):
        csv_path.unlink()

    output_files: Dict[str, Path] = {}
    all_dataframes: Dict[str, pd.DataFrame] = {}
    output_durations_s: Dict[str, float] = {}
    coverage_by_output: Dict[str, Tuple[int, int]] = {}

    parsers = {
        "fast_data.bin": parse_fast_data,
        "medium_data.bin": parse_medium_data,
        "slow_data.bin": parse_slow_data,
    }

    for file_name, sensor_ids in DATA_FILE_SENSOR_IDS.items():
        data_path = run_dir / file_name
        if not data_path.exists():
            continue

        if verbose:
            print(f"  Processing {file_name}...")
        records = parsers[file_name](data_path)
        if verbose:
            print(f"    Read {len(records)} records")

        if time_window:
            coverage_by_output.update(_collect_record_coverage(records, sensor_ids))
            records = _filter_records_by_time(records, start_ms, end_ms)
            if verbose:
                print(f"    Kept {len(records)} records in requested time window")

        sensor_dfs = split_records_by_sensor_kind(records, sensor_ids)
        for output_name, df in sensor_dfs.items():
            if output_name in all_dataframes:
                all_dataframes[output_name] = pd.concat(
                    [all_dataframes[output_name], df], ignore_index=True
                )
            else:
                all_dataframes[output_name] = df

    if time_window:
        _print_time_window_warnings(
            coverage=coverage_by_output,
            output_names=set(all_dataframes.keys()),
            time_window=time_window,
            run_start_ms=run_start_ms,
            start_ms=start_ms,
            end_ms=end_ms,
            verbose=verbose,
        )

    # Write CSV files
    for sensor_name, df in all_dataframes.items():
        csv_path = run_output_dir / f"{sensor_name}.csv"
        df.to_csv(csv_path, index=False)
        output_files[sensor_name] = csv_path
        output_durations_s[sensor_name] = _dataframe_duration_s(df)
        if verbose:
            print(f"    Wrote {csv_path.name}: {len(df)} samples")

    # Generate session metadata
    session_records = generate_session_metadata(
        firmware_meta=firmware_meta,
        unit_id=unit_id,
        run_id=run_id,
        output_files=output_files,
        output_durations_s=output_durations_s,
        health_label=health_label,
    )

    return output_files, session_records


def convert_all_runs(
    input_dir: Path,
    output_dir: Path,
    health_label: HealthLabel = HealthLabel.UNKNOWN,
    start_hour: Optional[float] = None,
    end_hour: Optional[float] = None,
    verbose: bool = False,
) -> List[SessionRecord]:
    """
    Convert all run directories found in input directory.

    Looks for directories matching UNIT_XXX_RUN_YYY or RUN_XXX patterns,
    or directories containing binary data files.

    Args:
        input_dir: Base input directory (e.g., SD card mount point)
        output_dir: Base output directory
        health_label: Health label to apply to all sensors
        start_hour: Optional start time relative to each run start, in hours
        end_hour: Optional end time relative to each run start, in hours
        verbose: Print progress messages

    Returns:
        List of all SessionRecord objects created
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    _normalize_time_window(start_hour, end_hour)
    all_sessions: List[SessionRecord] = []

    # Find run directories
    run_dirs = []
    for item in input_dir.iterdir():
        if item.is_dir():
            # Check if directory contains binary data files
            has_data = any(
                (item / f).exists()
                for f in ["fast_data.bin", "medium_data.bin", "slow_data.bin"]
            )
            if has_data:
                run_dirs.append(item)

    if verbose:
        print(f"Found {len(run_dirs)} run directories")

    # Convert each run
    for run_dir in sorted(run_dirs):
        try:
            _, sessions = convert_run(
                run_dir=run_dir,
                output_dir=output_dir,
                health_label=health_label,
                start_hour=start_hour,
                end_hour=end_hour,
                verbose=verbose,
            )
            all_sessions.extend(sessions)
        except Exception as e:
            print(f"Error converting {run_dir.name}: {e}")

    # Write combined sessions metadata
    if all_sessions:
        metadata_dir = output_dir / "metadata" / "sessions"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sessions_csv = metadata_dir / "sessions.csv"

        sessions_df = pd.DataFrame([s.to_csv_row() for s in all_sessions])
        sessions_df.to_csv(sessions_csv, index=False)
        if verbose:
            print(f"Wrote sessions metadata: {sessions_csv}")

    return all_sessions


def validate_run(run_dir: Path, verbose: bool = False) -> Dict[str, Any]:
    """
    Validate binary files in a run directory.

    Checks file existence, record counts, and compares with metadata.

    Args:
        run_dir: Path to run directory
        verbose: Print detailed information

    Returns:
        Validation results dictionary
    """
    run_dir = Path(run_dir)
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "files": {},
    }

    # Check for binary files
    for fname, record_size, parser in [
        ("fast_data.bin", FAST_RECORD_SIZE, parse_fast_data),
        ("medium_data.bin", MEDIUM_RECORD_SIZE, parse_medium_data),
        ("slow_data.bin", SLOW_RECORD_SIZE, parse_slow_data),
    ]:
        fpath = run_dir / fname
        if fpath.exists():
            file_size = fpath.stat().st_size
            expected_records = file_size // record_size
            remainder = file_size % record_size

            results["files"][fname] = {
                "exists": True,
                "size_bytes": file_size,
                "expected_records": expected_records,
                "has_partial_record": remainder > 0,
            }

            if remainder > 0:
                results["warnings"].append(
                    f"{fname}: File size ({file_size}) not evenly divisible by "
                    f"record size ({record_size})"
                )

            records = parser(fpath)
            if records.size:
                unknown_sensor_mask = ~np.isin(
                    records["sensor_id"], list(SENSOR_ID_TO_NAME.keys())
                )
                unknown_kind_mask = ~np.isin(records["kind"], [RAW_KIND, PROCESSED_KIND])

                known_sensor_mask = ~unknown_sensor_mask
                expected_axis_counts = np.where(
                    np.isin(records["sensor_id"], list(THREE_AXIS_SENSOR_IDS)), 3, 1
                )
                bad_axis_mask = known_sensor_mask & (
                    records["axis_count"] != expected_axis_counts
                )

                unknown_sensor_count = int(np.count_nonzero(unknown_sensor_mask))
                unknown_kind_count = int(np.count_nonzero(unknown_kind_mask))
                bad_axis_count = int(np.count_nonzero(bad_axis_mask))

                results["files"][fname].update(
                    {
                        "unknown_sensor_records": unknown_sensor_count,
                        "unknown_kind_records": unknown_kind_count,
                        "bad_axis_count_records": bad_axis_count,
                    }
                )

                if unknown_sensor_count:
                    results["warnings"].append(
                        f"{fname}: {unknown_sensor_count} records have unknown sensor_id"
                    )
                if unknown_kind_count:
                    results["warnings"].append(
                        f"{fname}: {unknown_kind_count} records have unknown kind"
                    )
                if bad_axis_count:
                    results["warnings"].append(
                        f"{fname}: {bad_axis_count} records have unexpected axis_count"
                    )

            if verbose:
                print(f"  {fname}: {file_size} bytes, ~{expected_records} records")
        else:
            results["files"][fname] = {"exists": False}
            if verbose:
                print(f"  {fname}: not found")

    # Check metadata
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = load_firmware_metadata(meta_path)
            results["metadata"] = {
                "exists": True,
                "run_id": meta.run_id,
                "duration_ms": meta.statistics.duration_ms,
                "total_samples": meta.statistics.total_samples,
            }
            if verbose:
                print(
                    f"  meta.json: run_id={meta.run_id}, duration={meta.statistics.duration_ms}ms"
                )
        except Exception as e:
            results["errors"].append(f"Could not parse meta.json: {e}")
            results["valid"] = False
    else:
        results["metadata"] = {"exists": False}
        results["warnings"].append("meta.json not found")

    return results
