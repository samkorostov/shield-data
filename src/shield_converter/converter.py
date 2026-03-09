"""
Binary parser and CSV converter for ESP32 DAQ data.

Parses fast_data.bin, medium_data.bin, and slow_data.bin files
and converts them to per-sensor CSV files.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    FirmwareMetadata,
    HealthLabel,
    SessionRecord,
    SensorInfo,
    SensorType,
)

# ==================== Binary Record Formats ====================
# Based on data_types.h with __attribute__((packed))

# Fast data: uint32_t timestamp + uint8_t sensor_id + uint8_t[3] reserved + float data[3]
FAST_RECORD_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u4"),
        ("sensor_id", "u1"),
        ("_reserved", "u1", (3,)),
        ("data", "<f4", (3,)),
    ],
    align=False,
)
FAST_RECORD_SIZE = FAST_RECORD_DTYPE.itemsize  # 20 bytes

# Medium data: uint32_t timestamp + uint8_t sensor_id + uint8_t[3] reserved + float data
MEDIUM_RECORD_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u4"),
        ("sensor_id", "u1"),
        ("_reserved", "u1", (3,)),
        ("value", "<f4"),
    ],
    align=False,
)
MEDIUM_RECORD_SIZE = MEDIUM_RECORD_DTYPE.itemsize  # 12 bytes

# Slow data: uint32_t timestamp + uint8_t sensor_id + uint8_t[3] reserved + float data
SLOW_RECORD_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u4"),
        ("sensor_id", "u1"),
        ("_reserved", "u1", (3,)),
        ("value", "<f4"),
    ],
    align=False,
)
SLOW_RECORD_SIZE = SLOW_RECORD_DTYPE.itemsize  # 12 bytes

# ==================== Sensor ID Mapping ====================

SENSOR_ID_TO_NAME = {
    0: "imu",
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

THREE_AXIS_SENSOR_IDS = {0, 7, 8, 9}  # imu, magnetometer, gyroscope, accelerometer

SENSOR_NAME_TO_INFO = {
    "imu": {"id": 0, "type": SensorType.IMU, "rate": 1000, "unit": "m/s^2"},
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


def parse_fast_data(filepath: Path) -> np.ndarray:
    """
    Parse fast_data.bin file.

    Each record is 20 bytes: uint32 timestamp_ms, uint8 sensor_id,
    3 reserved bytes, float data[0], float data[1], float data[2].

    Args:
        filepath: Path to fast_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, and data[3]
    """
    return _read_structured_records(filepath, FAST_RECORD_DTYPE)


def parse_medium_data(filepath: Path) -> np.ndarray:
    """
    Parse medium_data.bin file.

    Args:
        filepath: Path to medium_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, and value
    """
    return _read_structured_records(filepath, MEDIUM_RECORD_DTYPE)


def parse_slow_data(filepath: Path) -> np.ndarray:
    """
    Parse slow_data.bin file.

    Args:
        filepath: Path to slow_data.bin

    Returns:
        Structured NumPy array with fields timestamp_ms, sensor_id, and value
    """
    return _read_structured_records(filepath, SLOW_RECORD_DTYPE)


# ==================== Data Processing ====================


def split_fast_by_sensor(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split fast data records by sensor ID into separate DataFrames.

    Three-axis sensors (IMU, magnetometer, gyroscope, accelerometer) produce
    DataFrames with columns [timestamp_ms, x, y, z].
    Scalar sensors produce DataFrames with columns [timestamp_ms, value].

    Args:
        records: Structured NumPy array from parse_fast_data
        sensor_ids: List of expected sensor IDs in fast data tier

    Returns:
        Dictionary mapping sensor name to DataFrame
    """
    if records.size == 0:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    timestamps = records["timestamp_ms"]
    sensor_ids_arr = records["sensor_id"]
    values = records["data"]

    for sid in sensor_ids:
        name = SENSOR_ID_TO_NAME.get(sid)
        if name is None:
            continue

        mask = sensor_ids_arr == sid
        if not np.any(mask):
            continue

        sensor_timestamps = timestamps[mask]
        sensor_values = values[mask]

        if sid in THREE_AXIS_SENSOR_IDS:
            result[name] = pd.DataFrame(
                {
                    "timestamp_ms": sensor_timestamps,
                    "x": sensor_values[:, 0],
                    "y": sensor_values[:, 1],
                    "z": sensor_values[:, 2],
                }
            )
        else:
            result[name] = pd.DataFrame(
                {
                    "timestamp_ms": sensor_timestamps,
                    "value": sensor_values[:, 0],
                }
            )

    return result


def split_by_sensor(
    records: np.ndarray, sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split records by sensor ID into separate DataFrames.

    Used for medium_data and slow_data which have scalar values only.

    Args:
        records: Structured NumPy array from parse_medium_data/parse_slow_data
        sensor_ids: List of expected sensor IDs in this data tier

    Returns:
        Dictionary mapping sensor name to DataFrame with timestamp_ms and value columns
    """
    if records.size == 0:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    timestamps = records["timestamp_ms"]
    sensor_ids_arr = records["sensor_id"]
    values = records["value"]

    for sid in sensor_ids:
        sensor_name = SENSOR_ID_TO_NAME.get(sid)
        if sensor_name is None:
            continue

        mask = sensor_ids_arr == sid
        if not np.any(mask):
            continue

        result[sensor_name] = pd.DataFrame(
            {
                "timestamp_ms": timestamps[mask],
                "value": values[mask],
            }
        )

    return result


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
) -> List[SessionRecord]:
    """
    Generate session metadata records for all output files.

    Args:
        firmware_meta: Parsed firmware metadata (or None if unavailable)
        unit_id: Physical unit identifier
        run_id: Run/session identifier
        output_files: Dictionary mapping sensor name to output CSV path
        health_label: Health label to apply to all sensors

    Returns:
        List of SessionRecord objects
    """
    records = []

    # Determine start time and duration
    if firmware_meta:
        try:
            start_timestamp = int(firmware_meta.start_time)
            start_time = datetime.fromtimestamp(start_timestamp, tz=timezone.utc)
        except (ValueError, TypeError):
            start_time = datetime.now(tz=timezone.utc)
        duration_s = firmware_meta.statistics.duration_ms / 1000.0
    else:
        start_time = datetime.now(tz=timezone.utc)
        duration_s = 0.0

    for sensor_name, csv_path in output_files.items():
        info = SENSOR_NAME_TO_INFO.get(sensor_name, {})
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
    verbose: bool = False,
) -> Tuple[Dict[str, Path], List[SessionRecord]]:
    """
    Convert a single run directory from binary to CSV format.

    Args:
        run_dir: Path to run directory containing binary files
        output_dir: Base output directory
        unit_id: Override unit ID (default: extracted from folder name)
        health_label: Health label to apply to all sensors
        verbose: Print progress messages

    Returns:
        Tuple of (output_files dict, session_records list)
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)

    # Parse folder name for unit_id and run_id
    folder_unit_id, run_id = parse_folder_name(run_dir.name)
    if unit_id is None:
        unit_id = folder_unit_id

    if verbose:
        print(f"Converting run: {run_dir.name}")
        print(f"  Unit ID: {unit_id}")
        print(f"  Run ID: {run_id}")

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

    output_files: Dict[str, Path] = {}
    all_dataframes: Dict[str, pd.DataFrame] = {}

    # Process fast data (IMU + Vibration + Microphone)
    fast_path = run_dir / "fast_data.bin"
    if fast_path.exists():
        if verbose:
            print(f"  Processing fast_data.bin...")
        records = parse_fast_data(fast_path)
        if verbose:
            print(f"    Read {len(records)} records")
        sensor_dfs = split_fast_by_sensor(
            records, [0, 1, 5, 7, 8, 9]
        )  # IMU=0, Vibration=1, Microphone=5, Magnetometer=7, Gyroscope=8, Accelerometer=9
        all_dataframes.update(sensor_dfs)

    # Process medium data (Current + Photodiode)
    medium_path = run_dir / "medium_data.bin"
    if medium_path.exists():
        if verbose:
            print(f"  Processing medium_data.bin...")
        records = parse_medium_data(medium_path)
        if verbose:
            print(f"    Read {len(records)} records")
        sensor_dfs = split_by_sensor(records, [2, 6])  # Current=2, Photodiode=6
        all_dataframes.update(sensor_dfs)

    # Process slow data (Pressure + Temperature)
    slow_path = run_dir / "slow_data.bin"
    if slow_path.exists():
        if verbose:
            print(f"  Processing slow_data.bin...")
        records = parse_slow_data(slow_path)
        if verbose:
            print(f"    Read {len(records)} records")
        sensor_dfs = split_by_sensor(records, [3, 4])  # Pressure=3, Temperature=4
        all_dataframes.update(sensor_dfs)

    # Write CSV files
    for sensor_name, df in all_dataframes.items():
        csv_path = run_output_dir / f"{sensor_name}.csv"
        df.to_csv(csv_path, index=False)
        output_files[sensor_name] = csv_path
        if verbose:
            print(f"    Wrote {csv_path.name}: {len(df)} samples")

    # Generate session metadata
    session_records = generate_session_metadata(
        firmware_meta=firmware_meta,
        unit_id=unit_id,
        run_id=run_id,
        output_files=output_files,
        health_label=health_label,
    )

    return output_files, session_records


def convert_all_runs(
    input_dir: Path,
    output_dir: Path,
    health_label: HealthLabel = HealthLabel.UNKNOWN,
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
        verbose: Print progress messages

    Returns:
        List of all SessionRecord objects created
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
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


def validate_run(run_dir: Path, verbose: bool = False) -> Dict[str, any]:
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
                    f"{fname}: File size ({file_size}) not evenly divisible by record size ({record_size})"
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
