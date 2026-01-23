"""
Binary parser and CSV converter for ESP32 DAQ data.

Parses fast_data.bin, medium_data.bin, and slow_data.bin files
and converts them to per-sensor CSV files.
"""

import json
import re
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Fast data: uint32_t timestamp + uint8_t sensor_id + uint8_t[3] reserved + float data
FAST_RECORD_FORMAT = "<I B 3x f"  # < = little-endian, I = uint32, B = uint8, 3x = 3 pad bytes, f = float
FAST_RECORD_SIZE = struct.calcsize(FAST_RECORD_FORMAT)  # Should be 12 bytes

# Medium data: uint32_t timestamp + float current
MEDIUM_RECORD_FORMAT = "<I f"
MEDIUM_RECORD_SIZE = struct.calcsize(MEDIUM_RECORD_FORMAT)  # Should be 8 bytes

# Slow data: uint32_t timestamp + uint8_t sensor_id + uint8_t[3] reserved + float data
SLOW_RECORD_FORMAT = "<I B 3x f"
SLOW_RECORD_SIZE = struct.calcsize(SLOW_RECORD_FORMAT)  # Should be 12 bytes

# ==================== Sensor ID Mapping ====================

SENSOR_ID_TO_NAME = {
    0: "imu",
    1: "vibration",
    2: "current",
    3: "pressure",
    4: "temperature",
}

SENSOR_NAME_TO_INFO = {
    "imu": {"id": 0, "type": SensorType.IMU, "rate": 1000, "unit": "m/s^2"},
    "vibration": {"id": 1, "type": SensorType.VIBRATION, "rate": 1000, "unit": "binary"},
    "current": {"id": 2, "type": SensorType.CURRENT, "rate": 200, "unit": "A"},
    "pressure": {"id": 3, "type": SensorType.PRESSURE, "rate": 50, "unit": "kPa"},
    "temperature": {"id": 4, "type": SensorType.TEMPERATURE, "rate": 50, "unit": "C"},
}

# ==================== Binary Parsers ====================


def parse_fast_data(filepath: Path) -> List[Tuple[int, int, float]]:
    """
    Parse fast_data.bin file.

    Args:
        filepath: Path to fast_data.bin

    Returns:
        List of (timestamp_ms, sensor_id, value) tuples
    """
    records = []
    with open(filepath, "rb") as f:
        while True:
            data = f.read(FAST_RECORD_SIZE)
            if len(data) < FAST_RECORD_SIZE:
                break
            timestamp_ms, sensor_id, value = struct.unpack(FAST_RECORD_FORMAT, data)
            records.append((timestamp_ms, sensor_id, value))
    return records


def parse_medium_data(filepath: Path) -> List[Tuple[int, float]]:
    """
    Parse medium_data.bin file.

    Args:
        filepath: Path to medium_data.bin

    Returns:
        List of (timestamp_ms, current) tuples
    """
    records = []
    with open(filepath, "rb") as f:
        while True:
            data = f.read(MEDIUM_RECORD_SIZE)
            if len(data) < MEDIUM_RECORD_SIZE:
                break
            timestamp_ms, current = struct.unpack(MEDIUM_RECORD_FORMAT, data)
            records.append((timestamp_ms, current))
    return records


def parse_slow_data(filepath: Path) -> List[Tuple[int, int, float]]:
    """
    Parse slow_data.bin file.

    Args:
        filepath: Path to slow_data.bin

    Returns:
        List of (timestamp_ms, sensor_id, value) tuples
    """
    records = []
    with open(filepath, "rb") as f:
        while True:
            data = f.read(SLOW_RECORD_SIZE)
            if len(data) < SLOW_RECORD_SIZE:
                break
            timestamp_ms, sensor_id, value = struct.unpack(SLOW_RECORD_FORMAT, data)
            records.append((timestamp_ms, sensor_id, value))
    return records


# ==================== Data Processing ====================


def split_by_sensor(
    records: List[Tuple[int, int, float]], sensor_ids: List[int]
) -> Dict[str, pd.DataFrame]:
    """
    Split records by sensor ID into separate DataFrames.

    Args:
        records: List of (timestamp_ms, sensor_id, value) tuples
        sensor_ids: List of expected sensor IDs in this data tier

    Returns:
        Dictionary mapping sensor name to DataFrame with timestamp_ms and value columns
    """
    sensor_data: Dict[str, List[Tuple[int, float]]] = {
        SENSOR_ID_TO_NAME[sid]: [] for sid in sensor_ids
    }

    for timestamp_ms, sensor_id, value in records:
        if sensor_id in SENSOR_ID_TO_NAME:
            sensor_name = SENSOR_ID_TO_NAME[sensor_id]
            if sensor_name in sensor_data:
                sensor_data[sensor_name].append((timestamp_ms, value))

    result = {}
    for sensor_name, data in sensor_data.items():
        if data:
            df = pd.DataFrame(data, columns=["timestamp_ms", "value"])
            result[sensor_name] = df

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

    # Process fast data (IMU + Vibration)
    fast_path = run_dir / "fast_data.bin"
    if fast_path.exists():
        if verbose:
            print(f"  Processing fast_data.bin...")
        records = parse_fast_data(fast_path)
        if verbose:
            print(f"    Read {len(records)} records")
        sensor_dfs = split_by_sensor(records, [0, 1])  # IMU=0, Vibration=1
        all_dataframes.update(sensor_dfs)

    # Process medium data (Current)
    medium_path = run_dir / "medium_data.bin"
    if medium_path.exists():
        if verbose:
            print(f"  Processing medium_data.bin...")
        records = parse_medium_data(medium_path)
        if verbose:
            print(f"    Read {len(records)} records")
        if records:
            df = pd.DataFrame(records, columns=["timestamp_ms", "value"])
            all_dataframes["current"] = df

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
                print(f"  meta.json: run_id={meta.run_id}, duration={meta.statistics.duration_ms}ms")
        except Exception as e:
            results["errors"].append(f"Could not parse meta.json: {e}")
            results["valid"] = False
    else:
        results["metadata"] = {"exists": False}
        results["warnings"].append("meta.json not found")

    return results
