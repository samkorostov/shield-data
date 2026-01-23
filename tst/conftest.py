"""
Pytest fixtures for shield_converter tests.

Provides synthetic binary data and temporary directories for testing.
"""

import json
import struct
from pathlib import Path

import pytest


@pytest.fixture
def fast_data_bytes() -> bytes:
    """Generate synthetic fast_data.bin content (IMU + Vibration)."""
    data = b""
    for i in range(100):
        # IMU sample (sensor_id=0)
        data += struct.pack("<I B 3x f", i, 0, 0.5 + i * 0.01)
        # Vibration sample (sensor_id=1)
        data += struct.pack("<I B 3x f", i, 1, float(i % 2))
    return data


@pytest.fixture
def medium_data_bytes() -> bytes:
    """Generate synthetic medium_data.bin content (Current)."""
    data = b""
    for i in range(50):
        data += struct.pack("<I f", i * 5, 1.5 + i * 0.02)
    return data


@pytest.fixture
def slow_data_bytes() -> bytes:
    """Generate synthetic slow_data.bin content (Pressure + Temperature)."""
    data = b""
    for i in range(20):
        # Pressure sample (sensor_id=3)
        data += struct.pack("<I B 3x f", i * 20, 3, 101.325 + i * 0.1)
        # Temperature sample (sensor_id=4)
        data += struct.pack("<I B 3x f", i * 20, 4, 25.0 + i * 0.5)
    return data


@pytest.fixture
def sample_metadata() -> dict:
    """Generate sample meta.json content."""
    return {
        "run_id": "RUN_001",
        "start_time": "1705920600",
        "end_time": "1705921200",
        "device_info": {
            "chip": "ESP32-S3",
            "cores": 2,
            "revision": 1,
            "firmware_version": "0.1.0",
            "idf_version": "5.1.0",
        },
        "sensors": {
            "fast": [
                {"id": 0, "name": "BNO085_IMU", "type": "IMU", "rate": 1000, "unit": "m/s^2"},
                {"id": 1, "name": "SW420_Vibration", "type": "VIBRATION", "rate": 1000, "unit": "binary"},
            ],
            "medium": [
                {"id": 2, "name": "ACS723_Current", "type": "CURRENT", "rate": 200, "unit": "A"},
            ],
            "slow": [
                {"id": 3, "name": "MPL3115_Pressure", "type": "PRESSURE", "rate": 50, "unit": "kPa"},
                {"id": 4, "name": "MCP9808_Temp", "type": "TEMPERATURE", "rate": 50, "unit": "C"},
            ],
        },
        "data_files": {
            "fast": "fast_data.bin",
            "medium": "medium_data.bin",
            "slow": "slow_data.bin",
        },
        "statistics": {
            "total_samples": {"fast": 200, "medium": 50, "slow": 40},
            "duration_ms": 600000,
            "queue_overruns": 0,
            "sd_write_errors": 0,
        },
    }


@pytest.fixture
def run_directory(tmp_path, fast_data_bytes, medium_data_bytes, slow_data_bytes, sample_metadata):
    """Create a complete run directory with all binary files and metadata."""
    run_dir = tmp_path / "UNIT_001_RUN_001"
    run_dir.mkdir()

    (run_dir / "fast_data.bin").write_bytes(fast_data_bytes)
    (run_dir / "medium_data.bin").write_bytes(medium_data_bytes)
    (run_dir / "slow_data.bin").write_bytes(slow_data_bytes)
    (run_dir / "meta.json").write_text(json.dumps(sample_metadata, indent=2))

    return run_dir


@pytest.fixture
def run_directory_no_metadata(tmp_path, fast_data_bytes, medium_data_bytes, slow_data_bytes):
    """Create a run directory without meta.json."""
    run_dir = tmp_path / "RUN_002"
    run_dir.mkdir()

    (run_dir / "fast_data.bin").write_bytes(fast_data_bytes)
    (run_dir / "medium_data.bin").write_bytes(medium_data_bytes)
    (run_dir / "slow_data.bin").write_bytes(slow_data_bytes)

    return run_dir


@pytest.fixture
def output_directory(tmp_path):
    """Create an empty output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
