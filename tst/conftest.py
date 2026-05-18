"""
Pytest fixtures for shield_converter tests.

Provides synthetic binary data and temporary directories for testing.
"""

import json
import struct
from pathlib import Path

import pytest


THREE_AXIS_SENSOR_IDS = {7, 8, 9}


def make_sensor_record(
    timestamp_ms: int,
    sensor_id: int,
    kind: int,
    values: tuple[float, ...],
    flags: int = 0,
) -> bytes:
    """Pack one sensor_data_record_v2_t record."""
    axis_count = 3 if sensor_id in THREE_AXIS_SENSOR_IDS else 1
    padded_values = values + (0.0,) * (3 - len(values))
    return struct.pack(
        "<I B B B B 3f",
        timestamp_ms,
        sensor_id,
        kind,
        axis_count,
        flags,
        *padded_values[:3],
    )


@pytest.fixture
def fast_data_bytes() -> bytes:
    """Generate synthetic fast_data.bin v2 content."""
    data = b""
    for i in range(100):
        vibration = (float(i % 2),)
        microphone = (0.25 + i * 0.01,)
        magnetometer = (0.5 + i * 0.01, 0.1 + i * 0.01, -0.3 + i * 0.01)
        gyroscope = (1.0 + i * 0.02, 2.0 + i * 0.02, 3.0 + i * 0.02)
        accelerometer = (0.0 + i * 0.01, 0.1 + i * 0.01, 9.8 + i * 0.01)

        for sensor_id, raw_values, processed_values in [
            (1, vibration, (vibration[0] + 0.25,)),
            (5, microphone, (microphone[0] * 2.0,)),
            (7, magnetometer, tuple(v + 0.001 for v in magnetometer)),
            (8, gyroscope, tuple(v * 0.5 for v in gyroscope)),
            (9, accelerometer, tuple(v + 0.01 for v in accelerometer)),
        ]:
            data += make_sensor_record(i, sensor_id, 0, raw_values)
            data += make_sensor_record(i, sensor_id, 1, processed_values, flags=0x02)
    return data


@pytest.fixture
def medium_data_bytes() -> bytes:
    """Generate synthetic medium_data.bin v2 content."""
    data = b""
    for i in range(50):
        current = 1.5 + i * 0.02
        photodiode = 0.8 + i * 0.01
        data += make_sensor_record(i * 5, 2, 0, (current,))
        data += make_sensor_record(i * 5, 2, 1, (current + 0.1,), flags=0x02)
        data += make_sensor_record(i * 5, 6, 0, (photodiode,))
        data += make_sensor_record(i * 5, 6, 1, (photodiode,), flags=0x01)
    return data


@pytest.fixture
def slow_data_bytes() -> bytes:
    """Generate synthetic slow_data.bin v2 content."""
    data = b""
    for i in range(20):
        pressure = 101.325 + i * 0.1
        temperature = 25.0 + i * 0.5
        data += make_sensor_record(i * 20, 3, 0, (pressure,))
        data += make_sensor_record(i * 20, 3, 1, (pressure + 0.05,), flags=0x02)
        data += make_sensor_record(i * 20, 4, 0, (temperature,))
        data += make_sensor_record(i * 20, 4, 1, (temperature,), flags=0x01)
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
                {
                    "id": 1,
                    "name": "SW420_Vibration",
                    "type": "VIBRATION",
                    "rate": 1000,
                    "unit": "binary",
                },
                {
                    "id": 5,
                    "name": "Microphone",
                    "type": "MICROPHONE",
                    "rate": 1000,
                    "unit": "dBFS",
                },
                {
                    "id": 7,
                    "name": "Magnetometer",
                    "type": "MAGNETOMETER",
                    "rate": 1000,
                    "unit": "uT",
                },
                {
                    "id": 8,
                    "name": "Gyroscope",
                    "type": "GYROSCOPE",
                    "rate": 1000,
                    "unit": "rad/s",
                },
                {
                    "id": 9,
                    "name": "Accelerometer",
                    "type": "ACCELEROMETER",
                    "rate": 1000,
                    "unit": "m/s^2",
                },
            ],
            "medium": [
                {"id": 2, "name": "ACS723_Current", "type": "CURRENT", "rate": 200, "unit": "A"},
                {"id": 6, "name": "Photodiode", "type": "PHOTODIODE", "rate": 200, "unit": "V"},
            ],
            "slow": [
                {
                    "id": 3,
                    "name": "MPL3115_Pressure",
                    "type": "PRESSURE",
                    "rate": 50,
                    "unit": "kPa",
                },
                {"id": 4, "name": "MCP9808_Temp", "type": "TEMPERATURE", "rate": 50, "unit": "C"},
            ],
        },
        "data_files": {
            "fast": "fast_data.bin",
            "medium": "medium_data.bin",
            "slow": "slow_data.bin",
        },
        "statistics": {
            "total_samples": {"fast": 1000, "medium": 200, "slow": 80},
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
