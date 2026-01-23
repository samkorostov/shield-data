"""Tests for binary parsing and conversion."""

import struct
from pathlib import Path

import pandas as pd
import pytest

from shield_converter.converter import (
    FAST_RECORD_SIZE,
    MEDIUM_RECORD_SIZE,
    SLOW_RECORD_SIZE,
    convert_run,
    load_firmware_metadata,
    parse_fast_data,
    parse_folder_name,
    parse_medium_data,
    parse_slow_data,
    split_by_sensor,
    validate_run,
)
from shield_converter.models import HealthLabel


class TestRecordSizes:
    """Verify struct sizes match firmware expectations."""

    def test_fast_record_size(self):
        # uint32 + uint8 + 3 padding + float = 4 + 1 + 3 + 4 = 12
        assert FAST_RECORD_SIZE == 12

    def test_medium_record_size(self):
        # uint32 + float = 4 + 4 = 8
        assert MEDIUM_RECORD_SIZE == 8

    def test_slow_record_size(self):
        # uint32 + uint8 + 3 padding + float = 4 + 1 + 3 + 4 = 12
        assert SLOW_RECORD_SIZE == 12


class TestParseFastData:
    def test_parse_fast_data(self, tmp_path, fast_data_bytes):
        filepath = tmp_path / "fast_data.bin"
        filepath.write_bytes(fast_data_bytes)

        records = parse_fast_data(filepath)

        assert len(records) == 200  # 100 IMU + 100 Vibration
        # Check first IMU record
        assert records[0] == (0, 0, pytest.approx(0.5, rel=1e-5))
        # Check first Vibration record
        assert records[1] == (0, 1, pytest.approx(0.0, rel=1e-5))

    def test_parse_empty_file(self, tmp_path):
        filepath = tmp_path / "empty.bin"
        filepath.write_bytes(b"")

        records = parse_fast_data(filepath)
        assert records == []

    def test_parse_partial_record(self, tmp_path):
        """Partial records at end of file should be ignored."""
        filepath = tmp_path / "partial.bin"
        # Write one complete record + partial data
        data = struct.pack("<I B 3x f", 0, 0, 1.0) + b"\x00\x00"
        filepath.write_bytes(data)

        records = parse_fast_data(filepath)
        assert len(records) == 1


class TestParseMediumData:
    def test_parse_medium_data(self, tmp_path, medium_data_bytes):
        filepath = tmp_path / "medium_data.bin"
        filepath.write_bytes(medium_data_bytes)

        records = parse_medium_data(filepath)

        assert len(records) == 50
        # Check first record
        assert records[0] == (0, pytest.approx(1.5, rel=1e-5))
        # Check timestamp progression
        assert records[1][0] == 5
        assert records[2][0] == 10


class TestParseSlowData:
    def test_parse_slow_data(self, tmp_path, slow_data_bytes):
        filepath = tmp_path / "slow_data.bin"
        filepath.write_bytes(slow_data_bytes)

        records = parse_slow_data(filepath)

        assert len(records) == 40  # 20 Pressure + 20 Temperature
        # Check first Pressure record (sensor_id=3)
        assert records[0][1] == 3
        assert records[0][2] == pytest.approx(101.325, rel=1e-4)
        # Check first Temperature record (sensor_id=4)
        assert records[1][1] == 4
        assert records[1][2] == pytest.approx(25.0, rel=1e-5)


class TestSplitBySensor:
    def test_split_fast_data(self, tmp_path, fast_data_bytes):
        filepath = tmp_path / "fast_data.bin"
        filepath.write_bytes(fast_data_bytes)
        records = parse_fast_data(filepath)

        sensor_dfs = split_by_sensor(records, [0, 1])

        assert "imu" in sensor_dfs
        assert "vibration" in sensor_dfs
        assert len(sensor_dfs["imu"]) == 100
        assert len(sensor_dfs["vibration"]) == 100
        assert list(sensor_dfs["imu"].columns) == ["timestamp_ms", "value"]

    def test_split_slow_data(self, tmp_path, slow_data_bytes):
        filepath = tmp_path / "slow_data.bin"
        filepath.write_bytes(slow_data_bytes)
        records = parse_slow_data(filepath)

        sensor_dfs = split_by_sensor(records, [3, 4])

        assert "pressure" in sensor_dfs
        assert "temperature" in sensor_dfs
        assert len(sensor_dfs["pressure"]) == 20
        assert len(sensor_dfs["temperature"]) == 20


class TestParseFolderName:
    def test_unit_and_run(self):
        unit_id, run_id = parse_folder_name("UNIT_001_RUN_001")
        assert unit_id == "unit_0001"
        assert run_id == "RUN_001"

    def test_large_numbers(self):
        unit_id, run_id = parse_folder_name("UNIT_42_RUN_123")
        assert unit_id == "unit_0042"
        assert run_id == "RUN_123"

    def test_run_only(self):
        unit_id, run_id = parse_folder_name("RUN_005")
        assert unit_id == "unit_0001"
        assert run_id == "RUN_005"

    def test_case_insensitive(self):
        unit_id, run_id = parse_folder_name("unit_001_run_001")
        assert unit_id == "unit_0001"
        assert run_id == "RUN_001"

    def test_unknown_format(self):
        unit_id, run_id = parse_folder_name("some_random_folder")
        assert unit_id == "unit_0001"
        assert run_id == "some_random_folder"


class TestLoadFirmwareMetadata:
    def test_load_metadata(self, run_directory):
        meta = load_firmware_metadata(run_directory / "meta.json")
        assert meta.run_id == "RUN_001"
        assert meta.device_info.chip == "ESP32-S3"

    def test_missing_metadata(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_firmware_metadata(tmp_path / "nonexistent.json")


class TestConvertRun:
    def test_convert_with_metadata(self, run_directory, output_directory):
        output_files, sessions = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
        )

        assert len(output_files) == 5
        assert "imu" in output_files
        assert "vibration" in output_files
        assert "current" in output_files
        assert "pressure" in output_files
        assert "temperature" in output_files

        # Check CSV files exist and have correct content
        imu_df = pd.read_csv(output_files["imu"])
        assert len(imu_df) == 100
        assert list(imu_df.columns) == ["timestamp_ms", "value"]

        # Check session metadata
        assert len(sessions) == 5
        imu_session = next(s for s in sessions if s.sensor_name == "imu")
        assert imu_session.sampling_rate_hz == 1000
        assert imu_session.units == "m/s^2"

    def test_convert_without_metadata(self, run_directory_no_metadata, output_directory):
        output_files, sessions = convert_run(
            run_dir=run_directory_no_metadata,
            output_dir=output_directory,
        )

        assert len(output_files) == 5
        assert len(sessions) == 5

    def test_convert_with_health_label(self, run_directory, output_directory):
        _, sessions = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            health_label=HealthLabel.HEALTHY,
        )

        for session in sessions:
            assert session.health_label == HealthLabel.HEALTHY

    def test_convert_with_unit_override(self, run_directory, output_directory):
        _, sessions = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            unit_id="unit_9999",
        )

        for session in sessions:
            assert session.unit_id == "unit_9999"

    def test_output_directory_structure(self, run_directory, output_directory):
        convert_run(run_dir=run_directory, output_dir=output_directory)

        # Check directory structure
        data_dir = output_directory / "data" / "UNIT_0001_RUN_001"
        assert data_dir.exists()
        assert (data_dir / "imu.csv").exists()
        assert (data_dir / "current.csv").exists()


class TestValidateRun:
    def test_validate_complete_run(self, run_directory):
        results = validate_run(run_directory)

        assert results["valid"] is True
        assert len(results["errors"]) == 0
        assert results["files"]["fast_data.bin"]["exists"] is True
        assert results["files"]["medium_data.bin"]["exists"] is True
        assert results["files"]["slow_data.bin"]["exists"] is True
        assert results["metadata"]["exists"] is True

    def test_validate_missing_metadata(self, run_directory_no_metadata):
        results = validate_run(run_directory_no_metadata)

        assert results["metadata"]["exists"] is False
        assert any("meta.json" in w for w in results["warnings"])

    def test_validate_empty_directory(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = validate_run(empty_dir)

        assert results["files"]["fast_data.bin"]["exists"] is False
        assert results["files"]["medium_data.bin"]["exists"] is False
        assert results["files"]["slow_data.bin"]["exists"] is False
