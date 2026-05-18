"""Tests for binary parsing and conversion."""

import struct
from pathlib import Path

import pandas as pd
import pytest

from shield_converter.converter import (
    FAST_RECORD_SIZE,
    MEDIUM_RECORD_SIZE,
    SLOW_RECORD_SIZE,
    convert_all_runs,
    convert_run,
    load_firmware_metadata,
    parse_fast_data,
    parse_folder_name,
    parse_medium_data,
    parse_slow_data,
    split_by_sensor,
    split_fast_by_sensor,
    validate_run,
)
from shield_converter.models import HealthLabel


MS_PER_HOUR = 3_600_000


def pack_v2_record(
    timestamp_ms: int,
    sensor_id: int,
    kind: int,
    axis_count: int,
    values: tuple[float, float, float],
) -> bytes:
    return struct.pack(
        "<I B B B B 3f",
        timestamp_ms,
        sensor_id,
        kind,
        axis_count,
        0,
        *values,
    )


class TestRecordSizes:
    """Verify struct sizes match firmware expectations."""

    def test_fast_record_size(self):
        # uint32 + sensor/kind/axis/flags uint8s + 3 floats = 4 + 4 + 12 = 20
        assert FAST_RECORD_SIZE == 20

    def test_medium_record_size(self):
        # All v2 binary files share the same packed record format.
        assert MEDIUM_RECORD_SIZE == 20

    def test_slow_record_size(self):
        # All v2 binary files share the same packed record format.
        assert SLOW_RECORD_SIZE == 20


class TestParseFastData:
    def test_parse_fast_data(self, tmp_path, fast_data_bytes):
        filepath = tmp_path / "fast_data.bin"
        filepath.write_bytes(fast_data_bytes)

        records = parse_fast_data(filepath)

        assert len(records) == 1000  # 100 samples * 5 sensors * raw+processed
        # Check first Vibration raw record (scalar, data[0] only)
        record = records[0]
        assert record["timestamp_ms"] == 0
        assert record["sensor_id"] == 1
        assert record["kind"] == 0
        assert record["axis_count"] == 1
        assert record["flags"] == 0
        assert record["data"][0] == pytest.approx(0.0, rel=1e-5)
        assert record["data"][1] == pytest.approx(0.0)
        assert record["data"][2] == pytest.approx(0.0)
        # Check paired processed Vibration record.
        record = records[1]
        assert record["timestamp_ms"] == 0
        assert record["sensor_id"] == 1
        assert record["kind"] == 1
        assert record["flags"] == 0x02
        assert record["data"][0] == pytest.approx(0.25, rel=1e-5)

    def test_parse_empty_file(self, tmp_path):
        filepath = tmp_path / "empty.bin"
        filepath.write_bytes(b"")

        records = parse_fast_data(filepath)
        assert len(records) == 0

    def test_parse_partial_record(self, tmp_path):
        """Partial records at end of file should be ignored."""
        filepath = tmp_path / "partial.bin"
        data = struct.pack("<I B B B B 3f", 0, 7, 0, 3, 0, 1.0, 2.0, 3.0) + b"\x00\x00"
        filepath.write_bytes(data)

        records = parse_fast_data(filepath)
        assert len(records) == 1


class TestParseMediumData:
    def test_parse_medium_data(self, tmp_path, medium_data_bytes):
        filepath = tmp_path / "medium_data.bin"
        filepath.write_bytes(medium_data_bytes)

        records = parse_medium_data(filepath)

        assert len(records) == 200  # 50 samples * 2 sensors * raw+processed
        # Check first Current raw and processed records.
        assert records[0]["timestamp_ms"] == 0
        assert records[0]["sensor_id"] == 2
        assert records[0]["kind"] == 0
        assert records[0]["data"][0] == pytest.approx(1.5, rel=1e-5)
        assert records[1]["timestamp_ms"] == 0
        assert records[1]["sensor_id"] == 2
        assert records[1]["kind"] == 1
        assert records[1]["data"][0] == pytest.approx(1.6, rel=1e-5)
        # Check timestamp progression
        assert records[4]["timestamp_ms"] == 5
        assert records[8]["timestamp_ms"] == 10


class TestParseSlowData:
    def test_parse_slow_data(self, tmp_path, slow_data_bytes):
        filepath = tmp_path / "slow_data.bin"
        filepath.write_bytes(slow_data_bytes)

        records = parse_slow_data(filepath)

        assert len(records) == 80  # 20 samples * 2 sensors * raw+processed
        # Check first Pressure raw and first Temperature raw records.
        assert records[0]["sensor_id"] == 3
        assert records[0]["kind"] == 0
        assert records[0]["data"][0] == pytest.approx(101.325, rel=1e-4)
        assert records[2]["sensor_id"] == 4
        assert records[2]["kind"] == 0
        assert records[2]["data"][0] == pytest.approx(25.0, rel=1e-5)


class TestSplitBySensor:
    def test_split_fast_data(self, tmp_path, fast_data_bytes):
        filepath = tmp_path / "fast_data.bin"
        filepath.write_bytes(fast_data_bytes)
        records = parse_fast_data(filepath)

        sensor_dfs = split_fast_by_sensor(records, [1, 7])

        assert "vibration" in sensor_dfs
        assert "proc_vibration" in sensor_dfs
        assert "magnetometer" in sensor_dfs
        assert "proc_magnetometer" in sensor_dfs
        assert len(sensor_dfs["vibration"]) == 100
        assert len(sensor_dfs["proc_vibration"]) == 100
        assert len(sensor_dfs["magnetometer"]) == 100
        # Magnetometer is 3-axis
        assert list(sensor_dfs["magnetometer"].columns) == ["timestamp_ms", "x", "y", "z"]
        assert sensor_dfs["magnetometer"].iloc[0]["x"] == pytest.approx(0.5, rel=1e-5)
        assert sensor_dfs["magnetometer"].iloc[0]["y"] == pytest.approx(0.1, rel=1e-5)
        assert sensor_dfs["magnetometer"].iloc[0]["z"] == pytest.approx(-0.3, rel=1e-5)
        # Vibration is scalar
        assert list(sensor_dfs["vibration"].columns) == ["timestamp_ms", "value"]
        assert list(sensor_dfs["proc_vibration"].columns) == ["timestamp_ms", "value"]
        assert sensor_dfs["proc_vibration"].iloc[0]["value"] == pytest.approx(0.25)

    def test_split_slow_data(self, tmp_path, slow_data_bytes):
        filepath = tmp_path / "slow_data.bin"
        filepath.write_bytes(slow_data_bytes)
        records = parse_slow_data(filepath)

        sensor_dfs = split_by_sensor(records, [3, 4])

        assert "pressure" in sensor_dfs
        assert "proc_pressure" in sensor_dfs
        assert "temperature" in sensor_dfs
        assert "proc_temperature" in sensor_dfs
        assert len(sensor_dfs["pressure"]) == 20
        assert len(sensor_dfs["proc_pressure"]) == 20
        assert len(sensor_dfs["temperature"]) == 20
        assert len(sensor_dfs["proc_temperature"]) == 20


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

        assert len(output_files) == 18
        assert "vibration" in output_files
        assert "proc_vibration" in output_files
        assert "current" in output_files
        assert "proc_current" in output_files
        assert "photodiode" in output_files
        assert "proc_photodiode" in output_files
        assert "pressure" in output_files
        assert "proc_pressure" in output_files
        assert "temperature" in output_files
        assert "proc_temperature" in output_files
        assert "accelerometer" in output_files
        assert "proc_accelerometer" in output_files

        # Check CSV files exist and have correct content
        accelerometer_df = pd.read_csv(output_files["accelerometer"])
        assert len(accelerometer_df) == 100
        assert list(accelerometer_df.columns) == ["timestamp_ms", "x", "y", "z"]

        # Check scalar sensor CSV
        vibration_df = pd.read_csv(output_files["vibration"])
        assert list(vibration_df.columns) == ["timestamp_ms", "value"]
        vibration_processed_df = pd.read_csv(output_files["proc_vibration"])
        assert list(vibration_processed_df.columns) == ["timestamp_ms", "value"]
        assert vibration_processed_df.iloc[0]["value"] == pytest.approx(0.25)

        # Check session metadata
        assert len(sessions) == 18
        vibration_session = next(s for s in sessions if s.sensor_name == "vibration")
        assert vibration_session.duration_s == pytest.approx(0.099)
        vibration_processed_session = next(
            s for s in sessions if s.sensor_name == "proc_vibration"
        )
        assert vibration_processed_session.sampling_rate_hz == 1000
        assert vibration_processed_session.units == "binary"
        assert vibration_processed_session.duration_s == pytest.approx(0.099)

    def test_convert_without_metadata(self, run_directory_no_metadata, output_directory):
        output_files, sessions = convert_run(
            run_dir=run_directory_no_metadata,
            output_dir=output_directory,
        )

        assert len(output_files) == 18
        assert len(sessions) == 18

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
        assert (data_dir / "vibration.csv").exists()
        assert (data_dir / "proc_vibration.csv").exists()
        assert (data_dir / "current.csv").exists()
        assert (data_dir / "proc_current.csv").exists()
        assert (data_dir / "photodiode.csv").exists()
        assert (data_dir / "accelerometer.csv").exists()
        assert (data_dir / "proc_accelerometer.csv").exists()

    def test_convert_with_end_hour_window(self, run_directory, output_directory):
        output_files, sessions = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            end_hour=50 / MS_PER_HOUR,
        )

        vibration_df = pd.read_csv(output_files["vibration"])
        current_df = pd.read_csv(output_files["current"])
        pressure_df = pd.read_csv(output_files["pressure"])

        assert len(vibration_df) == 50
        assert vibration_df["timestamp_ms"].min() == 0
        assert vibration_df["timestamp_ms"].max() == 49
        assert len(current_df) == 10
        assert current_df["timestamp_ms"].max() == 45
        assert len(pressure_df) == 3
        assert pressure_df["timestamp_ms"].tolist() == [0, 20, 40]

        vibration_session = next(s for s in sessions if s.sensor_name == "vibration")
        assert vibration_session.duration_s == pytest.approx(0.049)

    def test_convert_with_middle_window(self, run_directory, output_directory):
        output_files, _ = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            start_hour=25 / MS_PER_HOUR,
            end_hour=60 / MS_PER_HOUR,
        )

        vibration_df = pd.read_csv(output_files["vibration"])
        current_df = pd.read_csv(output_files["current"])

        assert len(vibration_df) == 35
        assert vibration_df["timestamp_ms"].min() == 25
        assert vibration_df["timestamp_ms"].max() == 59
        assert current_df["timestamp_ms"].tolist() == [25, 30, 35, 40, 45, 50, 55]

    def test_convert_with_start_hour_only(self, run_directory, output_directory):
        output_files, _ = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            start_hour=95 / MS_PER_HOUR,
        )

        vibration_df = pd.read_csv(output_files["vibration"])

        assert vibration_df["timestamp_ms"].tolist() == [95, 96, 97, 98, 99]

    def test_time_window_is_relative_to_run_start(self, tmp_path, output_directory):
        run_dir = tmp_path / "RUN_003"
        run_dir.mkdir()
        (run_dir / "fast_data.bin").write_bytes(
            pack_v2_record(1000, 1, 0, 1, (1.0, 0.0, 0.0))
            + pack_v2_record(1000, 1, 1, 1, (1.5, 0.0, 0.0))
            + pack_v2_record(2000, 1, 0, 1, (2.0, 0.0, 0.0))
            + pack_v2_record(2000, 1, 1, 1, (2.5, 0.0, 0.0))
        )

        output_files, _ = convert_run(
            run_dir=run_dir,
            output_dir=output_directory,
            end_hour=1 / MS_PER_HOUR,
        )

        vibration_df = pd.read_csv(output_files["vibration"])
        assert vibration_df["timestamp_ms"].tolist() == [1000]

    def test_time_window_clears_stale_csvs(self, run_directory, output_directory, capsys):
        convert_run(run_dir=run_directory, output_dir=output_directory)
        data_dir = output_directory / "data" / "UNIT_0001_RUN_001"
        assert (data_dir / "vibration.csv").exists()

        output_files, sessions = convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            start_hour=1000 / MS_PER_HOUR,
            end_hour=2000 / MS_PER_HOUR,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert output_files == {}
        assert sessions == []
        assert not (data_dir / "vibration.csv").exists()
        assert "vibration has no data in requested" in captured.out

    def test_time_window_warns_when_sensor_ends_before_requested_window(
        self, run_directory, output_directory, capsys
    ):
        convert_run(
            run_dir=run_directory,
            output_dir=output_directory,
            end_hour=500 / MS_PER_HOUR,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "vibration only covers" in captured.out
        assert "requested" in captured.out

    def test_invalid_time_window(self, run_directory, output_directory):
        with pytest.raises(ValueError, match="start_hour must be >= 0"):
            convert_run(
                run_dir=run_directory,
                output_dir=output_directory,
                start_hour=-1,
            )

        with pytest.raises(ValueError, match="end_hour must be greater"):
            convert_run(
                run_dir=run_directory,
                output_dir=output_directory,
                start_hour=1,
                end_hour=1,
            )

    def test_convert_all_passes_time_window(self, run_directory, output_directory):
        sessions = convert_all_runs(
            input_dir=run_directory.parent,
            output_dir=output_directory,
            end_hour=50 / MS_PER_HOUR,
        )

        data_dir = output_directory / "data" / "UNIT_0001_RUN_001"
        vibration_df = pd.read_csv(data_dir / "vibration.csv")

        assert len(vibration_df) == 50
        assert any(s.sensor_name == "vibration" for s in sessions)


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
