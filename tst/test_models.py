"""Tests for Pydantic models."""

from datetime import datetime, timezone

import pytest

from shield_converter.models import (
    DeviceInfo,
    FirmwareMetadata,
    HealthLabel,
    SensorInfo,
    SensorType,
    SessionRecord,
)


class TestSensorType:
    def test_enum_values(self):
        assert SensorType.IMU == "IMU"
        assert SensorType.VIBRATION == "VIBRATION"
        assert SensorType.CURRENT == "CURRENT"
        assert SensorType.PRESSURE == "PRESSURE"
        assert SensorType.TEMPERATURE == "TEMPERATURE"

    def test_enum_from_string(self):
        assert SensorType("IMU") == SensorType.IMU
        assert SensorType("CURRENT") == SensorType.CURRENT


class TestHealthLabel:
    def test_enum_values(self):
        assert HealthLabel.UNKNOWN == "unknown"
        assert HealthLabel.HEALTHY == "healthy"
        assert HealthLabel.DEGRADED == "degraded"
        assert HealthLabel.FAULTY == "faulty"

    def test_default_is_unknown(self):
        record = SessionRecord(
            session_id="RUN_001",
            unit_id="unit_0001",
            sensor_name="imu",
            file_name="imu.csv",
            start_time_utc=datetime.now(tz=timezone.utc),
            duration_s=600.0,
            sampling_rate_hz=1000,
            units="m/s^2",
        )
        assert record.health_label == HealthLabel.UNKNOWN


class TestSensorInfo:
    def test_create_sensor_info(self):
        info = SensorInfo(
            id=0,
            name="BNO085_IMU",
            type=SensorType.IMU,
            rate=1000,
            unit="m/s^2",
        )
        assert info.id == 0
        assert info.name == "BNO085_IMU"
        assert info.type == SensorType.IMU
        assert info.rate == 1000
        assert info.unit == "m/s^2"

    def test_sensor_info_from_dict(self):
        data = {"id": 2, "name": "ACS723_Current", "type": "CURRENT", "rate": 200, "unit": "A"}
        info = SensorInfo.model_validate(data)
        assert info.type == SensorType.CURRENT


class TestDeviceInfo:
    def test_create_device_info(self):
        info = DeviceInfo(
            chip="ESP32-S3",
            cores=2,
            revision=1,
            firmware_version="0.1.0",
            idf_version="5.1.0",
        )
        assert info.chip == "ESP32-S3"
        assert info.cores == 2


class TestFirmwareMetadata:
    def test_parse_metadata(self, sample_metadata):
        meta = FirmwareMetadata.model_validate(sample_metadata)
        assert meta.run_id == "RUN_001"
        assert meta.device_info.chip == "ESP32-S3"
        assert len(meta.sensors["fast"]) == 2
        assert len(meta.sensors["medium"]) == 1
        assert len(meta.sensors["slow"]) == 2
        assert meta.statistics.duration_ms == 600000

    def test_optional_end_time(self, sample_metadata):
        del sample_metadata["end_time"]
        meta = FirmwareMetadata.model_validate(sample_metadata)
        assert meta.end_time is None


class TestSessionRecord:
    def test_create_session_record(self):
        now = datetime.now(tz=timezone.utc)
        record = SessionRecord(
            session_id="RUN_001",
            unit_id="unit_0001",
            sensor_name="imu",
            file_name="imu.csv",
            start_time_utc=now,
            duration_s=600.0,
            sampling_rate_hz=1000,
            units="m/s^2",
            health_label=HealthLabel.HEALTHY,
        )
        assert record.session_id == "RUN_001"
        assert record.file_format == "csv"
        assert record.health_label == HealthLabel.HEALTHY

    def test_to_csv_row(self):
        now = datetime(2024, 1, 22, 10, 30, 0, tzinfo=timezone.utc)
        record = SessionRecord(
            session_id="RUN_001",
            unit_id="unit_0001",
            sensor_name="imu",
            file_name="imu.csv",
            start_time_utc=now,
            duration_s=600.0,
            sampling_rate_hz=1000,
            units="m/s^2",
        )
        row = record.to_csv_row()
        assert row["session_id"] == "RUN_001"
        assert row["unit_id"] == "unit_0001"
        assert row["sensor_name"] == "imu"
        assert row["file_format"] == "csv"
        assert row["sampling_rate_hz"] == 1000
        assert row["health_label"] == "unknown"
        assert "2024-01-22" in row["start_time_utc"]

    def test_optional_fields_default_to_none(self):
        record = SessionRecord(
            session_id="RUN_001",
            unit_id="unit_0001",
            sensor_name="current",
            file_name="current.csv",
            start_time_utc=datetime.now(tz=timezone.utc),
            duration_s=600.0,
            sampling_rate_hz=200,
            units="A",
        )
        assert record.voltage_supply_v is None
        assert record.temperature_c is None
        assert record.humidity_pct is None
