"""
Pydantic models for ESP32 DAQ metadata and session records.

Defines both input models (firmware meta.json) and output models (ML training metadata).
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SensorType(str, Enum):
    """Sensor type enumeration matching firmware sensor IDs."""
    IMU = "IMU"
    VIBRATION = "VIBRATION"
    CURRENT = "CURRENT"
    PRESSURE = "PRESSURE"
    TEMPERATURE = "TEMPERATURE"
    MICROPHONE = "MICROPHONE"
    PHOTODIODE = "PHOTODIODE"
    MAGNETOMETER = "MAGNETOMETER"
    GYROSCOPE = "GYROSCOPE"
    ACCELEROMETER = "ACCELEROMETER"


class HealthLabel(str, Enum):
    """Health status labels for ML training."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"


# ==================== Firmware Metadata Models (Input) ====================


class SensorInfo(BaseModel):
    """Sensor configuration from firmware meta.json."""
    id: int = Field(description="Sensor ID (0-6)")
    name: str = Field(description="Sensor hardware name")
    type: SensorType = Field(description="Sensor type category")
    rate: int = Field(description="Sampling rate in Hz")
    unit: str = Field(description="Physical unit of measurement")


class DeviceInfo(BaseModel):
    """ESP32 device information from firmware."""
    chip: str = Field(description="Chip model (e.g., ESP32-S3)")
    cores: int = Field(description="Number of CPU cores")
    revision: int = Field(description="Chip revision")
    firmware_version: str = Field(description="Firmware version string")
    idf_version: str = Field(description="ESP-IDF version")


class DataFiles(BaseModel):
    """Binary data file paths."""
    fast: str = Field(description="Fast data file path (1kHz)")
    medium: str = Field(description="Medium data file path (200Hz)")
    slow: str = Field(description="Slow data file path (50Hz)")


class Statistics(BaseModel):
    """Data acquisition statistics from firmware."""
    total_samples: Dict[str, int] = Field(
        description="Sample counts per data tier (fast, medium, slow)"
    )
    duration_ms: int = Field(description="Total acquisition duration in milliseconds")
    queue_overruns: int = Field(description="Number of queue overflow events")
    sd_write_errors: int = Field(description="Number of SD card write errors")


class FirmwareMetadata(BaseModel):
    """
    Complete firmware metadata from meta.json.

    Created by ESP32 at the start of each data acquisition run.
    """
    run_id: str = Field(description="Unique run identifier")
    start_time: str = Field(description="Unix timestamp string of run start")
    end_time: Optional[str] = Field(default=None, description="Unix timestamp string of run end")
    device_info: DeviceInfo = Field(description="ESP32 device information")
    sensors: Dict[str, List[SensorInfo]] = Field(
        description="Sensors grouped by tier (fast, medium, slow)"
    )
    data_files: DataFiles = Field(description="Binary data file paths")
    statistics: Statistics = Field(description="Acquisition statistics")


# ==================== Session Metadata Models (Output) ====================


class SessionRecord(BaseModel):
    """
    Session record for ML training metadata.

    One record per sensor per session, stored in sessions.csv.
    """
    session_id: str = Field(description="Session/run identifier")
    unit_id: str = Field(description="Physical unit identifier (e.g., unit_0001)")
    sensor_name: str = Field(description="Sensor name (imu, vibration, current, pressure, temperature, microphone, photodiode)")
    file_name: str = Field(description="Output CSV file name")
    file_format: str = Field(default="csv", description="Output file format")
    start_time_utc: datetime = Field(description="Session start time in UTC")
    duration_s: float = Field(description="Session duration in seconds")
    sampling_rate_hz: int = Field(description="Sensor sampling rate in Hz")
    units: str = Field(description="Physical measurement units")
    voltage_supply_v: Optional[float] = Field(default=None, description="Supply voltage")
    temperature_c: Optional[float] = Field(default=None, description="Ambient temperature")
    humidity_pct: Optional[float] = Field(default=None, description="Ambient humidity percentage")
    health_label: HealthLabel = Field(
        default=HealthLabel.UNKNOWN,
        description="Health status label for ML training"
    )

    def to_csv_row(self) -> Dict:
        """Convert to dictionary for CSV export."""
        return {
            "session_id": self.session_id,
            "unit_id": self.unit_id,
            "sensor_name": self.sensor_name,
            "file_name": self.file_name,
            "file_format": self.file_format,
            "start_time_utc": self.start_time_utc.isoformat() + "Z",
            "duration_s": self.duration_s,
            "sampling_rate_hz": self.sampling_rate_hz,
            "units": self.units,
            "voltage_supply_v": self.voltage_supply_v,
            "temperature_c": self.temperature_c,
            "humidity_pct": self.humidity_pct,
            "health_label": self.health_label.value,
        }
