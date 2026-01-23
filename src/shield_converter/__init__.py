"""
Shield Converter - ESP32 Binary Data to CSV Converter for Project SHIELD

Converts ESP32 DAQ binary data files from SD card into per-sensor CSV files
with Pydantic-based metadata models for ML training.
"""

from .models import (
    SensorType,
    HealthLabel,
    SensorInfo,
    DeviceInfo,
    DataFiles,
    Statistics,
    FirmwareMetadata,
    SessionRecord,
)
from .converter import (
    parse_fast_data,
    parse_medium_data,
    parse_slow_data,
    load_firmware_metadata,
    parse_folder_name,
    convert_run,
)

__version__ = "0.1.0"
__all__ = [
    "SensorType",
    "HealthLabel",
    "SensorInfo",
    "DeviceInfo",
    "DataFiles",
    "Statistics",
    "FirmwareMetadata",
    "SessionRecord",
    "parse_fast_data",
    "parse_medium_data",
    "parse_slow_data",
    "load_firmware_metadata",
    "parse_folder_name",
    "convert_run",
]
