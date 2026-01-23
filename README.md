# Shield Converter

ESP32 Binary Data to CSV Converter for Project SHIELD.

Converts binary data files from the ESP32 DAQ system into per-sensor CSV files with metadata for ML training pipelines.

## Installation

### Option 1: Install from source (recommended for development)

```bash
cd shield_converter
pip install -e .
```

### Option 2: Install dependencies only

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Convert a single run directory
shield-converter convert /path/to/UNIT_001_RUN_001 --output ./output

# Convert all runs from an SD card
shield-converter convert-all /Volumes/SDCARD --output ./output

# Validate binary files before conversion
shield-converter validate /path/to/UNIT_001_RUN_001

# Display binary format information
shield-converter info
```

## Input Format

The converter expects run directories containing binary files from the ESP32 DAQ system:

```
UNIT_001_RUN_001/
├── fast_data.bin      # 1kHz: IMU + Vibration
├── medium_data.bin    # 200Hz: Current
├── slow_data.bin      # 50Hz: Pressure + Temperature
└── meta.json          # Optional: firmware metadata
```

### Binary Record Formats

| File | Sample Rate | Record Size | Sensors |
|------|-------------|-------------|---------|
| `fast_data.bin` | 1kHz | 12 bytes | IMU (m/s²), Vibration (binary) |
| `medium_data.bin` | 200Hz | 8 bytes | Current (A) |
| `slow_data.bin` | 50Hz | 12 bytes | Pressure (kPa), Temperature (°C) |

### Folder Naming Convention

Unit tracking uses folder names:
- `UNIT_001_RUN_001` → unit_id: `unit_0001`, session_id: `RUN_001`
- `RUN_005` → unit_id: `unit_0001` (default), session_id: `RUN_005`

## Output Format

```
output/
├── data/
│   └── UNIT_0001_RUN_001/
│       ├── imu.csv
│       ├── vibration.csv
│       ├── current.csv
│       ├── pressure.csv
│       └── temperature.csv
└── metadata/
    └── sessions/
        └── sessions.csv
```

### Per-Sensor CSV Format

```csv
timestamp_ms,value
0,0.523
1,0.518
2,0.531
```

### Session Metadata CSV

```csv
session_id,unit_id,sensor_name,file_name,file_format,start_time_utc,duration_s,sampling_rate_hz,units,health_label
RUN_001,unit_0001,imu,imu.csv,csv,2026-01-22T10:30:00Z,600,1000,m/s^2,unknown
RUN_001,unit_0001,current,current.csv,csv,2026-01-22T10:30:00Z,600,200,A,unknown
```

## CLI Reference

### `convert`

Convert a single run directory.

```bash
shield-converter convert <RUN_DIR> [OPTIONS]

Options:
  -o, --output PATH        Output directory [default: ./output]
  -u, --unit-id TEXT       Override unit ID
  -l, --health-label TEXT  Health label: unknown|healthy|degraded|faulty [default: unknown]
  -v, --verbose            Print verbose output
```

### `convert-all`

Convert all run directories found in a base directory.

```bash
shield-converter convert-all <INPUT_DIR> [OPTIONS]

Options:
  -o, --output PATH        Output directory [default: ./output]
  -l, --health-label TEXT  Health label for all sensors [default: unknown]
  -v, --verbose            Print verbose output
```

### `validate`

Validate binary files without converting.

```bash
shield-converter validate <RUN_DIR> [OPTIONS]

Options:
  -v, --verbose    Print detailed information
```

### `info`

Display binary format specifications.

```bash
shield-converter info
```

## Python API

```python
from shield_converter import convert_run, HealthLabel
from pathlib import Path

# Convert a single run
output_files, sessions = convert_run(
    run_dir=Path("/path/to/UNIT_001_RUN_001"),
    output_dir=Path("./output"),
    health_label=HealthLabel.HEALTHY,
    verbose=True,
)

# Access results
for sensor_name, csv_path in output_files.items():
    print(f"{sensor_name}: {csv_path}")

for session in sessions:
    print(f"{session.sensor_name} @ {session.sampling_rate_hz}Hz")
```

### Parsing Binary Files Directly

```python
from shield_converter import parse_fast_data, parse_medium_data, parse_slow_data
from pathlib import Path

# Parse fast data (returns list of (timestamp_ms, sensor_id, value) tuples)
records = parse_fast_data(Path("fast_data.bin"))

# Parse medium data (returns list of (timestamp_ms, current) tuples)
records = parse_medium_data(Path("medium_data.bin"))

# Parse slow data (returns list of (timestamp_ms, sensor_id, value) tuples)
records = parse_slow_data(Path("slow_data.bin"))
```

## Health Labels

Health labels are used for ML training to indicate the condition of the monitored equipment:

| Label | Description |
|-------|-------------|
| `unknown` | Default for unlabeled data |
| `healthy` | Normal operating condition |
| `degraded` | Early signs of wear or degradation |
| `faulty` | Equipment failure or fault condition |

Set labels during conversion:

```bash
shield-converter convert /path/to/run --health-label healthy
```

## Testing

The package includes a comprehensive test suite using pytest with synthetic binary data.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=shield_converter

# Run specific test file
pytest tst/test_converter.py

# Run specific test class
pytest tst/test_converter.py::TestParseFastData
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `models.py` | 12 tests | Pydantic models, enums, serialization |
| `converter.py` | 25 tests | Binary parsing, conversion, validation |

Tests use synthetic binary data generated in `conftest.py` fixtures, allowing verification without real hardware data.

### Test Categories

- **Record sizes**: Verify struct sizes match firmware definitions
- **Binary parsing**: Parse all three data file formats
- **Sensor splitting**: Correctly separate multi-sensor files
- **Folder naming**: Parse all naming conventions
- **End-to-end conversion**: Full pipeline with metadata generation
- **Validation**: File integrity and metadata checks

## Project Structure

```
shield_converter/
├── pyproject.toml          # Project configuration
├── requirements.txt        # Dependencies
├── README.md
├── src/
│   └── shield_converter/
│       ├── __init__.py     # Public API exports
│       ├── __main__.py     # Module entry point
│       ├── cli.py          # Typer CLI
│       ├── converter.py    # Binary parsing & conversion
│       └── models.py       # Pydantic data models
└── tst/
    ├── conftest.py         # Pytest fixtures (synthetic data)
    ├── test_models.py      # Pydantic model tests
    └── test_converter.py   # Binary parsing & conversion tests
```

## License

MIT
