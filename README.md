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
shield-converter convert /Users/a11/Documents/UW/598/shield-data/data/5_5/RUN_249 --output ./output
shield-converter convert /path/to/RUN_067 --output ./output --start-hour 0 --end-hour 12

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
├── fast_data.bin      # Vibration + Microphone + Magnetometer + Gyroscope + Accelerometer
├── medium_data.bin    # Current + Photodiode
├── slow_data.bin      # Pressure + Temperature
├── meta.json          # Optional: firmware metadata
└── events.log         # Optional: firmware event log
```

### Binary Record Formats

| File | Sample Rate | Record Size | Sensors |
|------|-------------|-------------|---------|
| `fast_data.bin` | Fast tier | 20 bytes | Vibration, Microphone, Magnetometer, Gyroscope, Accelerometer |
| `medium_data.bin` | Medium tier | 20 bytes | Current, Photodiode |
| `slow_data.bin` | Slow tier | 20 bytes | Pressure, Temperature |

All three files use the same little-endian packed v2 record:

```c
typedef struct __attribute__((packed)) {
    uint32_t timestamp_ms;
    uint8_t  sensor_id;
    uint8_t  kind;        // 0=raw, 1=processed
    uint8_t  axis_count;  // scalar=1, vector=3
    uint8_t  flags;
    float    data[3];
} sensor_data_record_v2_t;
```

Sensor IDs:

| ID | Sensor | CSV columns |
|----|--------|-------------|
| 1 | vibration | `timestamp_ms,value` |
| 2 | current | `timestamp_ms,value` |
| 3 | pressure | `timestamp_ms,value` |
| 4 | temperature | `timestamp_ms,value` |
| 5 | microphone | `timestamp_ms,value` |
| 6 | photodiode | `timestamp_ms,value` |
| 7 | magnetometer | `timestamp_ms,x,y,z` |
| 8 | gyroscope | `timestamp_ms,x,y,z` |
| 9 | accelerometer | `timestamp_ms,x,y,z` |

### Folder Naming Convention

Unit tracking uses folder names:
- `UNIT_001_RUN_001` → unit_id: `unit_0001`, session_id: `RUN_001`
- `RUN_005` → unit_id: `unit_0001` (default), session_id: `RUN_005`

## Output Format

```
output/
├── data/
│   └── UNIT_0001_RUN_001/
│       ├── vibration.csv
│       ├── proc_vibration.csv
│       ├── current.csv
│       ├── proc_current.csv
│       ├── pressure.csv
│       ├── proc_pressure.csv
│       ├── temperature.csv
│       ├── proc_temperature.csv
│       └── ...
└── metadata/
    └── sessions/
        └── sessions.csv
```

### Per-Sensor CSV Format

Raw records use `<sensor>.csv`; processed records use `proc_<sensor>.csv`.
The column format is unchanged from earlier converter output so downstream code
can keep reading the files the same way.

```csv
timestamp_ms,value
0,0.523
1,0.518
2,0.531
```

Vector sensors use:

```csv
timestamp_ms,x,y,z
0,0.01,0.02,9.81
1,0.01,0.02,9.80
```

### Session Metadata CSV

```csv
session_id,unit_id,sensor_name,file_name,file_format,start_time_utc,duration_s,sampling_rate_hz,units,health_label
RUN_001,unit_0001,vibration,vibration.csv,csv,2026-01-22T10:30:00Z,600,1000,binary,unknown
RUN_001,unit_0001,proc_vibration,proc_vibration.csv,csv,2026-01-22T10:30:00Z,600,1000,binary,unknown
RUN_001,unit_0001,current,current.csv,csv,2026-01-22T10:30:00Z,600,200,A,unknown
```

`duration_s` is calculated from each output CSV's actual timestamp span. If a
fast data file ends earlier than medium or slow data, the fast sensors will show
the shorter duration in metadata.

## CLI Reference

### `convert`

Convert a single run directory.

```bash
shield-converter convert <RUN_DIR> [OPTIONS]

Options:
  -o, --output PATH        Output directory [default: ./output]
  -u, --unit-id TEXT       Override unit ID
  -l, --health-label TEXT  Health label: unknown|healthy|degraded|faulty [default: unknown]
  --start-hour FLOAT       Start of export window relative to run start
  --end-hour FLOAT         End of export window relative to run start
  -v, --verbose            Print verbose output
```

If neither `--start-hour` nor `--end-hour` is provided, conversion exports all
available data. If only `--end-hour 12` is provided, conversion exports `0h-12h`.
If only `--start-hour 12` is provided, conversion exports from 12h to the end of
available data. Window exports overwrite the run's existing CSV files in the
output directory.

### `convert-all`

Convert all run directories found in a base directory.

```bash
shield-converter convert-all <INPUT_DIR> [OPTIONS]

Options:
  -o, --output PATH        Output directory [default: ./output]
  -l, --health-label TEXT  Health label for all sensors [default: unknown]
  --start-hour FLOAT       Start of export window relative to each run start
  --end-hour FLOAT         End of export window relative to each run start
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
    start_hour=0,
    end_hour=12,
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

# Parse fast data (returns a NumPy structured array of v2 records)
records = parse_fast_data(Path("fast_data.bin"))

# Parse medium data
records = parse_medium_data(Path("medium_data.bin"))

# Parse slow data
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
