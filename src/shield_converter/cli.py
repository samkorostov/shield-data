"""
Command-line interface for Shield Converter.

Usage:
    python -m shield_converter convert /path/to/RUN_001 --output ./output
    python -m shield_converter convert-all /Volumes/SDCARD --output ./output
    python -m shield_converter validate /path/to/RUN_001
"""

from pathlib import Path
from typing import Optional

import typer

from .converter import (
    convert_all_runs,
    convert_run,
    validate_run,
    FAST_RECORD_SIZE,
    MEDIUM_RECORD_SIZE,
    SLOW_RECORD_SIZE,
)
from .models import HealthLabel

app = typer.Typer(
    name="shield-converter",
    help="ESP32 Binary Data to CSV Converter for Project SHIELD",
    add_completion=False,
)


@app.command()
def convert(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to run directory containing binary data files",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory for CSV files and metadata",
    ),
    unit_id: Optional[str] = typer.Option(
        None,
        "--unit-id", "-u",
        help="Override unit ID (default: extracted from folder name)",
    ),
    health_label: HealthLabel = typer.Option(
        HealthLabel.UNKNOWN,
        "--health-label", "-l",
        help="Health label for all sensors",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print verbose output",
    ),
):
    """
    Convert a single run directory from binary to CSV format.

    Parses fast_data.bin, medium_data.bin, and slow_data.bin files
    and outputs per-sensor CSV files with metadata.
    """
    output_files, sessions = convert_run(
        run_dir=run_dir,
        output_dir=output,
        unit_id=unit_id,
        health_label=health_label,
        verbose=verbose,
    )

    # Write session metadata for single run
    if sessions:
        import pandas as pd
        metadata_dir = output / "metadata" / "sessions"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sessions_csv = metadata_dir / "sessions.csv"

        # Append to existing or create new
        sessions_df = pd.DataFrame([s.to_csv_row() for s in sessions])
        if sessions_csv.exists():
            existing_df = pd.read_csv(sessions_csv)
            sessions_df = pd.concat([existing_df, sessions_df], ignore_index=True)
        sessions_df.to_csv(sessions_csv, index=False)

        if verbose:
            print(f"Updated sessions metadata: {sessions_csv}")

    typer.echo(f"Converted {len(output_files)} sensor files")
    for sensor_name, csv_path in output_files.items():
        typer.echo(f"  {sensor_name}: {csv_path}")


@app.command("convert-all")
def convert_all(
    input_dir: Path = typer.Argument(
        ...,
        help="Base directory containing run directories (e.g., SD card mount)",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    output: Path = typer.Option(
        Path("./output"),
        "--output", "-o",
        help="Output directory for CSV files and metadata",
    ),
    health_label: HealthLabel = typer.Option(
        HealthLabel.UNKNOWN,
        "--health-label", "-l",
        help="Health label for all sensors",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print verbose output",
    ),
):
    """
    Convert all run directories found in input directory.

    Searches for directories containing binary data files and converts each.
    Outputs combined session metadata CSV.
    """
    sessions = convert_all_runs(
        input_dir=input_dir,
        output_dir=output,
        health_label=health_label,
        verbose=verbose,
    )

    typer.echo(f"Converted {len(sessions)} sensor sessions")


@app.command()
def validate(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to run directory to validate",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Print verbose output",
    ),
):
    """
    Validate binary files in a run directory.

    Checks file existence, record integrity, and metadata consistency.
    """
    typer.echo(f"Validating: {run_dir}")
    typer.echo(f"  Expected record sizes: fast={FAST_RECORD_SIZE}, medium={MEDIUM_RECORD_SIZE}, slow={SLOW_RECORD_SIZE}")

    results = validate_run(run_dir, verbose=verbose)

    # Print file status
    for fname, info in results["files"].items():
        if info.get("exists"):
            status = typer.style("OK", fg=typer.colors.GREEN)
            typer.echo(f"  {fname}: {status} ({info['expected_records']} records)")
        else:
            status = typer.style("NOT FOUND", fg=typer.colors.YELLOW)
            typer.echo(f"  {fname}: {status}")

    # Print metadata status
    if results.get("metadata", {}).get("exists"):
        meta = results["metadata"]
        typer.echo(f"  meta.json: run_id={meta['run_id']}, duration={meta['duration_ms']}ms")
        typer.echo(f"    total_samples: {meta['total_samples']}")
    else:
        typer.echo(f"  meta.json: {typer.style('NOT FOUND', fg=typer.colors.YELLOW)}")

    # Print warnings
    for warning in results.get("warnings", []):
        typer.echo(typer.style(f"  WARNING: {warning}", fg=typer.colors.YELLOW))

    # Print errors
    for error in results.get("errors", []):
        typer.echo(typer.style(f"  ERROR: {error}", fg=typer.colors.RED))

    if results["valid"]:
        typer.echo(typer.style("Validation passed", fg=typer.colors.GREEN))
    else:
        typer.echo(typer.style("Validation failed", fg=typer.colors.RED))
        raise typer.Exit(1)


@app.command()
def info():
    """
    Display binary format information.
    """
    import struct

    typer.echo("Binary Record Formats:")
    typer.echo("")
    typer.echo("fast_data.bin (1kHz - IMU + Vibration):")
    typer.echo(f"  Format: <I B 3x f (little-endian)")
    typer.echo(f"  Size: {FAST_RECORD_SIZE} bytes/record")
    typer.echo("  Fields: timestamp_ms (uint32), sensor_id (uint8), reserved[3], data (float)")
    typer.echo("  Sensors: 0=IMU (m/s^2), 1=Vibration (binary)")
    typer.echo("")
    typer.echo("medium_data.bin (200Hz - Current):")
    typer.echo(f"  Format: <I f (little-endian)")
    typer.echo(f"  Size: {MEDIUM_RECORD_SIZE} bytes/record")
    typer.echo("  Fields: timestamp_ms (uint32), current (float)")
    typer.echo("  Units: Amperes (A)")
    typer.echo("")
    typer.echo("slow_data.bin (50Hz - Pressure + Temperature):")
    typer.echo(f"  Format: <I B 3x f (little-endian)")
    typer.echo(f"  Size: {SLOW_RECORD_SIZE} bytes/record")
    typer.echo("  Fields: timestamp_ms (uint32), sensor_id (uint8), reserved[3], data (float)")
    typer.echo("  Sensors: 3=Pressure (kPa), 4=Temperature (C)")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
