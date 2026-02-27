import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# [TODO]: Confirm all units
SENSOR_UNITS = {
    "Accelerometer": "m/s²",
    "Gyroscope": "rad/s",
    "Magnetometer": "µT",
    "Pressure": "kPa",
    "Temperature": "°C",
    "Current": "A",
    "Microphone": "dBFS",
    "Photodiode": "V",
    "Vibration": "Binary",
}
    
def plot_sensor_data(*sensor_csvs: str, exclude_startup_noise: bool = False) -> None:
    sensor_csvs = [Path(csv) for csv in sensor_csvs]
    for csv in sensor_csvs:
        if not csv.exists():
            raise FileNotFoundError(f"CSV file {csv} does not exist.")
    
    n = len(sensor_csvs)

    fig, axes = plt.subplots(n, 1, figsize=(30, 48), sharex=True)
    if n == 1:
        axes = [axes]
    
    for ax, sensor_csv in zip(axes, sensor_csvs):
        df = pd.read_csv(sensor_csv)
        if 'timestamp_ms' not in df.columns or 'value' not in df.columns:
            raise ValueError(f"CSV file {sensor_csv} must contain 'timestamp_ms' and 'value' columns.")
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
        sensor_name = sensor_csv.stem.capitalize()
        if exclude_startup_noise:
            df = df.iloc[10000:]  # Exclude first 10,000 rows (startup noise)
        ax.plot(df['timestamp'], df['value'])
        ax.set_title(sensor_name)
        ax.set_ylabel(f"Sensor Reading ({SENSOR_UNITS.get(sensor_name)})")
        ax.grid()
    
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    
def plot_run_data(run_dir: str, exclude_startup_noise: bool = False) -> None:
    """
    Plot time-series data for all sensors in a run directory.

    Args:
        run_dir (Path): Path to the run directory containing sensor CSV files.
    """
    
    run_dir = Path(run_dir)
    
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist or is not a directory.")
    
    sensor_csvs = list(run_dir.glob("*.csv"))
    if not sensor_csvs:
        raise FileNotFoundError(f"No CSV files found in {run_dir}.")
    
    plot_sensor_data(*sensor_csvs, exclude_startup_noise=exclude_startup_noise)
    
def bin_data(sensor_csv: str, bin_size_s: str = '10s', exclude_startup_noise: bool = False) -> pd.DataFrame:
    """
    Bin time-series data from multiple sensor CSV files into fixed time intervals.
    Utilizes rolling mean to smooth data within each bin.

    Args:
        sensor_csvs (List[str]): List of paths to sensor CSV files.
        bin_size_s (str): Size of each time bin in seconds (e.g., '10s', '30s').

    Returns:
        pd.DataFrame: Binned data with mean values for each sensor in each time bin.
    """
    
    sensor_csv = Path(sensor_csv)
    if not sensor_csv.exists():
        raise FileNotFoundError(f"CSV file {sensor_csv} does not exist.")
    
    
    df = pd.read_csv(sensor_csv)
    if 'timestamp_ms' not in df.columns or 'value' not in df.columns:
        raise ValueError(f"CSV file {sensor_csv} must contain 'timestamp_ms' and 'value' columns.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    if exclude_startup_noise:
        df = df.iloc[10000:]  # Exclude first 10,000 rows (startup noise)
    
    binned_df = df.resample(f'{bin_size_s}').mean().reset_index()
    binned_df['sensor'] = sensor_csv.stem.capitalize()
    
    return binned_df

def plot_sensor_data_binned(*sensor_csvs: str, bin_size_s: str = '10s', exclude_startup_noise: bool = False) -> None:
    """
    Plot binned time-series data for multiple sensors in order to improve interpretability.
    Utilizes rolling mean

    Args:
        sensor_csvs (List[str]): List of paths to sensor CSV files.
        bin_size_s (str): Size of each time bin in seconds (e.g., '10s', '30s').
    """
    
    sensor_csvs = [Path(csv) for csv in sensor_csvs]
    for csv in sensor_csvs:
        if not csv.exists():
            raise FileNotFoundError(f"CSV file {csv} does not exist.")

    fig, axes = plt.subplots(len(sensor_csvs), 1, figsize=(30, 48), sharex=True)
    if len(sensor_csvs) == 1:
        axes = [axes]
        
    for ax, sensor_csv in zip(axes, sensor_csvs):
        binned_df = bin_data(sensor_csv, bin_size_s, exclude_startup_noise=exclude_startup_noise)
        sensor_name = sensor_csv.stem.capitalize()
        ax.plot(binned_df['timestamp'], binned_df['value'], label='Binned Sensor Value')
        ax.set_title(f"Binned Time-Series Data for {sensor_name} (Bin Size: {bin_size_s})")
        ax.set_ylabel(f"Mean Sensor Reading ({SENSOR_UNITS.get(sensor_name)})")
        ax.legend()
        ax.grid()
    
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()
    
def plot_run_data_binned(run_dir: str, bin_size_s: str = '10s', exclude_startup_noise: bool = False) -> None:
    """
    Plot binned time-series data for all sensors in a run directory.

    Args:
        run_dir (Path): Path to the run directory containing sensor CSV files.
        bin_size_s (str): Size of each time bin in seconds (e.g., '10s', '30s').
    """
    
    run_dir = Path(run_dir)
    
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist or is not a directory.")
    
    sensor_csvs = list(run_dir.glob("*.csv"))
    if not sensor_csvs:
        raise FileNotFoundError(f"No CSV files found in {run_dir}.")
    
    plot_sensor_data_binned(*sensor_csvs, bin_size_s=bin_size_s, exclude_startup_noise=exclude_startup_noise)
    
    
    