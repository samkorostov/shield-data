"""
Entry point for running shield_converter as a module.

Usage:
    python -m shield_converter convert /path/to/run --output ./output
    python -m shield_converter convert-all /path/to/sdcard --output ./output
    python -m shield_converter validate /path/to/run
    python -m shield_converter info
"""

from .cli import main

if __name__ == "__main__":
    main()
