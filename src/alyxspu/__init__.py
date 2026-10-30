"""alyxspu: control an Alyx I100 SPU (ESP32) over USB serial."""
from pathlib import Path

from .device import (
    MAX_HIDDEN, MAX_INPUTS, MAX_OUTPUTS, MAX_SAMPLES,
    AlyxError, AlyxSPU, parse_samples,
)

__version__ = "0.1.0"
__all__ = ["AlyxSPU", "AlyxError", "parse_samples", "firmware_path", "__version__",
           "MAX_INPUTS", "MAX_HIDDEN", "MAX_OUTPUTS", "MAX_SAMPLES"]


def firmware_path() -> Path:
    """Path to the bundled Arduino sketch to flash onto the ESP32."""
    return Path(__file__).parent / "firmware" / "alyx_i100" / "alyx_i100.ino"


def _print_firmware_path() -> None:
    print(firmware_path())
