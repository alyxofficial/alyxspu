
"""Driver for the Alyx I100 SPU (ESP32 running alyx_i100.ino) over USB serial."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import serial
from serial.tools import list_ports as serial_list_ports


MAX_INPUTS = 8
MAX_HIDDEN = 16
MAX_OUTPUTS = 4
MAX_SAMPLES = 32

_REPLY_HEADS = ("OK", "ERR", "RES", "OUT")

Sample = Tuple[Sequence[float], Sequence[float]]


class AlyxError(Exception):
    """Raised for connection problems, timeouts and device errors."""


def _fmt(values: Sequence[float]) -> str:
    return " ".join(format(float(v), ".9g") for v in values)


def parse_samples(text: str) -> List[Tuple[List[float], List[float]]]:
    """Parse lines such as '0 1 | 1'."""
    samples = []

    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            raise ValueError(f"line {n}: expected 'inputs | targets'")

        left, right = line.split("|", 1)

        try:
            inputs = [float(v) for v in left.split()]
            targets = [float(v) for v in right.split()]
        except ValueError as e:
            raise ValueError(f"line {n}: invalid numeric value") from e

        samples.append((inputs, targets))

    return samples


class AlyxSPU:
    """Connection to one Alyx I100 SPU.

    Example:
        with AlyxSPU("COM5") as spu:
            print(spu.ping())
            print(spu.calc("sqrt(2)*3"))
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        timeout: float = 3.0,
        auto_open: bool = True,
    ):
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")

        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()

        if auto_open:
            self.open()

    # --------------------------------------------------
    # CONNECTION
    # --------------------------------------------------

    @staticmethod
    def list_ports() -> List[str]:
        """Return available serial port names."""
        return [p.device for p in serial_list_ports.comports()]

    @staticmethod
    def port_info() -> List[Dict[str, str]]:
        """Return detailed information about available serial ports."""
        return [
            {
                "device": p.device,
                "description": p.description or "",
                "hardware_id": p.hwid or "",
                "manufacturer": p.manufacturer or "",
            }
            for p in serial_list_ports.comports()
        ]

    def _autodetect(self) -> str:
        ports = self.list_ports()

        if len(ports) == 1:
            return ports[0]

        if not ports:
            raise AlyxError(
                "No serial ports found. Connect your ESP32 and try again."
            )

        raise AlyxError(
            f"Multiple serial ports found: {', '.join(ports)}. "
            "Specify the port manually, e.g. AlyxSPU('COM5')."
        )

    def open(self) -> None:
        """Open the serial connection and verify the device responds."""

        with self._lock:
            if self.is_open:
                return

            port = self.port or self._autodetect()

            ser = serial.Serial(
                port=None,
                baudrate=self.baudrate,
                timeout=0.05,
                write_timeout=2.0,
            )

            ser.port = port

            # Avoid intentionally asserting the ESP32 reset lines.
            ser.dtr = False
            ser.rts = False

            try:
                ser.open()

            except (serial.SerialException, OSError) as e:
                ser.close()

                if getattr(e, "winerror", None) == 5 or "PermissionError" in str(e):
                    raise AlyxError(
                        f"Access denied to {port}.\n"
                        "Another program may be using this port.\n"
                        "Close Arduino Serial Monitor, Thonny, PuTTY "
                        "or other serial applications and try again."
                    ) from e

                raise AlyxError(
                    f"Cannot open {port}: {e}"
                ) from e

            self._ser = ser
            self.port = port

            try:
                # Allow ESP32 USB serial startup messages to finish.
                time.sleep(0.8)

                # Discard boot messages before checking the protocol.
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                last_error = None

                for attempt in range(3):
                    try:
                        self.ping()
                        return

                    except AlyxError as e:
                        last_error = e

                        if attempt < 2:
                            time.sleep(0.25)

                raise AlyxError(
                    f"No valid reply from {port}. "
                    "Check that alyx_i100.ino is flashed, "
                    f"the baudrate is {self.baudrate}, "
                    f"and the device is connected. Last error: {last_error}"
                )

            except Exception:
                self.close()
                raise

    def close(self) -> None:
        """Safely close the serial connection."""

        with self._lock:
            ser = self._ser
            self._ser = None

            if ser is not None:
                try:
                    if ser.is_open:
                        ser.close()
                except (serial.SerialException, OSError):
                    pass

    @property
    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    def __enter__(self) -> "AlyxSPU":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------
    # LOW-LEVEL PROTOCOL
    # --------------------------------------------------

    def command(
        self,
        line: str,
        timeout: Optional[float] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Send a command and collect its response.

        Timeout is restarted whenever new serial data arrives.
        Progress messages can be received through on_line.
        """

        wait = self.timeout if timeout is None else float(timeout)

        if wait <= 0:
            raise ValueError("timeout must be positive")

        if not line or "\n" in line or "\r" in line:
            raise ValueError("command must be a non-empty single line")

        try:
            encoded = (line + "\n").encode("ascii")
        except UnicodeEncodeError as e:
            raise ValueError("commands must contain ASCII characters only") from e

        with self._lock:
            if not self.is_open:
                raise AlyxError("Not connected. Call open() first.")

            ser = self._ser

            if ser is None:
                raise AlyxError("Serial connection is unavailable")

            try:
                ser.reset_input_buffer()

                ser.write(encoded)
                ser.flush()

                buffer = bytearray()
                lines: List[str] = []

                deadline = time.monotonic() + wait

                while time.monotonic() < deadline:
                    chunk = ser.read(max(1, ser.in_waiting))

                    if not chunk:
                        continue

                    buffer.extend(chunk)
                    deadline = time.monotonic() + wait

                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)

                        text = raw.decode(
                            "ascii", errors="replace"
                        ).strip()

                        if not text:
                            continue

                        lines.append(text)

                        if on_line is not None:
                            on_line(text)

                        head = text.split(maxsplit=1)[0]

                        if head == "ERR":
                            message = text[3:].strip()
                            raise AlyxError(message or "Device reported an error")

                        if head in _REPLY_HEADS:
                            return lines

                raise AlyxError(
                    f"Timed out waiting for reply to {line!r} "
                    f"on {self.port}"
                )

            except (serial.SerialException, OSError) as e:
                self.close()
                raise AlyxError(
                    f"Serial communication failed on {self.port}: {e}"
                ) from e

    # --------------------------------------------------
    # DEVICE INFORMATION
    # --------------------------------------------------

    def ping(self) -> str:
        return self.command("PING")[-1]

    def info(self) -> Dict[str, str]:
        """Return device information."""
        reply = self.command("INFO")[-1]

        result = {}

        for item in reply.split()[1:]:
            if "=" in item:
                key, value = item.split("=", 1)
                result[key] = value

        return result

    # --------------------------------------------------
    # CALCULATOR
    # --------------------------------------------------

    def calc(self, expression: str) -> float:
        """Evaluate an expression on the ESP32."""

        if not expression.strip():
            raise ValueError("expression cannot be empty")

        if "\n" in expression or "\r" in expression:
            raise ValueError("expression must be a single line")

        reply = self.command("CALC " + expression)[-1]
        parts = reply.split(maxsplit=1)

        if len(parts) < 2:
            raise AlyxError(f"Invalid CALC response: {reply}")

        try:
            return float(parts[1])
        except ValueError as e:
            raise AlyxError(f"Invalid CALC result: {reply}") from e

    # --------------------------------------------------
    # NEURAL NETWORK
    # --------------------------------------------------

    def build(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
    ) -> None:
        """Create a one-hidden-layer sigmoid network."""

        if not 1 <= inputs <= MAX_INPUTS:
            raise ValueError(f"inputs must be between 1 and {MAX_INPUTS}")

        if not 1 <= hidden <= MAX_HIDDEN:
            raise ValueError(f"hidden must be between 1 and {MAX_HIDDEN}")

        if not 1 <= outputs <= MAX_OUTPUTS:
            raise ValueError(f"outputs must be between 1 and {MAX_OUTPUTS}")

        self.command(f"NET {inputs} {hidden} {outputs}")

    def clear_data(self) -> None:
        self.command("CLEAR")

    def add_sample(
        self,
        inputs: Sequence[float],
        targets: Sequence[float],
    ) -> None:
        if not inputs or not targets:
            raise ValueError("inputs and targets cannot be empty")

        if not all(0.0 <= float(v) <= 1.0 for v in targets):
            raise ValueError("targets must be between 0 and 1")

        self.command(f"DATA {_fmt(inputs)} | {_fmt(targets)}")

    def set_data(self, samples: Sequence[Sample]) -> None:
        """Replace the complete training dataset."""

        if len(samples) > MAX_SAMPLES:
            raise ValueError(f"At most {MAX_SAMPLES} samples are supported")

        self.clear_data()

        for inputs, targets in samples:
            self.add_sample(inputs, targets)

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

    def train(
        self,
        epochs: int = 3000,
        lr: float = 0.5,
        on_loss: Optional[Callable[[int, float], None]] = None,
    ) -> List[Tuple[int, float]]:
        """Train on the ESP32 and return the loss history."""

        if not 1 <= epochs:
            raise ValueError("epochs must be positive")

        if lr <= 0:
            raise ValueError("learning rate must be positive")

        history: List[Tuple[int, float]] = []

        def handle(text: str) -> None:
            if text.startswith("LOSS "):
                parts = text.split()

                if len(parts) != 3:
                    return

                try:
                    epoch = int(parts[1])
                    loss = float(parts[2])
                except ValueError:
                    return

                history.append((epoch, loss))

                if on_loss is not None:
                    on_loss(epoch, loss)

        self.command(
            f"TRAIN {int(epochs)} {format(float(lr), '.9g')}",
            timeout=max(self.timeout, 10.0),
            on_line=handle,
        )

        return history

    # --------------------------------------------------
    # INFERENCE
    # --------------------------------------------------

    def run(self, inputs: Sequence[float]) -> List[float]:
        """Run a forward pass and return output activations."""

        if not inputs:
            raise ValueError("inputs cannot be empty")

        reply = self.command("RUN " + _fmt(inputs))[-1]

        try:
            return [float(v) for v in reply.split()[1:]]
        except ValueError as e:
            raise AlyxError(f"Invalid RUN response: {reply}") from e
