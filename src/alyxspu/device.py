"""Driver for the Alyx I100 SPU (ESP32 running alyx_i100.ino) over USB serial."""
from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import serial
import serial.tools.list_ports

MAX_INPUTS, MAX_HIDDEN, MAX_OUTPUTS, MAX_SAMPLES = 8, 16, 4, 32
_REPLY_HEADS = ("OK", "ERR", "RES", "OUT")

Sample = Tuple[Sequence[float], Sequence[float]]


class AlyxError(Exception):
    """Raised for connection problems, timeouts and errors reported by the device."""


def _fmt(values: Sequence[float]) -> str:
    return " ".join(format(float(v), ".9g") for v in values)


def parse_samples(text: str) -> List[Tuple[List[float], List[float]]]:
    """Parse lines like ``0 1 | 1`` into ``([0.0, 1.0], [1.0])`` pairs."""
    samples = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            raise ValueError(f"line {n}: expected 'inputs | targets'")
        left, right = line.split("|", 1)
        samples.append(([float(v) for v in left.split()], [float(v) for v in right.split()]))
    return samples


class AlyxSPU:
    """Connection to one Alyx I100.

    >>> with AlyxSPU("COM5") as spu:
    ...     spu.calc("sqrt(2)*3")
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200,
                 timeout: float = 3.0, auto_open: bool = True):
        self.port, self.baudrate, self.timeout = port, baudrate, timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()
        if auto_open:
            self.open()

    # ----- connection -----
    @staticmethod
    def list_ports() -> List[str]:
        return [p.device for p in serial.tools.list_ports.comports()]

    def _autodetect(self) -> str:
        ports = self.list_ports()
        if len(ports) == 1:
            return ports[0]
        raise AlyxError("No serial port given and " + (
            "none found." if not ports else f"several found ({', '.join(ports)}); pass port=..."))

    def open(self) -> None:
        with self._lock:
            if self._ser and self._ser.is_open:
                return
            port = self.port or self._autodetect()
            try:
                self._ser = serial.Serial(port, self.baudrate, timeout=0.05, dtr=False, rts=False)
            except serial.SerialException as e:
                raise AlyxError(f"Cannot open {port}: {e}") from e
            self.port = port
            time.sleep(0.3)
            last = None
            for _ in range(3):
                try:
                    self.ping()
                    return
                except AlyxError as e:
                    last = e
            self.close()
            raise AlyxError(f"No reply from device on {port} (is alyx_i100.ino flashed?): {last}")

    def close(self) -> None:
        with self._lock:
            if self._ser:
                self._ser.close()
            self._ser = None

    @property
    def is_open(self) -> bool:
        return bool(self._ser and self._ser.is_open)

    def __enter__(self) -> "AlyxSPU":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ----- protocol -----
    def command(self, line: str, timeout: Optional[float] = None,
                on_line: Optional[Callable[[str], None]] = None) -> List[str]:
        """Send one raw command and return every line up to the final reply.

        The timeout restarts whenever data arrives, so long jobs that stream
        progress (like training) don't time out.
        """
        wait = timeout or self.timeout
        with self._lock:
            if not self.is_open:
                raise AlyxError("Not connected")
            ser = self._ser
            ser.reset_input_buffer()
            ser.write((line + "\n").encode("ascii"))
            buf, lines = bytearray(), []
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                chunk = ser.read(max(1, ser.in_waiting))
                if not chunk:
                    continue
                buf += chunk
                deadline = time.monotonic() + wait
                while b"\n" in buf:
                    raw, _, rest = bytes(buf).partition(b"\n")
                    buf = bytearray(rest)
                    text = raw.decode("ascii", "ignore").strip()
                    if not text:
                        continue
                    lines.append(text)
                    if on_line:
                        on_line(text)
                    head = text.split(" ", 1)[0]
                    if head == "ERR":
                        raise AlyxError(text[4:] or "device error")
                    if head in _REPLY_HEADS:
                        return lines
            raise AlyxError(f"Timed out waiting for reply to {line!r}")

    # ----- high level API -----
    def ping(self) -> str:
        return self.command("PING")[-1]

    def info(self) -> Dict[str, str]:
        """Return e.g. {'net': '2-4-1', 'samples': '4', 'heap': '212000'}."""
        reply = self.command("INFO")[-1]
        return dict(kv.split("=", 1) for kv in reply.split()[1:] if "=" in kv)

    def calc(self, expression: str) -> float:
        """Evaluate an expression on the ESP32 (+ - * / ^ ( ) pi sqrt sin cos tan exp ln abs)."""
        reply = self.command("CALC " + expression)[-1]
        return float(reply.split()[1])

    def build(self, inputs: int, hidden: int, outputs: int) -> None:
        """Create a fresh 1-hidden-layer sigmoid network with random weights."""
        if not (1 <= inputs <= MAX_INPUTS and 1 <= hidden <= MAX_HIDDEN and 1 <= outputs <= MAX_OUTPUTS):
            raise ValueError(f"limits: inputs 1-{MAX_INPUTS}, hidden 1-{MAX_HIDDEN}, outputs 1-{MAX_OUTPUTS}")
        self.command(f"NET {inputs} {hidden} {outputs}")

    def clear_data(self) -> None:
        self.command("CLEAR")

    def add_sample(self, inputs: Sequence[float], targets: Sequence[float]) -> None:
        self.command(f"DATA {_fmt(inputs)} | {_fmt(targets)}")

    def set_data(self, samples: Sequence[Sample]) -> None:
        """Replace the training set. Targets must be between 0 and 1."""
        if len(samples) > MAX_SAMPLES:
            raise ValueError(f"at most {MAX_SAMPLES} samples")
        self.clear_data()
        for x, y in samples:
            self.add_sample(x, y)

    def train(self, epochs: int = 3000, lr: float = 0.5,
              on_loss: Optional[Callable[[int, float], None]] = None) -> List[Tuple[int, float]]:
        """Train on the device. Returns [(epoch, loss), ...]; on_loss fires live."""
        history: List[Tuple[int, float]] = []

        def handle(text: str) -> None:
            if text.startswith("LOSS "):
                _, e, v = text.split()
                history.append((int(e), float(v)))
                if on_loss:
                    on_loss(int(e), float(v))

        self.command(f"TRAIN {int(epochs)} {format(float(lr), '.9g')}",
                     timeout=max(self.timeout, 10.0), on_line=handle)
        return history

    def run(self, inputs: Sequence[float]) -> List[float]:
        """Run a forward pass and return the output activations."""
        reply = self.command("RUN " + _fmt(inputs))[-1]
        return [float(v) for v in reply.split()[1:]]
