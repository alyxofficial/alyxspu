# Alyxspu

Python driver and desktop studio for the **Alyx I100 SPU** (a simple processing unit built on an ESP32). Everything runs over a plain **USB serial** connection: no WiFi, no web.

The ESP32 does the work. It evaluates math expressions and trains and runs a small neural network on the chip. This library sends commands and reads the results.

## Install

```
pip install alyxspu
```

## 1. Flash the firmware

The Arduino sketch is [here](https://github.com/alyxofficial/Alyx-I100/blob/main/alyx-i100/alyx-i100.ino). Open that `alyx-i100.ino` in the Arduino IDE (ESP32 board package installed), select your board and port, and upload. Close the Serial Monitor afterwards, because it blocks the port.

## 2. Use it from Python

```python
from alyxspu import AlyxSPU

with AlyxSPU("COM5") as spu:              # "/dev/ttyUSB0" on Linux, omit if only one port exists
    print(spu.calc("sqrt(2)*3 + sin(pi/2)"))

    spu.build(inputs=2, hidden=4, outputs=1)
    spu.set_data([([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])])   # XOR
    history = spu.train(epochs=3000, lr=0.5, on_loss=lambda e, loss: print(e, loss))
    print(spu.run([1, 0]))                # close to [1.0]
```

| Method | What it does |
|---|---|
| `AlyxSPU.list_ports()` | List serial ports |
| `calc(expr)` | Evaluate `+ - * / ^ ( )`, `pi sqrt sin cos tan exp ln abs` on the ESP32 |
| `build(i, h, o)` | New 1-hidden-layer sigmoid network (max 8 / 16 / 4) |
| `set_data(samples)`, `add_sample(x, y)`, `clear_data()` | Training set (max 32 samples, targets 0 to 1) |
| `train(epochs, lr, on_loss=None)` | Train on-device, returns `[(epoch, loss), ...]` |
| `run(inputs)` | Forward pass, returns the outputs |
| `info()`, `ping()`, `command(line)` | Status and raw access |

Errors from the device or the connection raise `AlyxError`.

## 3. Desktop app

```
alyxspu-studio
```

(or `python -m alyxspu`). Pick the port, connect, and use the calculator and neural network tabs. It needs tkinter, which ships with most Python installs.

## Serial protocol

115200 baud, one text command per line: `PING`, `INFO`, `CALC <expr>`, `NET <i> <h> <o>`, `DATA x.. | y..`, `CLEAR`, `TRAIN <epochs> <lr>`, `RUN x..`. Replies start with `OK`, `ERR`, `RES`, `OUT` or `LOSS`.

## License

MIT
