import math
from alyxspu import AlyxSPU, AlyxError, parse_samples


class FakeSerial:
    """Mimics the firmware's replies."""
    is_open = True

    def __init__(self):
        self.out = bytearray()

    @property
    def in_waiting(self):
        return len(self.out)

    def reset_input_buffer(self):
        self.out.clear()

    def close(self):
        self.is_open = False

    def read(self, n):
        d, self.out = bytes(self.out[:n]), self.out[n:]
        return d

    def write(self, data):
        cmd = data.decode().strip()
        head = cmd.split()[0]
        reply = {
            "PING": "OK ALYX-I100 v1\n",
            "CALC": "RES 4.24264\n",
            "NET": "OK net 2-4-1\n",
            "TRAIN": "LOSS 100 0.250000\nLOSS 200 0.100000\nOK trained loss=0.100000\n",
            "RUN": "OUT 0.9712 0.0100\n",
            "INFO": "OK net=2-4-1 samples=4 heap=1000\n",
            "BAD": "ERR unknown command\n",
        }[head]
        self.out += reply.encode()


def make():
    spu = AlyxSPU(auto_open=False)
    spu._ser = FakeSerial()
    return spu


def test_calc_and_run():
    spu = make()
    assert math.isclose(spu.calc("3*sqrt(2)"), 4.24264)
    assert spu.run([1, 0]) == [0.9712, 0.01]


def test_train_streams_losses():
    spu, seen = make(), []
    hist = spu.train(200, 0.5, on_loss=lambda e, v: seen.append(e))
    assert hist == [(100, 0.25), (200, 0.1)] and seen == [100, 200]


def test_info_and_errors():
    spu = make()
    assert spu.info()["net"] == "2-4-1"
    try:
        spu.command("BAD")
    except AlyxError as e:
        assert "unknown" in str(e)
    else:
        raise AssertionError


def test_parse_samples():
    assert parse_samples("0 1 | 1\n\n1 1 | 0") == [([0, 1], [1]), ([1, 1], [0])]
