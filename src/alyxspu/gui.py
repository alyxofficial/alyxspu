"""Desktop studio for the Alyx I100. Run with `alyxspu-studio` or `python -m alyxspu`."""
import queue
import threading
import tkinter as tk
from tkinter import ttk

from .device import AlyxSPU, parse_samples

XOR = "0 0 | 0\n0 1 | 1\n1 0 | 1\n1 1 | 0"


class Studio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Alyx Studio - I100 SPU")
        self.geometry("900x640")
        self.spu = None
        self.q = queue.Queue()
        self.losses = []
        self._build_ui()
        self.refresh_ports()
        self.after(50, self._poll)

    # ----- UI -----
    def _build_ui(self):
        top = ttk.Frame(self, padding=6); top.pack(fill="x")
        self.port = ttk.Combobox(top, width=18, state="readonly"); self.port.pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_ports).pack(side="left", padx=4)
        self.btn = ttk.Button(top, text="Connect", command=self.toggle); self.btn.pack(side="left")
        self.status = ttk.Label(top, text="Disconnected"); self.status.pack(side="left", padx=10)
        ttk.Button(top, text="Device info", command=lambda: self.bg(lambda: self.spu.info(), self.log_obj)).pack(side="right")

        nb = ttk.Notebook(self); nb.pack(fill="both", expand=True, padx=6)
        calc = ttk.Frame(nb, padding=10); nb.add(calc, text="Calculator")
        self.expr = ttk.Entry(calc, font=("Consolas", 14)); self.expr.pack(fill="x")
        self.expr.bind("<Return>", lambda e: self.calc())
        ttk.Button(calc, text="Calculate on ESP32", command=self.calc).pack(anchor="w", pady=6)
        self.result = ttk.Label(calc, text="=", font=("Consolas", 22)); self.result.pack(anchor="w")
        ttk.Label(calc, text="Supports + - * / ^ ( ) pi sqrt sin cos tan exp ln abs").pack(anchor="w", pady=8)

        nn = ttk.Frame(nb, padding=10); nb.add(nn, text="Neural network")
        row = ttk.Frame(nn); row.pack(fill="x")
        self.dims = []
        for label, val in (("Inputs", 2), ("Hidden", 4), ("Outputs", 1)):
            ttk.Label(row, text=label).pack(side="left")
            s = ttk.Spinbox(row, from_=1, to=16, width=4); s.set(val); s.pack(side="left", padx=(2, 10))
            self.dims.append(s)
        ttk.Button(row, text="Build network", command=self.build).pack(side="left")
        ttk.Label(nn, text="Training data, one sample per line: inputs | targets (0 to 1)").pack(anchor="w", pady=(8, 0))
        self.data = tk.Text(nn, height=6, font=("Consolas", 11)); self.data.pack(fill="x")
        self.data.insert("1.0", XOR)
        row = ttk.Frame(nn); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Epochs").pack(side="left")
        self.epochs = ttk.Entry(row, width=7); self.epochs.insert(0, "3000"); self.epochs.pack(side="left", padx=(2, 10))
        ttk.Label(row, text="Learning rate").pack(side="left")
        self.lr = ttk.Entry(row, width=6); self.lr.insert(0, "0.5"); self.lr.pack(side="left", padx=(2, 10))
        ttk.Button(row, text="Send data", command=self.send_data).pack(side="left")
        ttk.Button(row, text="Train", command=self.train).pack(side="left", padx=6)
        self.plot = tk.Canvas(nn, height=140, bg="#101820", highlightthickness=0); self.plot.pack(fill="x")
        row = ttk.Frame(nn); row.pack(fill="x", pady=6)
        ttk.Label(row, text="Run inputs").pack(side="left")
        self.run_in = ttk.Entry(row, width=20); self.run_in.insert(0, "1 0"); self.run_in.pack(side="left", padx=6)
        ttk.Button(row, text="Run", command=self.run).pack(side="left")
        self.out = ttk.Label(row, text="", font=("Consolas", 13)); self.out.pack(side="left", padx=10)

        self.logbox = tk.Text(self, height=9, state="disabled", font=("Consolas", 9))
        self.logbox.pack(fill="x", padx=6, pady=6)

    # ----- helpers -----
    def bg(self, fn, done=None):
        """Run a device call off the UI thread; report errors in the log."""
        def work():
            try:
                res = fn()
            except Exception as e:
                self.q.put(("log", f"! {e}")); return
            if done:
                self.q.put(("call", lambda: done(res)))
        threading.Thread(target=work, daemon=True).start()

    def log(self, text):
        self.logbox.config(state="normal"); self.logbox.insert("end", text + "\n")
        self.logbox.see("end"); self.logbox.config(state="disabled")

    def log_obj(self, obj):
        self.log(str(obj))

    def _poll(self):
        while not self.q.empty():
            item = self.q.get()
            if item[0] == "log":
                self.log(item[1])
            elif item[0] == "call":
                item[1]()
            elif item[0] == "loss":
                self.losses.append(item[1]); self.draw()
        self.after(50, self._poll)

    def need(self):
        if not (self.spu and self.spu.is_open):
            self.log("! Not connected"); return False
        return True

    # ----- actions -----
    def refresh_ports(self):
        ports = AlyxSPU.list_ports()
        self.port["values"] = ports
        if ports and not self.port.get():
            self.port.set(ports[0])

    def toggle(self):
        if self.spu and self.spu.is_open:
            self.spu.close(); self.btn.config(text="Connect"); self.status.config(text="Disconnected"); return
        spu = AlyxSPU(self.port.get(), auto_open=False)
        self.status.config(text="Connecting...")

        def connected(_):
            self.spu = spu; self.btn.config(text="Disconnect"); self.status.config(text="Connected")

        def failed_reset():
            if not (self.spu and self.spu.is_open):
                self.status.config(text="Disconnected")
        self.bg(spu.open, connected)
        self.after(4000, failed_reset)

    def calc(self):
        if self.need():
            self.bg(lambda: self.spu.calc(self.expr.get()), lambda r: self.result.config(text=f"= {r:g}"))

    def build(self):
        if self.need():
            n = [int(s.get()) for s in self.dims]
            self.bg(lambda: self.spu.build(*n), lambda _: self.log("network built"))

    def send_data(self):
        if self.need():
            try:
                samples = parse_samples(self.data.get("1.0", "end"))
            except ValueError as e:
                self.log(f"! {e}"); return
            self.bg(lambda: self.spu.set_data(samples), lambda _: self.log(f"{len(samples)} samples sent"))

    def train(self):
        if self.need():
            self.losses = []
            ep, lr = int(self.epochs.get()), float(self.lr.get())
            self.bg(lambda: self.spu.train(ep, lr, on_loss=lambda e, v: self.q.put(("loss", v))),
                    lambda h: self.log(f"trained, final loss {h[-1][1]:.6f}" if h else "trained"))

    def run(self):
        if self.need():
            x = [float(v) for v in self.run_in.get().split()]
            self.bg(lambda: self.spu.run(x), lambda r: self.out.config(text="  ".join(f"{v:.4f}" for v in r)))

    def draw(self):
        c = self.plot; c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if len(self.losses) < 2:
            return
        m = max(self.losses) or 1
        pts = []
        for i, v in enumerate(self.losses):
            pts += [i / (len(self.losses) - 1) * (w - 10) + 5, h - 8 - v / m * (h - 16)]
        c.create_line(*pts, fill="#3ddc97", width=2)
        c.create_text(8, 8, anchor="nw", fill="#9fb3c8", text=f"loss {self.losses[-1]:.5f}")


def main():
    Studio().mainloop()


if __name__ == "__main__":
    main()
