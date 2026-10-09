#Copyright©2026. chutiphong bunloed
#All Rights Reserved.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Demo:
    def __init__(self, L=64):
        self.e = _H(L=L)
        self.x0 = np.exp(-self.e.idx ** 2 / 100.0).astype(complex)

    def fig_01_axis(self, path="fig_01_axis.png"):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.scatter(self.e.idx, np.zeros(self.e.N), c=np.where(self.e.idx % 2 == 0, "blue", "red"), s=10)
        ax.axvline(-64, color="black", linestyle="--", label="fold -64")
        ax.axvline(64, color="black", linestyle="--", label="fold +64")
        ax.axvline(0, color="green", linestyle=":", label="centre 0")
        ax.set_title("Honeycomb E8 Axis  A=blue  B=red")
        ax.set_yticks([])
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_02_fibonacci(self, path="fig_02_fibonacci.png"):
        seq = self.e.fib(15)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(len(seq)), seq, color="gold", edgecolor="black")
        ax.set_title("Fibonacci Sequence")
        ax.set_xlabel("n")
        ax.set_ylabel("F_n")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_03_pulses(self, path="fig_03_pulses.png"):
        t = np.linspace(0.0, 25.0, 2500)
        p = np.array([self.e.pulse(x) for x in t])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, p, color="cyan")
        ax.set_title("Fibonacci Pulse Train")
        ax.set_xlabel("t")
        ax.set_ylabel("envelope")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_04_golden(self, path="fig_04_golden.png"):
        t = np.linspace(0.0, 30.0, 3000)
        g = np.array([self.e.golden(x) for x in t])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(t, g, color="orange")
        ax.set_title("Golden Scale phi^(alpha n)")
        ax.set_xlabel("t")
        ax.set_ylabel("scale (log)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_05_mirror(self, path="fig_05_mirror.png"):
        f = self.x0.real
        r1 = self.e.R1(self.x0).real
        r2 = self.e.R2(self.x0).real
        r3 = self.e.R3(self.x0).real
        total = f + r1 + r2 + r3
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.e.idx, f, "k-", label="field")
        ax.plot(self.e.idx, r1, "o-", label="R1", markersize=3)
        ax.plot(self.e.idx, r2, "s-", label="R2", markersize=3)
        ax.plot(self.e.idx, r3, "^-", label="R3", markersize=3)
        ax.plot(self.e.idx, total, "r-", linewidth=2, label="sum = 0")
        ax.legend()
        ax.set_title("Alternating Mirror  R1 R2 R3 on Field")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_06_heal(self, path="fig_06_heal.png"):
        h = self.e.heal(self.x0)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.e.idx, h.real, label="Re(heal)")
        ax.plot(self.e.idx, h.imag, label="Im(heal)")
        ax.legend()
        ax.set_title("Heal Operator |phi|^2 phi")
        ax.set_xlabel("index")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_07_attractor(self, path="fig_07_attractor.png"):
        tr = self.e.trajectory(0.5 + 0.5j, 200)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(tr.real, tr.imag, "o-", markersize=3, color="green")
        ax.plot(tr[-1].real, tr[-1].imag, "r*", markersize=20, label="limit cycle")
        ax.set_title("Attractor T(z)=e^{(a+iB)z}")
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def fig_08_MS(self, path="fig_08_MS.png"):
        s = self.x0.real
        ms = self.e.MS(self.x0).real
        total = s + ms
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.e.idx, s, "o-", label="S(x)", markersize=3)
        ax.plot(self.e.idx, ms, "s-", label="MS(x)=C-S(-x)", markersize=3)
        ax.plot(self.e.idx, total, "k-", linewidth=2, label="S + MS")
        ax.legend()
        ax.set_title("MS Operator Cancellation")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def run(self):
        return {
            "axis": self.fig_01_axis(),
            "fibonacci": self.fig_02_fibonacci(),
            "pulses": self.fig_03_pulses(),
            "golden": self.fig_04_golden(),
            "mirror": self.fig_05_mirror(),
            "heal": self.fig_06_heal(),
            "attractor": self.fig_07_attractor(),
            "MS": self.fig_08_MS(),
        }
