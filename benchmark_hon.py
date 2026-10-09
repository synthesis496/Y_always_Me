#Copyright©2026 chutiphong bunloed
#All Rights Reserved.

import time
import numpy as np

class Benchmark:
    def __init__(self, L=64, repeats=10):
        self.e = _H(L=L)
        self.repeats = repeats
        self.x = np.exp(-self.e.idx ** 2 / 100.0).astype(complex)

    def _time(self, fn, *args):
        s = np.zeros(self.repeats)
        for i in range(self.repeats):
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            s[i] = t1 - t0
        return float(np.mean(s))

    def b_fib_100(self):
        return self._time(self.e.fib, 100)

    def b_fib_1000(self):
        return self._time(self.e.fib, 1000)

    def b_pulse(self):
        return self._time(self.e.pulse, 1.0)

    def b_pulse_index(self):
        return self._time(self.e.pulse_index, 1.0)

    def b_golden(self):
        return self._time(self.e.golden, 5.0)

    def b_MS(self):
        return self._time(self.e.MS, self.x)

    def b_R1(self):
        return self._time(self.e.R1, self.x)

    def b_R2(self):
        return self._time(self.e.R2, self.x)

    def b_R3(self):
        return self._time(self.e.R3, self.x)

    def b_mirror_mod0(self):
        return self._time(self.e.mirror_at, self.x, 0)

    def b_mirror_mod1(self):
        return self._time(self.e.mirror_at, self.x, 1)

    def b_mirror_mod2(self):
        return self._time(self.e.mirror_at, self.x, 2)

    def b_heal(self):
        return self._time(self.e.heal, self.x)

    def b_attractor(self):
        return self._time(self.e.attractor_map, 0.5 + 0.5j)

    def b_cycle(self):
        return self._time(self.e.mirror_cycle_sum, self.x)

    def b_spectrum_64(self):
        return self._time(self.e.spectrum, 1.0, 64)

    def b_spectrum_256(self):
        return self._time(self.e.spectrum, 1.0, 256)

    def run(self):
        return {
            "fib_100": self.b_fib_100(),
            "fib_1000": self.b_fib_1000(),
            "pulse": self.b_pulse(),
            "pulse_index": self.b_pulse_index(),
            "golden": self.b_golden(),
            "MS": self.b_MS(),
            "R1": self.b_R1(),
            "R2": self.b_R2(),
            "R3": self.b_R3(),
            "mirror_mod0": self.b_mirror_mod0(),
            "mirror_mod1": self.b_mirror_mod1(),
            "mirror_mod2": self.b_mirror_mod2(),
            "heal": self.b_heal(),
            "attractor": self.b_attractor(),
            "cycle": self.b_cycle(),
            "spectrum_64": self.b_spectrum_64(),
            "spectrum_256": self.b_spectrum_256(),
        }
