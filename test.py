#Copyright©2026.chutiphong bunloed
#All Rights Reserved.

import numpy as np

class _H:
    def __init__(self, L=64):
        self.L = L
        self.idx = np.arange(-L, L + 1)
        self.N = len(self.idx)
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.psi = (1.0 - np.sqrt(5.0)) / 2.0
        self.C = 0.0

    def fib(self, n):
        k = np.arange(n)
        return np.round(np.real((self.phi ** k - self.psi ** k) / np.sqrt(5.0))).astype(int)

    def pulse(self, t, n=20, width=0.05):
        p = self.fib(n)
        d = t - p
        return float(np.sum(np.exp(-(d ** 2) / (2.0 * width ** 2))))

    def pulse_index(self, t, n=20):
        p = self.fib(n)
        return int(np.argmin(np.abs(p - t)))

    def golden(self, t, n=20, alpha=0.5):
        i = self.pulse_index(t, n)
        return float(self.phi ** (alpha * i))

    def MS(self, x):
        return self.C - np.flip(x)

    def R1(self, x):
        return -np.flip(x)

    def R2(self, x):
        return np.flip(x)

    def R3(self, x):
        return -x

    def mirror_at(self, x, mode_index):
        m = mode_index % 3
        return (m == 0) * self.R1(x) + (m == 1) * self.R2(x) + (m == 2) * self.R3(x)

    def heal(self, x, g=0.1):
        return g * (np.abs(x) ** 2) * x

    def attractor_map(self, z, a=0.1, beta=1.0):
        return np.exp((a + 1j * beta) * z)

    def mirror_cycle_sum(self, x):
        return x + self.R1(x) + self.R2(x) + self.R3(x)


class TestSuite:
    def __init__(self, L=64):
        self.e = _H(L=L)
        self.x = np.exp(-self.e.idx ** 2 / 100.0).astype(complex)

    def t_fib_seq(self):
        return bool(np.array_equal(self.e.fib(10), np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])))

    def t_phi(self):
        return bool(abs(self.e.phi - 1.6180339887) < 1e-6)

    def t_psi(self):
        return bool(abs(self.e.psi + 0.6180339887) < 1e-6)

    def t_fib_ratio(self):
        s = self.e.fib(15)
        return bool(abs(s[-1] / s[-2] - self.e.phi) < 0.5)

    def t_boundary_low(self):
        return bool(self.e.idx[0] == -64)

    def t_boundary_high(self):
        return bool(self.e.idx[-1] == 64)

    def t_centre(self):
        return bool(self.e.idx[self.e.N // 2] == 0)

    def t_site_count(self):
        return bool(self.e.N == 129)

    def t_pulse_finite(self):
        return bool(np.isfinite(self.e.pulse(1.0)))

    def t_pulse_positive(self):
        return bool(self.e.pulse(0.0) > 0.0)

    def t_golden_growth(self):
        return bool(self.e.golden(5.0) > self.e.golden(2.0))

    def t_golden_positive(self):
        return bool(self.e.golden(2.0) > 0.0)

    def t_R1_inv(self):
        return bool(np.allclose(self.e.R1(self.e.R1(self.x)), self.x))

    def t_R2_inv(self):
        return bool(np.allclose(self.e.R2(self.e.R2(self.x)), self.x))

    def t_R3_inv(self):
        return bool(np.allclose(self.e.R3(self.e.R3(self.x)), self.x))

    def t_R_sum(self):
        return bool(np.allclose(self.e.R1(self.x) + self.e.R2(self.x) + self.e.R3(self.x), -self.x))

    def t_cycle_zero(self):
        return bool(np.allclose(self.e.mirror_cycle_sum(self.x), np.zeros_like(self.x)))

    def t_MS_involution(self):
        return bool(np.allclose(self.e.MS(self.e.MS(self.x)), self.x))

    def t_heal_finite(self):
        return bool(np.all(np.isfinite(self.e.heal(self.x))))

    def t_heal_nonzero(self):
        return bool(np.any(np.abs(self.e.heal(self.x)) > 0.0))

    def t_attractor_finite(self):
        return bool(np.isfinite(self.e.attractor_map(0.5 + 0.5j)))

    def t_attractor_bounded(self):
        vals = [np.abs(self.e.attractor_map(x + 0.5j)) for x in np.linspace(-2.0, 2.0, 8)]
        return bool(np.all(np.array(vals) < 1e3))

    def t_mirror_mod0(self):
        return bool(np.allclose(self.e.mirror_at(self.x, 0), self.e.R1(self.x)))

    def t_mirror_mod1(self):
        return bool(np.allclose(self.e.mirror_at(self.x, 1), self.e.R2(self.x)))

    def t_mirror_mod2(self):
        return bool(np.allclose(self.e.mirror_at(self.x, 2), self.e.R3(self.x)))

    def t_mirror_mod3(self):
        return bool(np.allclose(self.e.mirror_at(self.x, 3), self.e.R1(self.x)))

    def run(self):
        return {
            "fib_seq": self.t_fib_seq(),
            "phi": self.t_phi(),
            "psi": self.t_psi(),
            "fib_ratio": self.t_fib_ratio(),
            "boundary_low": self.t_boundary_low(),
            "boundary_high": self.t_boundary_high(),
            "centre": self.t_centre(),
            "site_count": self.t_site_count(),
            "pulse_finite": self.t_pulse_finite(),
            "pulse_positive": self.t_pulse_positive(),
            "golden_growth": self.t_golden_growth(),
            "golden_positive": self.t_golden_positive(),
            "R1_inv": self.t_R1_inv(),
            "R2_inv": self.t_R2_inv(),
            "R3_inv": self.t_R3_inv(),
            "R_sum": self.t_R_sum(),
            "cycle_zero": self.t_cycle_zero(),
            "MS_involution": self.t_MS_involution(),
            "heal_finite": self.t_heal_finite(),
            "heal_nonzero": self.t_heal_nonzero(),
            "attractor_finite": self.t_attractor_finite(),
            "attractor_bounded": self.t_attractor_bounded(),
            "mirror_mod0": self.t_mirror_mod0(),
            "mirror_mod1": self.t_mirror_mod1(),
            "mirror_mod2": self.t_mirror_mod2(),
            "mirror_mod3": self.t_mirror_mod3(),
        }
