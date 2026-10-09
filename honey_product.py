#Copyright (c) 2026 [Chutiphong Bunloed]            All Rights Reserved.
#All Right Reserved

import numpy as np

class HoneycombField:
    def __init__(self, L=64, tau0=1.0, alpha=0.5, gamma=0.1, a=0.1, beta=1.0, C=0.0):
        self.L = L
        self.idx = np.arange(-L, L + 1)
        self.N = len(self.idx)
        self.tau0 = tau0
        self.alpha = alpha
        self.gamma = gamma
        self.a = a
        self.beta = beta
        self.C = C
        self.phi = (1.0 + np.sqrt(5.0)) / 2.0
        self.psi = (1.0 - np.sqrt(5.0)) / 2.0
        self.sub_A = self.idx % 2 == 0
        self.sub_B = self.idx % 2 != 0
        self.centre = 0
        self.fold_low = -L
        self.fold_high = L

    def fib(self, n):
        k = np.arange(n)
        return np.round(np.real((self.phi ** k - self.psi ** k) / np.sqrt(5.0))).astype(int)

    def pulse(self, t, n=20, width=0.05):
        p = self.fib(n) * self.tau0
        d = t - p
        return float(np.sum(np.exp(-(d ** 2) / (2.0 * width ** 2))))

    def pulse_index(self, t, n=20):
        p = self.fib(n) * self.tau0
        return int(np.argmin(np.abs(p - t)))

    def golden(self, t, n=20):
        i = self.pulse_index(t, n)
        return float(self.phi ** (self.alpha * i))

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
        r0 = self.R1(x)
        r1 = self.R2(x)
        r2 = self.R3(x)
        return (m == 0) * r0 + (m == 1) * r1 + (m == 2) * r2

    def mirror_gated(self, x, t, n=20):
        env = self.pulse(t, n)
        i = self.pulse_index(t, n)
        return self.mirror_at(x, i) * env

    def heal_gated(self, x, t, n=20):
        env = self.pulse(t, n)
        return self.gamma * env * (np.abs(x) ** 2) * x

    def expansion_gated(self, x, t, n=20):
        env = self.pulse(t, n)
        s = self.golden(t, n)
        return x * (s - 1.0) * env

    def attractor_map(self, z):
        return np.exp((self.a + 1j * self.beta) * z)

    def attractor_gated(self, x, t, n=20):
        env = self.pulse(t, n)
        amp = float(np.mean(np.abs(x)))
        phase = float(np.mean(np.angle(x)))
        z = amp + 1j * phase
        zn = self.attractor_map(z)
        return x * (np.abs(zn) - amp) * env

    def evolve(self, x0, t_final, n=20, dt=0.01):
        x = np.array(x0, dtype=complex)
        steps = int(t_final / dt)
        for i in range(steps):
            t = i * dt
            x = x + self.mirror_gated(x, t, n) * dt
            x = x + self.heal_gated(x, t, n) * dt
            x = x + self.expansion_gated(x, t, n) * dt
            x = x + self.attractor_gated(x, t, n) * dt
        return x

    def trajectory(self, z0, steps=200):
        z = z0
        tr = np.zeros(steps, dtype=complex)
        for i in range(steps):
            z = self.attractor_map(z)
            tr[i] = z
        return tr

    def spectrum(self, w, modes=64):
        m = np.arange(1, modes + 1)
        amp = self.phi ** (-self.alpha * m)
        return float(np.sum(amp * np.cos(self.phi ** m * w * self.tau0 + m * np.pi / 4)))

    def mirror_cycle_sum(self, x):
        return x + self.R1(x) + self.R2(x) + self.R3(x)

    def residual(self, x):
        return float(np.abs(np.sum(self.mirror_cycle_sum(x))))

    def norm_growth(self, x0, t_final, n=20, dt=0.01):
        x = self.evolve(x0, t_final, n, dt)
        return float(np.linalg.norm(x) / np.linalg.norm(x0))
