from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class Regulation:
    source: Gene
    strength: float


class Gene:
    def __init__(
        self,
        gid: int,
        base: float = 0.0,
        limit: float = 1.0,
        decay: float = 0.1,
        noise_sigma: float = 0.0,
    ):
        self.id = gid
        self.base = base
        self.limit = limit
        self.decay = decay
        self.noise_sigma = noise_sigma

        self.value = 0.0
        self.prev_value = 0.0
        self.history = []
        self.regulators: list[Regulation] = []

        self.knocked_out = False

    def knock_out(self):
        self.knocked_out = True
        self.value = 0.0
        self.prev_value = 0.0

    def reset(self):
        self.value = self.base + random.uniform(-0.1, 0.1)
        if self.knocked_out:
            self.value = 0.0
        self.prev_value = self.value
        self.history.clear()

    @staticmethod
    def hill(x, K=1.0, n=2, scale=1.0):
        return scale * (x**n / (K**n + x**n))

    def compute_input(self):
        if self.knocked_out:
            return 0.0

        total = 0.0
        for reg in self.regulators:
            x = reg.source.prev_value
            total += reg.strength * x
        return total

    def step(self, delta, input_signal):
        if self.knocked_out:
            self.value = 0.0
            self.history.append(0.0)
            return

        noise = random.gauss(0.0, self.noise_sigma)

        target = max(0.0, self.base + input_signal + noise)

        tau = 1.0 / max(self.decay, 1e-9)

        alpha = 1.0 - pow(np.e, -delta / tau)
        relaxed = self.value + alpha * (target - self.value)

        self.value = self.hill(relaxed, K=self.limit / 2.0, scale=self.limit)

        self.history.append(self.value)

    def sync(self):
        self.prev_value = self.value
