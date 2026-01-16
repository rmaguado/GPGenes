from __future__ import annotations
import random
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
        limit: float = 10.0,
        decay: float = 0.1,
        noise_sigma: float = 0.02,
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
    def hill(x, K=1.0, n=2, scale=10.0):
        return scale * (x**n / (K**n + x**n))

    def compute_input(self):
        if self.knocked_out:
            return 0.0

        total = 0.0
        for reg in self.regulators:
            x = reg.source.prev_value
            h = self.hill(x)
            total += reg.strength * h
        return total

    def step(self, delta, input_signal):
        if self.knocked_out:
            self.value = 0.0
            self.history.append(0.0)
            return

        noise = random.gauss(0.0, self.noise_sigma)
        dv = delta * (self.base + input_signal - self.decay * self.value + noise)
        raw = max(0.0, self.value + dv)
        self.value = self.hill(raw, K=self.limit / 2.0, scale=self.limit)
        self.history.append(self.value)

    def sync(self):
        self.prev_value = self.value
