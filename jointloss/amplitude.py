"""Iterative quantum amplitude estimation (Grinko et al., 2021) with pluggable oracles.

An oracle answers one question: "run Q^k A for `shots` shots and tell me how many times
the objective qubit read 1". IdealOracle does this for a noise-free device whose
amplitude is known; CircuitOracle (see circuits.py) does it from a real Qiskit circuit;
a hardware oracle would send the same circuits to a QPU.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import asin, log, pi, sqrt

import numpy as np
from scipy.stats import beta


class AmplitudeOracle(ABC):
    @abstractmethod
    def good_probability(self, power: int) -> float:
        """P(objective = 1) after Q^power A."""

    def good_counts(self, power: int, shots: int, rng: np.random.Generator) -> int:
        return int(rng.binomial(shots, min(max(self.good_probability(power), 0.0), 1.0)))


class IdealOracle(AmplitudeOracle):
    def __init__(self, amplitude: float):
        self.amplitude = float(np.clip(amplitude, 0.0, 1.0))
        self.theta = asin(sqrt(self.amplitude))

    def good_probability(self, power: int) -> float:
        return float(np.sin((2 * power + 1) * self.theta) ** 2)


@dataclass
class AEResult:
    estimate: float
    interval: tuple[float, float]
    queries: int             # sum over rounds of shots * (2k + 1)
    max_power: int


def _clopper_pearson(ones: int, shots: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if ones == 0 else beta.ppf(alpha / 2, ones, shots - ones + 1)
    hi = 1.0 if ones == shots else beta.ppf(1 - alpha / 2, ones + 1, shots - ones)
    return float(lo), float(hi)


class IterativeAmplitudeEstimation:
    def __init__(self, epsilon: float, alpha: float = 0.05, shots: int = 100,
                 min_ratio: float = 2.0):
        if not 0 < epsilon <= 0.5:
            raise ValueError("epsilon must be in (0, 0.5]")
        self.epsilon, self.alpha, self.shots, self.min_ratio = epsilon, alpha, shots, min_ratio

    def _next_power(self, k: int, upper: bool, lo: float, hi: float) -> tuple[int, bool]:
        old = 4 * k + 2
        scale = int(1 / (2 * (hi - lo)))
        scale -= (scale - 2) % 4
        while scale >= self.min_ratio * old:
            t_lo, t_hi = scale * lo - int(scale * lo), scale * hi - int(scale * hi)
            if t_lo <= t_hi <= 0.5 and t_lo <= 0.5:
                return (scale - 2) // 4, True
            if t_hi >= 0.5 and t_hi >= t_lo >= 0.5:
                return (scale - 2) // 4, False
            scale -= 4
        return k, upper

    def estimate(self, oracle: AmplitudeOracle, rng: np.random.Generator) -> AEResult:
        # theta is stored in units of a full turn: a = sin^2(2*pi*theta), theta in [0, 1/4]
        lo, hi = 0.0, 0.25
        k, upper, queries = 0, True, 0
        rounds = int(log(self.min_ratio * pi / 8 / self.epsilon) / log(self.min_ratio)) + 1
        history: list[tuple[int, int]] = []          # (power, ones)
        # Stop on the width of the amplitude interval. Qiskit stops on the angle interval,
        # which is far stricter than asked for when the amplitude is small.
        while (np.sin(2 * pi * hi) ** 2 - np.sin(2 * pi * lo) ** 2) / 2 > self.epsilon:
            k, upper = self._next_power(k, upper, lo, hi)
            ones = oracle.good_counts(k, self.shots, rng)
            history.append((k, ones))
            queries += self.shots * (2 * k + 1)
            # pool consecutive rounds that used the same power
            tail = 0
            for p, _ in reversed(history):
                if p != k:
                    break
                tail += 1
            pooled = sum(o for _, o in history[-tail:])
            a_lo, a_hi = _clopper_pearson(pooled, tail * self.shots, self.alpha / rounds)
            if upper:
                t_min, t_max = np.arccos(1 - 2 * a_lo) / (2 * pi), np.arccos(1 - 2 * a_hi) / (2 * pi)
            else:
                t_min = 1 - np.arccos(1 - 2 * a_hi) / (2 * pi)
                t_max = 1 - np.arccos(1 - 2 * a_lo) / (2 * pi)
            scale = 4 * k + 2
            new_hi = (int(scale * hi) + t_max) / scale
            new_lo = (int(scale * lo) + t_min) / scale
            lo, hi = max(lo, new_lo), min(hi, new_hi)
            if hi < lo:                                   # numerical guard
                lo, hi = sorted((lo, hi))
        a_lo, a_hi = np.sin(2 * pi * lo) ** 2, np.sin(2 * pi * hi) ** 2
        return AEResult(float((a_lo + a_hi) / 2), (float(a_lo), float(a_hi)), queries, k)
