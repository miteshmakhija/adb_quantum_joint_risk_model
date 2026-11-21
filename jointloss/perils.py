"""Peril covers and the small distribution helpers they need."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

MAX_STEPS = 4          # earthquake pays in quarter steps, 4 steps = full sub-limit
QUAKE, RAIN = "quake", "rain"


def capped_compound_poisson(rate: float, step_weights: np.ndarray, cap: int = MAX_STEPS) -> np.ndarray:
    """P(min(S, cap) = s) for S = sum of Poisson(rate) jumps with sizes 1..len(step_weights).

    Uses the Panjer recursion, so it is exact (no truncation of event counts).
    """
    g = np.zeros(cap + 1)
    g[0] = exp(-rate)
    for s in range(1, cap):
        g[s] = rate / s * sum(j * step_weights[j - 1] * g[s - j]
                              for j in range(1, min(s, len(step_weights)) + 1))
    g[cap] = max(0.0, 1.0 - g[:cap].sum())
    return g


def capped_convolve(p: np.ndarray, q: np.ndarray, cap: int = MAX_STEPS) -> np.ndarray:
    """Distribution of min(X + Y, cap) for independent X ~ p and Y ~ q on 0..cap."""
    full = np.convolve(p, q)
    out = full[:cap + 1].copy()
    out[cap] += full[cap + 1:].sum()
    return out


@dataclass(frozen=True)
class EarthquakeCover:
    """Quarter-step earthquake cover, fitted to the annual exceedance curve.

    exceedance = P(annual payout >= 25%, 50%, 75%, 100% of the sub-limit).
    """
    limit: float
    exceedance: tuple[float, float, float, float]

    def jump_rates(self) -> np.ndarray:
        """Rates lambda_1..lambda_4 of events paying 1..4 quarter steps (eqs. 2-3)."""
        t1, t2, t3, t4 = self.exceedance
        p0, p1, p2, p3 = 1 - t1, t1 - t2, t2 - t3, t3 - t4
        total = -log(p0)
        l1 = p1 / p0
        l2 = p2 / p0 - l1 ** 2 / 2
        l3 = p3 / p0 - l1 * l2 - l1 ** 3 / 6
        return np.array([l1, l2, l3, total - l1 - l2 - l3])

    @property
    def total_rate(self) -> float:
        return float(self.jump_rates().sum())

    @property
    def step_weights(self) -> np.ndarray:
        rates = self.jump_rates()
        return rates / rates.sum()

    @property
    def step_value(self) -> float:
        return self.limit / MAX_STEPS


@dataclass(frozen=True)
class RainCover:
    """Precipitation cover that pays its full sub-limit at most once per term."""
    limit: float
    annual_prob: float
    rate_multiplier: float = 1.0

    @property
    def total_rate(self) -> float:
        return -log(1 - self.annual_prob) * self.rate_multiplier
