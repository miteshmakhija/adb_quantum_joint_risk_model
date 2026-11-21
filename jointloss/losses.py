"""Mapping from country states (quake steps, rain flag) to money."""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd

import numpy as np

from .perils import MAX_STEPS
from .portfolio import Portfolio


@dataclass
class Summary:
    """What every focus target needs to know about a term outcome (arrays broadcast)."""
    total: np.ndarray          # portfolio principal loss, USD
    attached: np.ndarray       # number of countries with a payout
    exhausted: np.ndarray      # number of countries whose bond is fully used


class LossModel:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.step = np.array([c.quake.step_value for c in portfolio.countries])
        self.rain = np.array([c.rain.limit for c in portfolio.countries])
        self.principal = np.array([c.principal for c in portfolio.countries])

    def country_losses(self, steps: np.ndarray, rain: np.ndarray) -> np.ndarray:
        return steps * self.step + rain * self.rain

    def summarize(self, steps: np.ndarray, rain: np.ndarray) -> Summary:
        """steps, rain: integer arrays of shape (..., n)."""
        losses = self.country_losses(steps, rain)
        return Summary(total=losses.sum(-1),
                       attached=(losses > 0).sum(-1),
                       exhausted=(losses >= self.principal - 1e-6).sum(-1))

    # integer loss grid used by the contraction solver ---------------------------------
    @property
    def grid_unit(self) -> int:
        values = [int(round(v)) for v in np.r_[self.step, self.rain] if v > 0]
        return reduce(gcd, values)

    def grid_size(self) -> int:
        return int(round(self.portfolio.total_principal / self.grid_unit)) + 1

    def country_entries(self, i: int, pmf: np.ndarray):
        """(prob, loss index, attached, exhausted) for each state of country i."""
        unit, out = self.grid_unit, []
        for a in range(MAX_STEPS + 1):
            for r in (0, 1):
                p = pmf[a, r]
                if p <= 0:
                    continue
                loss = a * self.step[i] + r * self.rain[i]
                out.append((p, int(round(loss / unit)), int(loss > 0),
                            int(loss >= self.principal[i] - 1e-6)))
        return out
