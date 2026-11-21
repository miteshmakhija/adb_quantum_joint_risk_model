"""Shock-conditioned view of the portfolio.

Two facts make this model easy to work with:

1. Given how many times each shared shock fired during the term, the countries are
   independent. All dependence sits in a handful of shock variables.
2. Annual caps followed by a term cap equal one cap on the term total, because payouts
   only go up. So the 3-year term is itself a compound Poisson model with 3x the rates,
   and the year-by-year convolution is not needed.

Shared quake counts can be stored as 0, 1, 2, 3 or "4 or more" without any error,
because each shared quake adds at least one quarter step and the cover caps at four.
"""
from __future__ import annotations

import itertools
from functools import lru_cache
from math import exp

import numpy as np
from scipy.stats import poisson

from .perils import MAX_STEPS, QUAKE, RAIN, capped_compound_poisson, capped_convolve
from .portfolio import Portfolio

STATES_PER_COUNTRY = (MAX_STEPS + 1) * 2       # (quake steps 0..4) x (rain 0/1)


class ShockLatentModel:
    def __init__(self, portfolio: Portfolio, years: int | None = None):
        self.portfolio = portfolio
        self.years = portfolio.years if years is None else years
        self._country_pmf = lru_cache(maxsize=None)(self._country_pmf_uncached)

    # shocks ---------------------------------------------------------------------------
    def shock_prior(self, s: int) -> np.ndarray:
        shock = self.portfolio.shocks[s]
        mean = self.years * shock.rate
        if shock.peril == QUAKE:
            pmf = poisson.pmf(np.arange(MAX_STEPS), mean)
            return np.append(pmf, max(0.0, 1.0 - pmf.sum()))
        fired = 1.0 - exp(-mean)
        return np.array([1.0 - fired, fired])

    def latent_patterns(self):
        """All shock patterns with their probabilities (small portfolios only)."""
        priors = [self.shock_prior(s) for s in range(len(self.portfolio.shocks))]
        for values in itertools.product(*[range(len(p)) for p in priors]):
            weight = float(np.prod([p[v] for p, v in zip(priors, values)]))
            if weight > 0:
                yield values, weight

    def country_inputs(self, country: int, values) -> tuple[int, bool]:
        """(shared quake count, shared rain hit) seen by one country."""
        quakes = sum(values[s] for s in self.portfolio.shocks_of(country, QUAKE))
        hit = any(values[s] > 0 for s in self.portfolio.shocks_of(country, RAIN))
        return min(quakes, MAX_STEPS), hit

    # countries ------------------------------------------------------------------------
    def country_pmf(self, country: int, shared_quakes: int, rain_hit: bool) -> np.ndarray:
        """P(quake steps = a, rain = r | shocks) as a (5, 2) array."""
        return self._country_pmf(country, shared_quakes, bool(rain_hit))

    def _country_pmf_uncached(self, country: int, shared_quakes: int, rain_hit: bool):
        pf, c = self.portfolio, self.portfolio.countries[country]
        steps = capped_compound_poisson(self.years * pf.own_rate(country, QUAKE),
                                        c.quake.step_weights)
        jump = np.r_[0.0, c.quake.step_weights]
        for _ in range(shared_quakes):
            steps = capped_convolve(steps, jump)
        p_rain = 1.0 if rain_hit else 1.0 - exp(-self.years * pf.own_rate(country, RAIN))
        return np.outer(steps, [1.0 - p_rain, p_rain])

    def conditional_cdf_table(self) -> np.ndarray:
        """(n, 5, 2, 10) table of cumulative country-state probabilities, for sampling."""
        n = self.portfolio.size
        table = np.zeros((n, MAX_STEPS + 1, 2, STATES_PER_COUNTRY))
        for i in range(n):
            for k in range(MAX_STEPS + 1):
                for h in (0, 1):
                    table[i, k, h] = np.cumsum(self.country_pmf(i, k, h).ravel())
        return table
