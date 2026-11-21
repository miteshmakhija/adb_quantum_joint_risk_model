"""Classical sampling estimators.

The first four follow the challenge text (paths of three annual states).
The last two use the shock-conditioned model and work for any portfolio size.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from scipy.stats import qmc

from .estimators import Estimate, Estimator
from .exact import LatentEnumerationSolver, _JointStateSpace
from .latent import ShockLatentModel
from .losses import LossModel, Summary
from .perils import MAX_STEPS, QUAKE, RAIN
from .portfolio import Portfolio
from .targets import Target

Z95 = 1.959964


def _mean_estimate(values: np.ndarray, weights_sum: float = 1.0, method: str = ""):
    n = len(values)
    return values.mean() * weights_sum, Z95 * values.std(ddof=1) / np.sqrt(n) * weights_sum


class AnnualPathSpace:
    """Annual joint distribution and the mapping from 3-year paths to term summaries."""

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.losses = LossModel(portfolio)
        self.space = _JointStateSpace(portfolio.size)
        self.p = LatentEnumerationSolver(portfolio, years=1).state_pmf()
        self.p = self.p / self.p.sum()
        self.cdf = np.cumsum(self.p)
        self.zero = int(self.space.index_of(np.zeros((1, portfolio.size), int),
                                            np.zeros((1, portfolio.size), int))[0])
        self.state_loss = self.losses.country_losses(self.space.steps, self.space.rain).sum(-1)

    def draw(self, u: np.ndarray, cdf: np.ndarray | None = None) -> np.ndarray:
        cdf = self.cdf if cdf is None else cdf
        return np.minimum(np.searchsorted(cdf, u, side="right"), len(cdf) - 1)

    def summarize(self, paths: np.ndarray) -> Summary:
        steps = np.minimum(self.space.steps[paths].sum(1), MAX_STEPS)
        rain = np.minimum(self.space.rain[paths].sum(1), 1)
        return self.losses.summarize(steps, rain)


class _PathEstimator(Estimator):
    def __init__(self, portfolio: Portfolio, space: AnnualPathSpace | None = None):
        self.space = space or AnnualPathSpace(portfolio)
        self.years = portfolio.years


class MonteCarlo(_PathEstimator):
    name = "Monte Carlo"

    def run(self, targets, effort, rng):
        n = int(effort)
        s = self.space.summarize(self.space.draw(rng.random((n, self.years))))
        return {t.name: Estimate(*_mean_estimate(t.payoff(s)), n, self.name) for t in targets}


class ScrambledSobol(_PathEstimator):
    name = "Scrambled Sobol"

    def __init__(self, portfolio, space=None, scrambles: int = 8):
        super().__init__(portfolio, space)
        self.scrambles = scrambles

    def run(self, targets, effort, rng):
        per = max(2, int(effort) // self.scrambles)
        means = {t.name: [] for t in targets}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.scrambles):
                u = qmc.Sobol(d=self.years, scramble=True, seed=rng).random(per)
                s = self.space.summarize(self.space.draw(u))
                for t in targets:
                    means[t.name].append(t.payoff(s).mean())
        tq = stats.t.ppf(0.975, self.scrambles - 1)
        out = {}
        for t in targets:
            m = np.array(means[t.name])
            out[t.name] = Estimate(m.mean(), tq * m.std(ddof=1) / np.sqrt(len(m)),
                                   per * self.scrambles, self.name)
        return out


class FirstPayoutStratified(_PathEstimator):
    """Strata = first year with any payout. A path with no payout contributes zero."""
    name = "Stratified"

    def run(self, targets, effort, rng):
        sp, y = self.space, self.years
        p0 = sp.p[sp.zero]
        cond = sp.p.copy()
        cond[sp.zero] = 0.0
        cond_cdf = np.cumsum(cond / cond.sum())
        per = max(2, int(effort) // y)
        totals = {t.name: 0.0 for t in targets}
        var = {t.name: 0.0 for t in targets}
        zero_summary = sp.summarize(np.full((1, y), sp.zero))
        for k in range(1, y + 1):
            w = p0 ** (k - 1) * (1 - p0)
            paths = np.full((per, y), sp.zero)
            paths[:, k - 1] = sp.draw(rng.random(per), cond_cdf)
            if k < y:
                paths[:, k:] = sp.draw(rng.random((per, y - k)))
            s = sp.summarize(paths)
            for t in targets:
                v = t.payoff(s)
                totals[t.name] += w * v.mean()
                var[t.name] += w ** 2 * v.var(ddof=1) / per
        out = {}
        for t in targets:
            value = totals[t.name] + p0 ** y * float(t.payoff(zero_summary)[0])
            out[t.name] = Estimate(value, Z95 * np.sqrt(var[t.name]), per * y, self.name)
        return out


class LossTiltedImportance(_PathEstimator):
    """Proposal per year = half the true law, half the law tilted by principal loss."""
    name = "Importance sampling"

    def run(self, targets, effort, rng):
        sp, n = self.space, int(effort)
        tilt = sp.p * sp.state_loss
        q = 0.5 * sp.p + 0.5 * tilt / tilt.sum()
        paths = sp.draw(rng.random((n, self.years)), np.cumsum(q))
        w = np.prod(sp.p[paths] / q[paths], axis=1)
        s = sp.summarize(paths)
        return {t.name: Estimate(*_mean_estimate(w * t.payoff(s)), n, self.name) for t in targets}


class LatentMonteCarlo(Estimator):
    """Draw the shared shocks, then each country given its shocks. Any n."""
    name = "Latent Monte Carlo"

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.model = ShockLatentModel(portfolio)
        self.losses = LossModel(portfolio)
        self.cdf = self.model.conditional_cdf_table()
        self.quake_idx = [s for s, sh in enumerate(portfolio.shocks) if sh.peril == QUAKE]
        self.rain_idx = [s for s, sh in enumerate(portfolio.shocks) if sh.peril == RAIN]
        self.q_inc, self.r_inc = portfolio.incidence(QUAKE), portfolio.incidence(RAIN)

    def sample_shocks(self, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        pf, yrs = self.portfolio, self.model.years
        qc = np.stack([rng.poisson(yrs * pf.shocks[s].rate, n) for s in self.quake_idx], 1) \
            if self.quake_idx else np.zeros((n, 0), int)
        rf = np.stack([rng.random(n) < 1 - np.exp(-yrs * pf.shocks[s].rate)
                       for s in self.rain_idx], 1).astype(int) \
            if self.rain_idx else np.zeros((n, 0), int)
        return qc, rf

    def sample_countries(self, qc, rf, rng) -> Summary:
        n = qc.shape[0]
        k = np.minimum(qc @ self.q_inc, MAX_STEPS) if qc.shape[1] else \
            np.zeros((n, self.portfolio.size), int)
        hit = ((rf @ self.r_inc) > 0).astype(int) if rf.shape[1] else \
            np.zeros((n, self.portfolio.size), int)
        cdf = self.cdf[np.arange(self.portfolio.size)[None, :], k, hit]      # (n, m, 10)
        code = (rng.random((n, self.portfolio.size))[..., None] > cdf).sum(-1)
        code = np.minimum(code, cdf.shape[-1] - 1)
        return self.losses.summarize(code // 2, code % 2)

    def run(self, targets, effort, rng):
        n = int(effort)
        s = self.sample_countries(*self.sample_shocks(n, rng), rng)
        return {t.name: Estimate(*_mean_estimate(t.payoff(s)), n, self.name) for t in targets}


class CalmPeeledMonteCarlo(LatentMonteCarlo):
    """Strongest classical sampler we use: the calm stratum (no shared shock fired) is
    computed exactly, and all samples go to the shock strata where the joint tail lives.
    """
    name = "Calm-peeled MC"

    def __init__(self, portfolio: Portfolio, calm):
        super().__init__(portfolio)
        self.calm = calm            # CalmStratumSolver

    def _sample_shocked(self, n: int, rng):
        """Rejection-free draw of shock patterns conditional on at least one firing."""
        qc_all, rf_all, got = [], [], 0
        while got < n:
            m = max(64, int(1.3 * (n - got) / max(1 - self.calm.weight, 1e-12)))
            m = min(m, 5_000_000)
            qc, rf = self.sample_shocks(m, rng)
            keep = (qc.sum(1) + rf.sum(1)) > 0
            qc_all.append(qc[keep]); rf_all.append(rf[keep]); got += int(keep.sum())
        return np.concatenate(qc_all)[:n], np.concatenate(rf_all)[:n]

    def run(self, targets, effort, rng):
        w_calm = self.calm.weight
        if w_calm >= 1.0 - 1e-15:              # no shared shocks: the answer is exact
            return {t.name: Estimate(self.calm.conditional(t), 0.0, 0, self.name)
                    for t in targets}
        n = int(effort)
        s = self.sample_countries(*self._sample_shocked(n, rng), rng)
        out = {}
        for t in targets:
            m, h = _mean_estimate(t.payoff(s), 1 - w_calm)
            out[t.name] = Estimate(w_calm * self.calm.conditional(t) + m, h, n, self.name)
        return out
