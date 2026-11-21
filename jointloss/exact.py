"""Exact solvers.

PathConvolutionSolver   the challenge's own method (annual states, 3 yearly passes)
LatentEnumerationSolver sum over shock patterns of independent country products
ShockGraphContraction   our scalable solver: sweeps countries left to right and keeps
                        only the shocks that are still "open". Cost grows with the width
                        of the shock graph, not with 10^n.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .latent import STATES_PER_COUNTRY, ShockLatentModel
from .losses import LossModel, Summary
from .perils import MAX_STEPS, QUAKE
from .portfolio import Portfolio
from .targets import Target


class ExactSolver(ABC):
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.losses = LossModel(portfolio)
        self._dist: tuple[np.ndarray, Summary] | None = None

    @abstractmethod
    def _distribution(self) -> tuple[np.ndarray, Summary]:
        """Probabilities and the matching term summaries."""

    def distribution(self):
        if self._dist is None:
            self._dist = self._distribution()
        return self._dist

    def value(self, target: Target) -> float:
        prob, summary = self.distribution()
        return float((prob * target.payoff(summary)).sum())


class _JointStateSpace:
    """All 10^n joint country states, used by the brute-force solvers."""

    def __init__(self, n: int):
        if n > 6:
            raise ValueError("brute-force state space is limited to n <= 6")
        self.n = n
        idx = np.indices((STATES_PER_COUNTRY,) * n).reshape(n, -1).T     # (S, n)
        self.steps, self.rain = idx // 2, idx % 2

    def index_of(self, steps: np.ndarray, rain: np.ndarray) -> np.ndarray:
        codes = steps * 2 + rain
        return np.ravel_multi_index(tuple(codes.T), (STATES_PER_COUNTRY,) * self.n)


class LatentEnumerationSolver(ExactSolver):
    def __init__(self, portfolio: Portfolio, years: int | None = None):
        super().__init__(portfolio)
        self.model = ShockLatentModel(portfolio, years)
        self.space = _JointStateSpace(portfolio.size)

    def state_pmf(self) -> np.ndarray:
        n, total = self.portfolio.size, 0.0
        for values, weight in self.model.latent_patterns():
            parts = [self.model.country_pmf(i, *self.model.country_inputs(i, values)).ravel()
                     for i in range(n)]
            joint = parts[0]
            for p in parts[1:]:
                joint = np.multiply.outer(joint, p)
            total = total + weight * joint.ravel()
        return total

    def _distribution(self):
        return self.state_pmf(), self.losses.summarize(self.space.steps, self.space.rain)


class PathConvolutionSolver(ExactSolver):
    """Reference method from the challenge text: annual states, capped sum over years."""

    def __init__(self, portfolio: Portfolio):
        super().__init__(portfolio)
        self.annual = LatentEnumerationSolver(portfolio, years=1).state_pmf()
        self.space = _JointStateSpace(portfolio.size)

    def _distribution(self):
        sp, term = self.space, np.zeros_like(self.annual)
        term[sp.index_of(np.zeros((1, sp.n), int), np.zeros((1, sp.n), int))] = 1.0
        support = np.flatnonzero(self.annual > 0)
        for _ in range(self.portfolio.years):
            nxt = np.zeros_like(term)
            for s in np.flatnonzero(term > 0):
                steps = np.minimum(sp.steps[s] + sp.steps[support], MAX_STEPS)
                rain = np.minimum(sp.rain[s] + sp.rain[support], 1)
                np.add.at(nxt, sp.index_of(steps, rain), term[s] * self.annual[support])
            term = nxt
        return term, self.losses.summarize(sp.steps, sp.rain)


@dataclass
class StratumAmplitude:
    label: str
    weight: float            # P(stratum), known exactly
    conditional: float       # E[payoff | stratum] / bound, in [0, 1]


class ShockGraphContraction(ExactSolver):
    """Exact, linear in n for portfolios whose shocks link nearby countries.

    The state carried along the sweep is a tensor over
    (number of shocks fired so far, loss on an integer grid, attached count, exhausted count)
    for every combination of open shock values. The first axis lets us read off the
    probability and conditional value of each shock stratum for free.
    """

    def __init__(self, portfolio: Portfolio, count_cap: int = 2, max_bucket: int = 2,
                 memory_limit_gb: float = 4.0):
        super().__init__(portfolio)
        self.memory_limit_gb = memory_limit_gb
        self.model = ShockLatentModel(portfolio)
        self.count_cap, self.max_bucket = count_cap, max_bucket
        self.width = 0
        self._tensor: np.ndarray | None = None

    # sweep ----------------------------------------------------------------------------
    def _bump(self, x: np.ndarray, axis: int, by: int) -> np.ndarray:
        if by == 0:
            return x
        cap = x.shape[axis] - 1
        y = np.zeros_like(x)
        src = [slice(None)] * x.ndim
        dst = [slice(None)] * x.ndim
        src[axis], dst[axis] = slice(0, cap), slice(by, cap + 1)
        y[tuple(dst)] = x[tuple(src)]
        src[axis], dst[axis] = slice(cap, cap + 1), slice(cap, cap + 1)
        y[tuple(dst)] += x[tuple(src)]
        return y

    def _absorb_country(self, x: np.ndarray, entries) -> np.ndarray:
        out = np.zeros_like(x)
        for p, l, att, exh in entries:
            y = self._bump(self._bump(x, 2, att), 3, exh)
            if l:
                out[:, l:] += p * y[:, :-l]
            else:
                out += p * y
        return out

    def peak_memory_gb(self) -> float:
        """Memory of the widest step of the sweep, worked out before running it."""
        pf = self.portfolio
        first = {s: min(sh.members) for s, sh in enumerate(pf.shocks)}
        last = {s: max(sh.members) for s, sh in enumerate(pf.shocks)}
        cell = (self.max_bucket + 1) * self.losses.grid_size() * (self.count_cap + 1) ** 2 * 8
        peak, open_ = 0, set()
        for i in range(pf.size):
            open_ |= {s for s in first if first[s] == i}
            combos = int(np.prod([len(self.model.shock_prior(s)) for s in open_])) if open_ else 1
            peak = max(peak, combos)
            self.width = max(self.width, len(open_))
            open_ -= {s for s in last if last[s] == i}
        return 2 * peak * cell / 1e9          # old and new state live together

    def tensor(self) -> np.ndarray:
        if self._tensor is not None:
            return self._tensor
        if self.peak_memory_gb() > self.memory_limit_gb:
            raise MemoryError(f"exact sweep needs about {self.peak_memory_gb():,.1f} GB")
        pf, n = self.portfolio, self.portfolio.size
        first = {s: min(sh.members) for s, sh in enumerate(pf.shocks)}
        last = {s: max(sh.members) for s, sh in enumerate(pf.shocks)}
        start = np.zeros((self.max_bucket + 1, self.losses.grid_size(),
                          self.count_cap + 1, self.count_cap + 1))
        start[0, 0, 0, 0] = 1.0
        state: dict[tuple, np.ndarray] = {(): start}

        for i in range(n):
            for s in [s for s in first if first[s] == i]:          # open new shocks
                prior, nxt = self.model.shock_prior(s), {}
                for key, x in state.items():
                    for v, pv in enumerate(prior):
                        if pv == 0:
                            continue
                        k2 = key + ((s, v),)
                        nxt[k2] = nxt.get(k2, 0) + pv * self._bump(x, 0, int(v > 0))
                state = nxt
            self.width = max(self.width, len(next(iter(state))))
            nxt = {}
            for key, x in state.items():                             # add country i
                values = dict(key)
                full = [values.get(s, 0) for s in range(len(pf.shocks))]
                pmf = self.model.country_pmf(i, *self.model.country_inputs(i, full))
                nxt[key] = self._absorb_country(x, self.losses.country_entries(i, pmf))
            state = nxt
            for s in [s for s in last if last[s] == i]:            # close finished shocks
                nxt = {}
                for key, x in state.items():
                    k2 = tuple(kv for kv in key if kv[0] != s)
                    nxt[k2] = nxt.get(k2, 0) + x
                state = nxt
        self._tensor = state[()]
        return self._tensor

    def _grid_summary(self) -> Summary:
        g, c = self.losses.grid_size(), self.count_cap + 1
        return Summary(total=(np.arange(g) * self.losses.grid_unit)[:, None, None] * 1.0,
                       attached=np.arange(c)[None, :, None],
                       exhausted=np.arange(c)[None, None, :])

    def _distribution(self):
        return self.tensor().sum(0), self._grid_summary()

    def value(self, target: Target) -> float:
        if target.count_needed > self.count_cap:
            raise ValueError("raise count_cap for this target")
        return super().value(target)

    def strata(self, target: Target) -> list[StratumAmplitude]:
        """Split by number of shared shocks that fired in the term (0, 1, ..., max)."""
        t, summary = self.tensor(), self._grid_summary()
        pay = target.payoff(summary) / target.bound(self.losses)
        out = []
        for b in range(t.shape[0]):
            w = float(t[b].sum())
            if w > 0:
                label = f"{b}+ shocks" if b == self.max_bucket else f"{b} shocks"
                out.append(StratumAmplitude(label, w, float((t[b] * pay).sum() / w)))
        return out


class CalmStratumSolver(ShockGraphContraction):
    """The stratum where no shared shock fires. Countries are independent there, so the
    conditional law is a plain convolution over countries: O(n), whatever the shock graph.
    """

    def __init__(self, portfolio: Portfolio, count_cap: int = 2):
        super().__init__(portfolio, count_cap=count_cap, max_bucket=0)
        self.weight = float(np.prod([self.model.shock_prior(s)[0]
                                     for s in range(len(portfolio.shocks))]))

    def tensor(self) -> np.ndarray:
        if self._tensor is None:
            x = np.zeros((1, self.losses.grid_size(), self.count_cap + 1, self.count_cap + 1))
            x[0, 0, 0, 0] = 1.0
            for i in range(self.portfolio.size):
                pmf = self.model.country_pmf(i, 0, False)
                x = self._absorb_country(x, self.losses.country_entries(i, pmf))
            self._tensor = x
        return self._tensor

    def conditional(self, target: Target) -> float:
        return self.value(target)
