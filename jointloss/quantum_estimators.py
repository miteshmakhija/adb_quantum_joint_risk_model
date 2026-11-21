"""Quantum estimators built on IQAE.

PlainQAE        one amplitude for the whole target (the textbook approach)
ShockSplitQAE   our approach:
                  1. the calm stratum (no shared shock) is solved exactly and classically,
                  2. each shock stratum gets its own QAE run on a loader conditioned on that
                     stratum, so the rare joint tail becomes a large amplitude,
                  3. the accuracy budget is split across strata from a cheap pilot run.

By default the oracles are ideal (noise-free device, amplitude taken from the exact
solver). Swap `oracle_factory` for CircuitOracle or a hardware backend without touching
the estimator logic.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .amplitude import AmplitudeOracle, IdealOracle, IterativeAmplitudeEstimation
from .estimators import Estimate, Estimator
from .exact import CalmStratumSolver, ShockGraphContraction, StratumAmplitude
from .targets import Target

OracleFactory = Callable[[Target, StratumAmplitude | None, float], AmplitudeOracle]


def ideal_oracle_factory(target: Target, stratum: StratumAmplitude | None,
                         amplitude: float) -> AmplitudeOracle:
    return IdealOracle(amplitude)


class PlainQAE(Estimator):
    name = "Plain QAE"
    effort_kind = "epsilon"

    def __init__(self, solver: ShockGraphContraction, shots: int = 100, alpha: float = 0.05,
                 oracle_factory: OracleFactory = ideal_oracle_factory):
        self.solver, self.shots, self.alpha = solver, shots, alpha
        self.oracle_factory = oracle_factory

    def run(self, targets, effort, rng):
        """effort = wanted half-width, as a fraction of each target's true size."""
        out = {}
        for t in targets:
            bound = t.bound(self.solver.losses)
            a = self.solver.value(t) / bound
            eps = float(np.clip(effort * a, 1e-9, 0.5))
            res = IterativeAmplitudeEstimation(eps, self.alpha, self.shots).estimate(
                self.oracle_factory(t, None, a), rng)
            out[t.name] = Estimate(res.estimate * bound,
                                   (res.interval[1] - res.interval[0]) / 2 * bound,
                                   res.queries, self.name, res.max_power)
        return out


class ShockSplitQAE(Estimator):
    name = "Shock-split QAE (ours)"
    effort_kind = "epsilon"

    def __init__(self, solver: ShockGraphContraction, calm: CalmStratumSolver,
                 shots: int = 100, alpha: float = 0.05, pilot_share: float = 0.3,
                 oracle_factory: OracleFactory = ideal_oracle_factory):
        self.solver, self.calm = solver, calm
        self.shots, self.alpha, self.pilot_share = shots, alpha, pilot_share
        self.oracle_factory = oracle_factory

    def _iqae(self, target, stratum, eps, rng):
        eps = float(np.clip(eps, 1e-9, 0.5))
        return IterativeAmplitudeEstimation(eps, self.alpha, self.shots).estimate(
            self.oracle_factory(target, stratum, stratum.conditional), rng)

    def run(self, targets, effort, rng):
        out = {}
        for t in targets:
            bound = t.bound(self.solver.losses)
            total = self.solver.value(t) / bound           # only used to set the scale
            eps_total = max(effort * total, 1e-12)
            calm_part = self.calm.weight * self.calm.conditional(t) / bound
            shocked = [s for s in self.solver.strata(t) if not s.label.startswith("0 ")]
            if not shocked:                        # no shared shocks: the answer is exact
                out[t.name] = Estimate(calm_part * bound, 0.0, 0, self.name)
                continue

            # pilot: coarse run per stratum, its error counts against the budget
            queries, pilots, deepest = 0, [], 0
            for s in shocked:
                pilot_eps = max(self.pilot_share * eps_total / s.weight, 1e-6)
                r = self._iqae(t, s, pilot_eps, rng) if pilot_eps < 0.5 else None
                queries += r.queries if r else 0
                deepest = max(deepest, r.max_power if r else 0)
                pilots.append(r)

            # Neyman-style split: stratum z gets eps_z ~ sqrt(sd_z / w_z)
            sds = []
            for s, r in zip(shocked, pilots):
                a_hi = r.interval[1] if r else 1.0
                sds.append(max(np.sqrt(a_hi * (1 - min(a_hi, 0.999))), 1e-6))
            lam = eps_total / sum(np.sqrt(s.weight * sd) for s, sd in zip(shocked, sds))

            value, half = calm_part, 0.0
            for s, r, sd in zip(shocked, pilots, sds):
                eps_z = lam * np.sqrt(sd / s.weight)
                if r is None or eps_z < (r.interval[1] - r.interval[0]) / 2:
                    r = self._iqae(t, s, eps_z, rng)
                    queries += r.queries
                    deepest = max(deepest, r.max_power)
                value += s.weight * r.estimate
                half += s.weight * (r.interval[1] - r.interval[0]) / 2
            out[t.name] = Estimate(value * bound, half * bound, queries, self.name, deepest)
        return out
