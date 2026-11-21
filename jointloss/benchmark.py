"""Benchmark runner: error versus number of queries, for every estimator and target."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimators import Estimator
from .exact import CalmStratumSolver, ShockGraphContraction
from .portfolio import Portfolio
from .quantum_estimators import PlainQAE, ShockSplitQAE
from .sampling import (AnnualPathSpace, CalmPeeledMonteCarlo, FirstPayoutStratified,
                       LatentMonteCarlo, LossTiltedImportance, MonteCarlo, ScrambledSobol)
from .targets import Target, focus_targets


@dataclass
class Record:
    scenario: str
    method: str
    target: str
    effort: float
    mean_queries: float
    exact: float
    rel_rmse: float
    coverage: float          # share of runs whose 95% interval holds the exact value
    max_power: int


class Benchmark:
    BUDGETS = [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]
    QAE_EFFORTS = [3.0, 1.5, 0.8, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003]

    def __init__(self, portfolio: Portfolio, scenario: str, targets: list[Target] | None = None,
                 repeats: int = 20, seed: int = 7, include_path_baselines: bool = True):
        self.portfolio, self.scenario = portfolio, scenario
        self.targets = targets or focus_targets()
        self.repeats, self.rng = repeats, np.random.default_rng(seed)
        self.solver = ShockGraphContraction(portfolio)
        self.calm = CalmStratumSolver(portfolio)
        self.exact = {t.name: self.solver.value(t) for t in self.targets}
        self.estimators = self._default_estimators(include_path_baselines)

    def _default_estimators(self, with_paths: bool) -> list[Estimator]:
        est: list[Estimator] = []
        if with_paths:
            space = AnnualPathSpace(self.portfolio)
            est += [MonteCarlo(self.portfolio, space), ScrambledSobol(self.portfolio, space),
                    FirstPayoutStratified(self.portfolio, space),
                    LossTiltedImportance(self.portfolio, space)]
        est += [LatentMonteCarlo(self.portfolio), CalmPeeledMonteCarlo(self.portfolio, self.calm),
                PlainQAE(self.solver), ShockSplitQAE(self.solver, self.calm)]
        return est

    def run(self, budgets=None, efforts=None, progress=print) -> list[Record]:
        records = []
        for est in self.estimators:
            grid = (efforts or self.QAE_EFFORTS) if est.effort_kind == "epsilon" \
                else (budgets or self.BUDGETS)
            for effort in grid:
                runs = [est.run(self.targets, effort, self.rng) for _ in range(self.repeats)]
                for t in self.targets:
                    ex = self.exact[t.name]
                    vals = np.array([r[t.name].value for r in runs])
                    half = np.array([r[t.name].half_width for r in runs])
                    records.append(Record(
                        self.scenario, est.name, t.name, effort,
                        float(np.mean([r[t.name].queries for r in runs])), ex,
                        float(np.sqrt(np.mean((vals - ex) ** 2)) / abs(ex)),
                        float(np.mean(np.abs(vals - ex) <= half + 1e-15)),
                        int(max(r[t.name].max_power for r in runs))))
                if progress:
                    progress(f"  {self.scenario:17s} {est.name:24s} effort={effort:g}")
        return records

    # reporting ---------------------------------------------------------------------------
    @staticmethod
    def queries_for(records: list[Record], method: str, target: str,
                    wanted: float = 0.1) -> float | None:
        """Queries needed for the wanted relative RMSE (log-log interpolation).

        Returns 0 when the method is exact for this target, and None when even the
        largest effort tried was not enough.
        """
        pts = sorted((r.mean_queries, max(r.rel_rmse, 1e-15)) for r in records
                     if r.method == method and r.target == target)
        if not pts:
            return None
        if all(e <= 1e-12 for _, e in pts):
            return 0.0
        pts = [(q, e) for q, e in pts if q > 0]
        if not pts:
            return None
        if pts[0][1] <= wanted:
            return pts[0][0]            # already good enough at the smallest effort
        for (q0, e0), (q1, e1) in zip(pts, pts[1:]):
            if (e0 - wanted) * (e1 - wanted) <= 0 and e0 != e1:
                f = (np.log(wanted) - np.log(e0)) / (np.log(e1) - np.log(e0))
                return float(np.exp(np.log(q0) + f * (np.log(q1) - np.log(q0))))
        return None

    @staticmethod
    def load_csv(path: Path) -> list[Record]:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        conv = {"effort": float, "mean_queries": float, "exact": float, "rel_rmse": float,
                "coverage": float, "max_power": int}
        return [Record(**{k: conv.get(k, str)(v) for k, v in r.items()}) for r in rows]

    @classmethod
    def summary_table(cls, records: list[Record], scenario: str, wanted: float) -> str:
        recs = [r for r in records if r.scenario == scenario]
        targets = list(dict.fromkeys(r.target for r in recs))
        methods = list(dict.fromkeys(r.method for r in recs))
        lines = [f"Queries needed for {wanted:.0%} relative RMSE ({scenario})",
                 f"{'method':26s}" + "".join(f"{t[:18]:>20s}" for t in targets)]
        for m in methods:
            cells = []
            for t in targets:
                q = cls.queries_for(recs, m, t, wanted)
                text = "exact (0)" if q == 0 else ("> max tried" if q is None else f"{q:,.0f}")
                cells.append(f"{text:>20s}")
            lines.append(f"{m:26s}" + "".join(cells))
        return "\n".join(lines)

    @staticmethod
    def save_csv(records: list[Record], path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(Record.__dataclass_fields__))
            w.writeheader()
            for r in records:
                w.writerow(r.__dict__)

    STYLE = {"Shock-split QAE (ours)": dict(color="black", lw=2.6, marker="o", ms=5, ls="-"),
             "Plain QAE": dict(color="#c2185b", lw=1.6, marker="o", ms=4, ls="-"),
             "Calm-peeled MC": dict(color="#1565c0", lw=1.4, marker=".", ls="--"),
             "Importance sampling": dict(color="#ef6c00", lw=1.2, marker=".", ls="--"),
             "Stratified": dict(color="#2e7d32", lw=1.0, marker=".", ls="--"),
             "Scrambled Sobol": dict(color="#6a1b9a", lw=1.0, marker=".", ls="--"),
             "Monte Carlo": dict(color="#757575", lw=1.0, marker=".", ls="--"),
             "Latent Monte Carlo": dict(color="#9e9e9e", lw=1.0, marker=".", ls=":")}

    @classmethod
    def _draw(cls, ax, records, target):
        for m in dict.fromkeys(r.method for r in records):
            pts = sorted((r.mean_queries, r.rel_rmse) for r in records
                         if r.method == m and r.target == target
                         and r.rel_rmse > 1e-9 and r.mean_queries > 0)
            if pts:
                q, e = zip(*pts)
                ax.loglog(q, e, label=m, **cls.STYLE.get(m, {}))
        ax.axhline(0.1, color="grey", lw=0.6, ls=":")
        ax.axhline(0.01, color="grey", lw=0.6, ls=":")
        ax.set_ylim(1e-4, 3)
        ax.set_xlabel("queries (paths, or calls to A)")
        ax.grid(alpha=0.3, which="both")

    @classmethod
    def plot(cls, records: list[Record], path: Path, scenario: str):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        targets = list(dict.fromkeys(r.target for r in records))
        fig, axes = plt.subplots(1, len(targets), figsize=(4.3 * len(targets), 4.0))
        for ax, t in zip(np.atleast_1d(axes), targets):
            cls._draw(ax, records, t)
            ax.set_title(t.replace("_", " "), fontsize=9)
        np.atleast_1d(axes)[0].set_ylabel("relative RMSE")
        np.atleast_1d(axes)[-1].legend(fontsize=7, loc="lower left")
        fig.suptitle(f"Error vs queries, {scenario} (20 repeats per point)", fontsize=11)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)

    @classmethod
    def plot_focus(cls, records: list[Record], path: Path, scenario: str,
                   target: str = "both_bonds_exhausted_probability"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        cls._draw(ax, records, target)
        ax.set_ylabel("relative RMSE")
        ax.set_title(f"Both bonds exhausted (p = 3.2e-5), {scenario}")
        ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
