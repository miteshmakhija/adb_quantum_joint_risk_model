"""Command line entry point.

    python run.py reference                 exact values, three independent methods
    python run.py benchmark [--quick]       error vs queries for all estimators
    python run.py scaling                   larger portfolios, classical limits, resources
    python run.py circuit                   check the Qiskit circuit against the exact values
    python run.py summary                   tables and plots again from results/benchmark.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from jointloss.exact import (CalmStratumSolver, LatentEnumerationSolver, PathConvolutionSolver,
                             ShockGraphContraction)
from jointloss.benchmark import Benchmark
from jointloss.quantum_estimators import ShockSplitQAE
from jointloss.resources import OracleResourceModel
from jointloss.sampling import CalmPeeledMonteCarlo, LatentMonteCarlo
from jointloss.scenarios import SCENARIOS, adb_portfolio, band_portfolio, chain_portfolio
from jointloss.targets import JointExhaustion, focus_targets

OUT = Path("results")


def _fmt(target, v):
    return f"{100 * v:.4f}%" if target.is_probability else f"{v:,.0f}"


def cmd_reference(_):
    targets = focus_targets()
    print(f"{'scenario':18s}" + "".join(f"{t.name[:22]:>25s}" for t in targets))
    for name in SCENARIOS:
        pf = adb_portfolio(name)
        solvers = [PathConvolutionSolver(pf), LatentEnumerationSolver(pf), ShockGraphContraction(pf)]
        vals = [[s.value(t) for s in solvers] for t in targets]
        gap = max(abs(v[0] - v[2]) / max(abs(v[0]), 1e-30) for v in vals)
        print(f"{name:18s}" + "".join(f"{_fmt(t, v[2]):>25s}" for t, v in zip(targets, vals))
              + f"   max gap between methods {gap:.1e}")


def cmd_benchmark(args):
    names = ["adb-base"] if args.quick else list(SCENARIOS)
    repeats = 8 if args.quick else args.repeats
    budgets = [2 ** 10, 2 ** 13, 2 ** 16, 2 ** 19] if args.quick else None
    efforts = [0.3, 0.1, 0.03, 0.01] if args.quick else None
    all_records = []
    for name in names:
        bench = Benchmark(adb_portfolio(name), name, repeats=repeats, seed=args.seed)
        recs = bench.run(budgets, efforts)
        all_records += recs
        Benchmark.plot(recs, OUT / f"error_vs_queries_{name}.png", name)
        Benchmark.plot_focus(recs, OUT / f"both_exhausted_{name}.png", name)
        for wanted in (0.10, 0.01):
            print("\n" + Benchmark.summary_table(recs, name, wanted))
    Benchmark.save_csv(all_records, OUT / "benchmark.csv")
    print(f"\nSaved {OUT / 'benchmark.csv'} and plots in {OUT}/")


def cmd_summary(_):
    recs = Benchmark.load_csv(OUT / "benchmark.csv")
    for name in dict.fromkeys(r.scenario for r in recs):
        sub = [r for r in recs if r.scenario == name]
        Benchmark.plot(sub, OUT / f"error_vs_queries_{name}.png", name)
        Benchmark.plot_focus(sub, OUT / f"both_exhausted_{name}.png", name)
        for wanted in (0.10, 0.01):
            print("\n" + Benchmark.summary_table(recs, name, wanted))


def cmd_scaling(args):
    rng = np.random.default_rng(args.seed)
    target = JointExhaustion(2)
    print("1) Chain portfolios (neighbours share shocks, plus one regional quake shock)")
    print(f"{'n':>4} {'width':>6} {'exact s':>8} {'P(>=2 exhausted)':>17} "
          f"{'peeled-MC relerr@1e5':>21} {'ShockSplitQAE queries (eps 10%)':>32} {'logical qubits':>15}")
    for n in args.sizes:
        pf = chain_portfolio(n, regional_quake_rate=0.001)
        t0 = time.time()
        sol, calm = ShockGraphContraction(pf), CalmStratumSolver(pf)
        exact = sol.value(target)
        dt = time.time() - t0
        mc = CalmPeeledMonteCarlo(pf, calm)
        errs = [mc.run([target], 1e5, rng)[target.name].value - exact for _ in range(5)]
        qae = ShockSplitQAE(sol, calm)
        runs = [qae.run([target], 0.1, rng)[target.name] for _ in range(5)]
        rep = OracleResourceModel(pf).report(max(r.max_power for r in runs))
        print(f"{n:>4} {sol.width:>6} {dt:>8.2f} {exact:>17.3e} "
              f"{np.sqrt(np.mean(np.square(errs))) / exact:>21.3f} "
              f"{np.mean([r.queries for r in runs]):>32,.0f} {rep.logical_qubits:>15}")

    print("\n2) Denser shock graphs (n = 12): classical exact cost vs graph width")
    print(f"{'reach':>6} {'shocks':>7} {'width':>6} {'memory GB':>10} {'exact s':>9} {'logical qubits':>15}")
    for reach in args.reaches:
        pf = band_portfolio(12, reach)
        sol = ShockGraphContraction(pf)
        mem = sol.peak_memory_gb()
        t0 = time.time()
        try:
            sol.value(target)
            took = f"{time.time() - t0:9.2f}"
        except MemoryError:
            took = f"{'too big':>9s}"
        print(f"{reach:>6} {len(pf.shocks):>7} {sol.width:>6} {mem:>10.3g} {took} "
              f"{OracleResourceModel(pf).report(1).logical_qubits:>15}")

    print("\n3) Resource estimate for the arithmetic stratum circuit (chain portfolios)")
    rows = []
    for n in [2, 5, 10, 20, 50]:
        pf = chain_portfolio(n, regional_quake_rate=0.001)
        rows.append(OracleResourceModel(pf).report(max_grover_power=200).as_row())
    keys = list(rows[0])
    print("  " + " | ".join(keys))
    for r in rows:
        print("  " + " | ".join(f"{r[k]:,}" if isinstance(r[k], int) else f"{r[k]:.3g}" for k in keys))


def cmd_circuit(_):
    from jointloss.circuits import CircuitOracle, ShockLoaderCircuit
    pf = adb_portfolio("adb-base")
    sol = ShockGraphContraction(pf)
    print("Full loader A (ADB base case):")
    for t in focus_targets():
        loader = ShockLoaderCircuit(pf, t)
        a = loader.amplitude() * t.bound(sol.losses)
        print(f"  {t.name:40s} circuit {a:.10g}   exact {sol.value(t):.10g}")
    t = JointExhaustion(2)
    print("\nStratum loaders for 'both exhausted', after Q^k A (circuit vs ideal):")
    for s in sol.strata(t)[1:]:
        bucket = int(s.label.split()[0].rstrip("+"))
        oracle = CircuitOracle(ShockLoaderCircuit(pf, t, bucket))
        theta = np.arcsin(np.sqrt(s.conditional))
        for k in (0, 1, 2):
            print(f"  {s.label:10s} k={k}  circuit {oracle.good_probability(k):.8f}   "
                  f"ideal {np.sin((2 * k + 1) * theta) ** 2:.8f}")
    print("\nGate counts of the stratum circuit (lookup objective, basis u/cx):")
    print(" ", oracle.gate_counts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("reference")
    b = sub.add_parser("benchmark")
    b.add_argument("--quick", action="store_true")
    b.add_argument("--repeats", type=int, default=20)
    b.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("scaling")
    s.add_argument("--sizes", type=int, nargs="+", default=[2, 5, 10, 20, 40])
    s.add_argument("--reaches", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    s.add_argument("--seed", type=int, default=7)
    sub.add_parser("circuit")
    sub.add_parser("summary")
    args = p.parse_args()
    {"reference": cmd_reference, "benchmark": cmd_benchmark,
     "scaling": cmd_scaling, "circuit": cmd_circuit, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
