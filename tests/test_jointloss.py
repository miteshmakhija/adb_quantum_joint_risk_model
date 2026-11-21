import numpy as np
import pytest

from jointloss.amplitude import IdealOracle, IterativeAmplitudeEstimation
from jointloss.exact import (CalmStratumSolver, LatentEnumerationSolver, PathConvolutionSolver,
                             ShockGraphContraction)
from jointloss.scenarios import SCENARIOS, adb_portfolio, chain_portfolio
from jointloss.targets import focus_targets

# Table 5 of the challenge text (rounded as published)
TABLE5 = {
    "adb-base": (10.06e6, 0.04414, 0.01205, 282_355, 0.000032),
    "adb-independent": (10.06e6, 0.02716, 0.00719, 171_266, 0.000004),
    "adb-common-shock": (10.06e6, 0.05053, 0.01607, 385_043, 0.000044),
    "adb-rain-stress": (11.17e6, 0.07046, 0.01322, 306_286, 0.000069),
    "adb-rain-cover": (9.97e6, 0.04414, 0.00700, 170_927, 0.000032),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_table5(name):
    sol = ShockGraphContraction(adb_portfolio(name))
    for target, want in zip(focus_targets(), TABLE5[name]):
        got = sol.value(target)
        assert abs(got - want) <= 0.006 * abs(want) + 6e-7, (target.name, got, want)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_three_exact_methods_agree(name):
    pf = adb_portfolio(name)
    a, b, c = PathConvolutionSolver(pf), LatentEnumerationSolver(pf), ShockGraphContraction(pf)
    for t in focus_targets():
        assert np.isclose(a.value(t), b.value(t), rtol=1e-10, atol=1e-14)
        assert np.isclose(a.value(t), c.value(t), rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("n", [3, 4])
def test_contraction_matches_brute_force_on_chains(n):
    pf = chain_portfolio(n)
    a, b = LatentEnumerationSolver(pf), ShockGraphContraction(pf)
    for t in focus_targets():
        assert np.isclose(a.value(t), b.value(t), rtol=1e-10, atol=1e-14)


def test_strata_add_up_and_calm_stratum_matches():
    pf = adb_portfolio()
    sol, calm = ShockGraphContraction(pf), CalmStratumSolver(pf)
    for t in focus_targets():
        strata = sol.strata(t)
        assert np.isclose(sum(s.weight for s in strata), 1.0)
        total = sum(s.weight * s.conditional for s in strata) * t.bound(sol.losses)
        assert np.isclose(total, sol.value(t), rtol=1e-10)
        assert np.isclose(strata[0].weight, calm.weight)
        assert np.isclose(strata[0].conditional * t.bound(sol.losses), calm.conditional(t))


def test_iqae_interval_covers_truth():
    rng = np.random.default_rng(0)
    for a in (0.1, 7e-4, 3e-5):
        hits = sum(lo <= a <= hi for lo, hi in
                   (IterativeAmplitudeEstimation(a / 5).estimate(IdealOracle(a), rng).interval
                    for _ in range(40)))
        assert hits >= 36


def test_circuit_amplitude_matches_exact():
    pytest.importorskip("qiskit")
    from jointloss.circuits import ShockLoaderCircuit
    pf = adb_portfolio()
    sol = ShockGraphContraction(pf)
    t = focus_targets()[4]
    assert np.isclose(ShockLoaderCircuit(pf, t).amplitude(), sol.value(t), rtol=1e-8)
