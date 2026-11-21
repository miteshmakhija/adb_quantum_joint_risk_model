"""The five ADB benchmark scenarios and a larger synthetic portfolio for scaling tests."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .perils import QUAKE, RAIN, EarthquakeCover, RainCover
from .portfolio import Country, Portfolio, SharedShock

ADB_QUAKE_CURVES = {
    "Kyrgyz Republic": (0.03883, 0.02331, 0.01790, 0.01184),
    "Tajikistan": (0.03925, 0.01743, 0.01332, 0.01043),
}
# P(the same quake triggers the other bond | it triggers this one)
TRIGGER_GIVEN_OTHER = (0.075, 0.074)


@dataclass(frozen=True)
class Scenario:
    name: str
    quake_shared_multiplier: float = 1.0     # m in eq. (4)
    rain_shared_share: float = 0.25          # shared rain rate as share of the smaller rate
    rain_rate_multiplier: float = 1.0
    quake_limit: float = 70e6
    rain_limit: float = 10e6
    rain_annual_prob: float = 0.02
    threshold: float = 80e6                  # T in the focus targets


BASE = Scenario("adb-base")
SCENARIOS = {s.name: s for s in [
    BASE,
    replace(BASE, name="adb-independent", quake_shared_multiplier=0.0, rain_shared_share=0.0),
    replace(BASE, name="adb-common-shock", quake_shared_multiplier=2.0),
    replace(BASE, name="adb-rain-stress", rain_rate_multiplier=2.0),
    replace(BASE, name="adb-rain-cover", quake_limit=60e6, rain_limit=20e6),
]}


def _country(name: str, curve, sc: Scenario) -> Country:
    return Country(name,
                   EarthquakeCover(sc.quake_limit, curve),
                   RainCover(sc.rain_limit, sc.rain_annual_prob, sc.rain_rate_multiplier))


def _pair_shocks(c1: Country, c2: Country, i: int, j: int, sc: Scenario,
                 cond=TRIGGER_GIVEN_OTHER) -> list[SharedShock]:
    shocks = []
    quake_rate = sc.quake_shared_multiplier * 0.5 * (cond[0] * c1.quake.total_rate
                                                     + cond[1] * c2.quake.total_rate)
    if quake_rate > 0:
        shocks.append(SharedShock(QUAKE, (i, j), quake_rate))
    rain_rate = sc.rain_shared_share * min(c1.rain.total_rate, c2.rain.total_rate)
    if rain_rate > 0:
        shocks.append(SharedShock(RAIN, (i, j), rain_rate))
    return shocks


def adb_portfolio(scenario: Scenario | str = BASE) -> Portfolio:
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    names = list(ADB_QUAKE_CURVES)
    countries = [_country(n, ADB_QUAKE_CURVES[n], sc) for n in names]
    return Portfolio(countries, _pair_shocks(countries[0], countries[1], 0, 1, sc))


def chain_portfolio(n: int, scenario: Scenario | str = BASE,
                    regional_quake_rate: float = 0.0) -> Portfolio:
    """n countries along a line of shared fault and glacier systems.

    Neighbours share shocks calibrated like the ADB pair. Countries alternate between the
    two ADB risk profiles. An optional regional quake shock hits every country at once.
    """
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    names = list(ADB_QUAKE_CURVES)
    countries = [_country(f"C{i + 1}", ADB_QUAKE_CURVES[names[i % 2]], sc) for i in range(n)]
    shocks: list[SharedShock] = []
    for i in range(n - 1):
        shocks += _pair_shocks(countries[i], countries[i + 1], i, i + 1, sc)
    if regional_quake_rate > 0 and n > 2:
        shocks.append(SharedShock(QUAKE, tuple(range(n)), regional_quake_rate))
    return Portfolio(countries, shocks)


def band_portfolio(n: int, reach: int, scenario: Scenario | str = BASE) -> Portfolio:
    """Every pair of countries closer than `reach` shares a quake shock.

    Used to stress the classical exact solver: its cost grows with how many shocks are
    open at once (the graph width), while the quantum loader stays linear in shocks.
    Pair rates are divided by `reach` so each country keeps a similar shared share.
    """
    sc = SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    base = chain_portfolio(n, sc)
    shocks: list[SharedShock] = [s for s in base.shocks if s.peril == RAIN]
    for i in range(n):
        for j in range(i + 1, min(n, i + reach + 1)):
            ci, cj = base.countries[i], base.countries[j]
            rate = sc.quake_shared_multiplier * 0.5 * (TRIGGER_GIVEN_OTHER[0] * ci.quake.total_rate
                                                       + TRIGGER_GIVEN_OTHER[1] * cj.quake.total_rate)
            shocks.append(SharedShock(QUAKE, (i, j), rate / reach))
    return Portfolio(base.countries, shocks)
