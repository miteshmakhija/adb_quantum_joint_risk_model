"""Countries, shared shocks and the portfolio that ties them together."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .perils import QUAKE, RAIN, EarthquakeCover, RainCover


@dataclass(frozen=True)
class Country:
    name: str
    quake: EarthquakeCover
    rain: RainCover

    @property
    def principal(self) -> float:
        return self.quake.limit + self.rain.limit


@dataclass(frozen=True)
class SharedShock:
    """A source of events that hit every member country at the same time.

    For earthquakes the payout size is still drawn per country (as in the ADB model);
    only the timing is shared. For rain a shared event triggers all members.
    """
    peril: str
    members: tuple[int, ...]
    rate: float                     # events per year

    def __post_init__(self):
        if self.peril not in (QUAKE, RAIN):
            raise ValueError(f"unknown peril {self.peril!r}")
        if len(self.members) < 2:
            raise ValueError("a shared shock needs at least two member countries")


@dataclass
class Portfolio:
    countries: list[Country]
    shocks: list[SharedShock] = field(default_factory=list)
    years: int = 3

    def __post_init__(self):
        for i in range(self.size):
            for peril in (QUAKE, RAIN):
                if self.own_rate(i, peril) < -1e-15:
                    raise ValueError(f"shared {peril} rates exceed the total rate of "
                                     f"{self.countries[i].name}")

    @property
    def size(self) -> int:
        return len(self.countries)

    @property
    def total_principal(self) -> float:
        return sum(c.principal for c in self.countries)

    def shocks_of(self, country: int, peril: str) -> list[int]:
        return [s for s, sh in enumerate(self.shocks)
                if sh.peril == peril and country in sh.members]

    def own_rate(self, country: int, peril: str) -> float:
        """Rate of events that hit this country alone. Total rate is kept fixed."""
        cover = self.countries[country].quake if peril == QUAKE else self.countries[country].rain
        shared = sum(self.shocks[s].rate for s in self.shocks_of(country, peril))
        return cover.total_rate - shared

    def incidence(self, peril: str) -> np.ndarray:
        """0/1 matrix (shocks of this peril x countries)."""
        rows = [s for s in self.shocks if s.peril == peril]
        mat = np.zeros((len(rows), self.size), dtype=int)
        for r, sh in enumerate(rows):
            mat[r, list(sh.members)] = 1
        return mat
