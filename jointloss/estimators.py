"""Common interface for every estimator (classical sampling and quantum)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .targets import Target


@dataclass
class Estimate:
    value: float
    half_width: float        # 95% interval half-width reported by the method
    queries: int             # paths for samplers, applications of A or A^dagger for QAE
    method: str
    max_power: int = 0       # deepest Grover power used (QAE only)


class Estimator(ABC):
    """`effort` is a path budget for samplers and a target accuracy for QAE."""
    name: str = "estimator"
    effort_kind: str = "budget"

    @abstractmethod
    def run(self, targets: list[Target], effort: float,
            rng: np.random.Generator) -> dict[str, Estimate]: ...
