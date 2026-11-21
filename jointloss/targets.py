"""The five focus targets. Each one is a function of a term Summary.

For n > 2 countries the joint targets read "at least k countries", which is the
challenge definition when n = 2 and k = 2.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .losses import LossModel, Summary


class Target(ABC):
    name: str = "target"
    is_probability: bool = True
    count_needed: int = 0          # how far the attach/exhaust counters must go

    @abstractmethod
    def payoff(self, s: Summary) -> np.ndarray: ...

    def bound(self, losses: LossModel) -> float:
        """Largest possible payoff. QAE needs payoffs scaled into [0, 1]."""
        return 1.0

    def __repr__(self):
        return self.name


class ExpectedLoss(Target):
    name, is_probability = "expected_principal_loss_usd", False

    def payoff(self, s):
        return s.total * 1.0

    def bound(self, losses):
        return losses.portfolio.total_principal


class JointAttachment(Target):
    def __init__(self, k: int = 2):
        self.k = self.count_needed = k
        self.name = "joint_attachment_probability" if k == 2 else f"attach_at_least_{k}"

    def payoff(self, s):
        return (s.attached >= self.k) * 1.0


class LossExceedance(Target):
    def __init__(self, threshold: float = 80e6):
        self.threshold = threshold
        self.name = "portfolio_loss_exceedance_probability"

    def payoff(self, s):
        return (s.total > self.threshold + 1e-6) * 1.0


class StopLoss(Target):
    is_probability = False

    def __init__(self, threshold: float = 80e6):
        self.threshold = threshold
        self.name = "portfolio_stop_loss_usd"

    def payoff(self, s):
        return np.maximum(s.total - self.threshold, 0.0)

    def bound(self, losses):
        return max(losses.portfolio.total_principal - self.threshold, 1.0)


class JointExhaustion(Target):
    def __init__(self, k: int = 2):
        self.k = self.count_needed = k
        self.name = "both_bonds_exhausted_probability" if k == 2 else f"exhaust_at_least_{k}"

    def payoff(self, s):
        return (s.exhausted >= self.k) * 1.0


def focus_targets(threshold: float = 80e6) -> list[Target]:
    return [ExpectedLoss(), JointAttachment(2), LossExceedance(threshold),
            StopLoss(threshold), JointExhaustion(2)]
