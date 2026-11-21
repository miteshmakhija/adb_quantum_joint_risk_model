"""First-order resource model for the scalable (arithmetic) version of the stratum circuit.

Counts are logical (error-corrected) qubits and gates. They are meant for planning and
for comparing designs, and will be refined with a compiler-level estimator in Phase 2.

Building blocks
  shock registers       3 qubits per quake shock, 1 per rain shock
  country loader        4 qubits; a multiplexed 4-qubit state preparation controlled by a
                        4-qubit (shared-count, rain-hit) register: 240 CNOT + 240 rotations
  shared-count adder    sum of the country's shock counts, capped at 4 (~10 Toffoli per shock)
  loss accumulation     4 controlled constant additions into a b-bit register (~2b Toffoli each)
  counters              attached / exhausted flags and 2-bit saturating counters (~16 Toffoli)
  objective             comparator (~2b Toffoli) or b controlled rotations for linear payoffs
  Grover iterate Q      2 x A plus two reflections (~2 Toffoli per qubit each)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log2

from .losses import LossModel
from .perils import QUAKE, RAIN
from .portfolio import Portfolio


@dataclass
class ResourceReport:
    countries: int
    shocks: int
    logical_qubits: int
    toffoli_per_A: int
    cnot_per_A: int
    rotations_per_A: int
    toffoli_per_Q: int
    max_grover_power: int
    toffoli_total_depth_path: int      # Toffolis on the deepest circuit, (2k+1) A's
    statevector_gb_fp32: float         # memory to simulate one stratum circuit exactly

    def as_row(self) -> dict:
        return asdict(self)


class OracleResourceModel:
    COUNTRY_CNOT = 240
    COUNTRY_ROT = 240

    def __init__(self, portfolio: Portfolio, count_cap: int = 2):
        self.portfolio = portfolio
        self.losses = LossModel(portfolio)
        self.count_cap = count_cap

    def report(self, max_grover_power: int, linear_payoff: bool = False) -> ResourceReport:
        pf = self.portfolio
        n = pf.size
        q_shocks = sum(1 for s in pf.shocks if s.peril == QUAKE)
        r_shocks = sum(1 for s in pf.shocks if s.peril == RAIN)
        b = ceil(log2(self.losses.grid_size()))
        cbits = ceil(log2(self.count_cap + 1))

        qubits = (3 * q_shocks + r_shocks          # shocks
                  + 4 * n                          # countries
                  + 4                              # shared-count / rain-hit scratch
                  + b + 1                          # loss register + carry
                  + 2 * cbits + 2                  # counters + flags
                  + 1 + 1)                         # comparator ancilla + objective

        degree_q = sum(len(s.members) for s in pf.shocks if s.peril == QUAKE)
        degree_r = sum(len(s.members) for s in pf.shocks if s.peril == RAIN)
        toffoli = (2 * (10 * degree_q + degree_r)  # shared counts, compute + uncompute
                   + n * 4 * 2 * b                 # loss accumulation
                   + n * 16                        # flags and counters
                   + (0 if linear_payoff else 2 * b))
        stratum_prep_rot = 10 * len(pf.shocks)     # restricted shock superposition
        rotations = n * self.COUNTRY_ROT + stratum_prep_rot + (b if linear_payoff else 1)
        cnot = n * self.COUNTRY_CNOT + 4 * toffoli

        toffoli_q = 2 * toffoli + 2 * 2 * qubits
        deepest = (2 * max_grover_power + 1) * toffoli + max_grover_power * 4 * qubits
        return ResourceReport(n, len(pf.shocks), qubits, toffoli, cnot, rotations, toffoli_q,
                              max_grover_power, deepest, round(8 * 2.0 ** qubits / 1e9, 3))
