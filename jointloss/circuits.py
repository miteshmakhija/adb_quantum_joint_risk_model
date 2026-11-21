"""Qiskit circuits for the shock-conditioned loader (small portfolios).

Layout of A:
  shock registers   quake shock: 3 qubits (count 0..4), rain shock: 1 qubit (fired)
  country registers 4 qubits each: bits 0-2 = quake steps, bit 3 = rain flag
  objective         1 qubit, rotated so that P(1) = payoff / bound

Shock registers are independent, so they are loaded one by one (shallow). Each country is
then prepared with gates controlled only by the shocks it belongs to, so the loader grows
linearly with the number of countries. Restricting the shock registers to one stratum
gives the stratum loader used by ShockSplitQAE.

The objective here is a lookup rotation over all country qubits. That is fine for n <= 3
and for checking correctness; for large n it is replaced by adders and a comparator
(see resources.py for the cost of that version).
"""
from __future__ import annotations

import itertools

import numpy as np

from .amplitude import AmplitudeOracle
from .exact import StratumAmplitude
from .latent import ShockLatentModel
from .losses import LossModel
from .perils import QUAKE
from .portfolio import Portfolio
from .targets import Target

try:
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import StatePreparation, UCRYGate
    from qiskit.quantum_info import Statevector
    try:
        from qiskit.circuit.library import grover_operator as _grover
    except ImportError:                               # older Qiskit
        from qiskit.circuit.library import GroverOperator as _grover
    HAVE_QISKIT = True
except ImportError:                                   # pragma: no cover
    HAVE_QISKIT = False


def _bucket(values) -> int:
    return sum(int(v > 0) for v in values)


class ShockLoaderCircuit:
    def __init__(self, portfolio: Portfolio, target: Target,
                 stratum_bucket: int | None = None, max_bucket: int = 2):
        if not HAVE_QISKIT:
            raise ImportError("pip install qiskit to build circuits")
        self.portfolio, self.target = portfolio, target
        self.model = ShockLatentModel(portfolio)
        self.losses = LossModel(portfolio)
        self.bucket, self.max_bucket = stratum_bucket, max_bucket
        self.circuit = self._build()

    # helpers ----------------------------------------------------------------------------
    def _shock_width(self, s: int) -> int:
        return 3 if self.portfolio.shocks[s].peril == QUAKE else 1

    def _in_stratum(self, values) -> bool:
        if self.bucket is None:
            return True
        return min(_bucket(values), self.max_bucket) == self.bucket

    def _build(self) -> "QuantumCircuit":
        pf = self.portfolio
        shock_regs = [QuantumRegister(self._shock_width(s), f"s{s}") for s in range(len(pf.shocks))]
        country_regs = [QuantumRegister(4, f"c{i}") for i in range(pf.size)]
        obj = QuantumRegister(1, "obj")
        qc = QuantumCircuit(*shock_regs, *country_regs, obj)

        # 1. shocks
        priors = [self.model.shock_prior(s) for s in range(len(pf.shocks))]
        if self.bucket is None:
            for reg, prior in zip(shock_regs, priors):
                amp = np.zeros(2 ** len(reg))
                amp[:len(prior)] = np.sqrt(prior)
                qc.append(StatePreparation(amp), reg)
        elif shock_regs:
            widths = [len(r) for r in shock_regs]
            amp = np.zeros(2 ** sum(widths))
            for values in itertools.product(*[range(len(p)) for p in priors]):
                if self._in_stratum(values):
                    idx, shift = 0, 0
                    for v, w in zip(values, widths):
                        idx += v << shift
                        shift += w
                    amp[idx] = np.prod([p[v] for p, v in zip(priors, values)])
            amp = np.sqrt(amp / amp.sum())
            qc.append(StatePreparation(amp), [q for r in shock_regs for q in r])

        # 2. countries, controlled by their own shocks only
        for i, reg in enumerate(country_regs):
            mine = [s for s, sh in enumerate(pf.shocks) if i in sh.members]
            for local in itertools.product(*[range(len(priors[s])) for s in mine]):
                full = [0] * len(pf.shocks)
                for s, v in zip(mine, local):
                    full[s] = v
                pmf = self.model.country_pmf(i, *self.model.country_inputs(i, full))
                amp = np.zeros(16)
                for a in range(5):
                    for r in (0, 1):
                        amp[a + 8 * r] = np.sqrt(pmf[a, r])
                prep = StatePreparation(amp)
                if not mine:
                    qc.append(prep, reg)
                    continue
                ctrl_qubits = [q for s in mine for q in shock_regs[s]]
                ctrl_state, shift = 0, 0
                for s, v in zip(mine, local):
                    ctrl_state += v << shift
                    shift += self._shock_width(s)
                qc.append(prep.control(len(ctrl_qubits), ctrl_state=ctrl_state),
                          ctrl_qubits + list(reg))

        # 3. objective rotation
        n = pf.size
        codes = np.arange(16 ** n)
        digits = (codes[:, None] // 16 ** np.arange(n)) % 16
        steps, rain = digits % 8, digits // 8
        valid = (steps <= 4).all(1)
        f = np.zeros(len(codes))
        s = self.losses.summarize(np.minimum(steps, 4), rain)
        f[valid] = (self.target.payoff(s) / self.target.bound(self.losses))[valid]
        angles = list(2 * np.arcsin(np.sqrt(np.clip(f, 0, 1))))
        qc.append(UCRYGate(angles), [obj[0]] + [q for r in country_regs for q in r])
        return qc

    @property
    def objective_index(self) -> int:
        return self.circuit.num_qubits - 1

    def amplitude(self) -> float:
        sv = Statevector.from_instruction(self.circuit)
        return float(sv.probabilities([self.objective_index])[1])

    def grover(self):
        oracle = QuantumCircuit(self.circuit.num_qubits)
        oracle.z(self.objective_index)
        return _grover(oracle, state_preparation=self.circuit)


class CircuitOracle(AmplitudeOracle):
    """Exact statevector simulation of Q^k A.

    Q is compiled to basic gates once. IQAE only ever asks for growing powers, so the
    state is evolved forward from the last power instead of starting again.
    """

    def __init__(self, loader: ShockLoaderCircuit):
        from qiskit import transpile
        self.loader = loader
        basis = ["u", "cx"]
        self._a = transpile(loader.circuit, basis_gates=basis, optimization_level=1)
        self._q = transpile(loader.grover(), basis_gates=basis, optimization_level=1)
        self._power, self._state = 0, Statevector.from_instruction(self._a)
        self._cache: dict[int, float] = {}

    def good_probability(self, power: int) -> float:
        if power not in self._cache:
            if power < self._power:
                self._power, self._state = 0, Statevector.from_instruction(self._a)
            while self._power < power:
                self._state = self._state.evolve(self._q)
                self._power += 1
            self._cache[power] = float(
                self._state.probabilities([self.loader.objective_index])[1])
        return self._cache[power]

    @property
    def gate_counts(self) -> dict:
        return {"A": dict(self._a.count_ops()), "A_depth": self._a.depth(),
                "Q": dict(self._q.count_ops()), "Q_depth": self._q.depth()}


def circuit_oracle_factory(portfolio: Portfolio):
    """Drop-in replacement for ideal_oracle_factory in quantum_estimators."""
    def factory(target: Target, stratum: StratumAmplitude | None, amplitude: float):
        bucket = None if stratum is None else int(stratum.label.split()[0].rstrip("+"))
        return CircuitOracle(ShockLoaderCircuit(portfolio, target, bucket))
    return factory
