"""
Pauli terms to circuits: qubit-wise-commuting grouping and shot allocation.

A :class:`MeasurementPlan` fixes every circuit and its shot count before
anything runs, so the same plan can be replayed on a simulator or on hardware
and :mod:`~mpstab.quantum_hardware.estimate` only ever reads back frequencies.

Basis rotation is left to qibo: ``gates.M(*qubits, basis=[...])`` measures each
qubit in the given single-qubit Pauli basis, so no basis-change circuit is built
here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qibo import Circuit, gates

from mpstab.pauli import weight
from mpstab.quantum_hardware.pauli_expansion import shadow_variance_from_mpo

_BASIS_GATES = {"X": gates.X, "Y": gates.Y, "Z": gates.Z}


def _qubitwise_commuting(a: str, b: str) -> bool:
    """Whether ``a`` and ``b`` agree at every qubit where neither is ``I``."""
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


def _merge_setting(setting: str, string: str) -> str:
    return "".join(a if a != "I" else b for a, b in zip(setting, string))


@dataclass(frozen=True)
class QWCGroup:
    """One measurement setting: a merged basis string and its Pauli members."""

    setting: str
    members: tuple


def group_qwc(strings, weights=None) -> tuple:
    """
    Group Pauli strings into qubit-wise-commuting sets, largest coefficient first.

    Only qubit-wise commutation is exploited, not the more general "fully
    commuting" partition, because the latter needs an entangling Clifford basis
    change that would add two-qubit gates to the very head circuit this scheme
    exists to keep cheap. Minimising the number of QWC groups is NP-hard, so this
    greedy count is an upper bound on the true number of settings.

    Args:
        strings: Pauli strings to group.
        weights: optional per-string sort key, e.g. ``|c_P|``.

    Returns:
        A tuple of :class:`QWCGroup`, with leftover ``I`` in each merged setting
        resolved to ``Z``.
    """
    strings = list(strings)
    order = (
        range(len(strings))
        if weights is None
        else sorted(range(len(strings)), key=lambda i: -abs(weights[i]))
    )

    settings: list = []
    members: list = []
    for i in order:
        string = strings[i]
        for group, setting in enumerate(settings):
            if _qubitwise_commuting(setting, string):
                settings[group] = _merge_setting(setting, string)
                members[group].append(string)
                break
        else:
            settings.append(string)
            members.append([string])

    return tuple(
        QWCGroup(setting=setting.replace("I", "Z"), members=tuple(group_members))
        for setting, group_members in zip(settings, members)
    )


@dataclass(frozen=True)
class PauliMeasurementPlan:
    """QWC groups plus the Pauli coefficients they were built from."""

    groups: tuple
    coefficients: dict

    @property
    def n_settings(self) -> int:
        return len(self.groups)

    @property
    def l1_norm(self) -> float:
        return float(sum(abs(c) for c in self.coefficients.values()))

    @property
    def max_weight(self) -> int:
        return max((weight(string) for string in self.coefficients), default=0)

    @property
    def group_variances(self) -> tuple:
        """
        Per-group a-priori variance, ``sum_{P in group} |c_P|**2``.

        Each member's single-shot parity variance is bounded by 1, saturated when
        ``<P> = 0``, and covariance between members of a group is neglected.
        Needs no sampling and no state, which is what lets shot allocation and
        shot budgeting be decided up front.
        """
        return tuple(
            float(sum(abs(self.coefficients[m]) ** 2 for m in group.members))
            for group in self.groups
        )

    def shots_for_precision(self, epsilon: float) -> float:
        """
        Total shots predicted for standard error ``epsilon`` under
        variance-proportional (Neyman-optimal) allocation across groups,
        ``(sum_G sqrt(V_G))**2 / epsilon**2``.

        Prefer this over :meth:`shots_upper_bound` when comparing measurement
        routes: by Cauchy-Schwarz ``sqrt(V_G) <= sum_{P in G} |c_P|``, so the L1
        bound always overestimates, by exactly the saving grouping buys. Sizing
        budgets from the L1 bound can rank the routes backwards.
        """
        return float(sum(np.sqrt(v) for v in self.group_variances)) ** 2 / epsilon**2

    def shots_upper_bound(self, epsilon: float) -> float:
        """
        Worst-case predicted shots, ``||c||_1**2 / epsilon**2``: what
        :meth:`shots_for_precision` would need if every group's terms were
        perfectly correlated.
        """
        return self.l1_norm**2 / epsilon**2


def build_measurement_plan(coefficients: dict) -> PauliMeasurementPlan:
    """QWC-group a ``{pauli_string: c_P}`` map into a :class:`PauliMeasurementPlan`."""
    strings = list(coefficients)
    weights = [abs(coefficients[s]) for s in strings]
    return PauliMeasurementPlan(
        groups=group_qwc(strings, weights=weights), coefficients=dict(coefficients)
    )


def allocate_shots_by_variance(plan: PauliMeasurementPlan, n_shots: int) -> tuple:
    """
    Split ``n_shots`` across groups proportionally to ``sqrt(V_G)``.

    This is the Neyman-optimal allocation, and matches the variance model
    :meth:`PauliMeasurementPlan.shots_for_precision` sizes the total against.
    """
    weights = np.sqrt(np.asarray(plan.group_variances))
    if weights.sum() == 0:
        weights = np.ones(len(plan.groups))
    raw = n_shots * weights / weights.sum()
    return tuple(int(x) for x in np.maximum(1, np.round(raw)))


@dataclass(frozen=True)
class MeasurementPlan:
    """
    Circuits and their shot allocation, with whatever the estimator needs to
    recombine the results.

    ``recombination`` is route-specific:

    - ``"pauli"``: ``(groups, coefficients)``, one :class:`QWCGroup` per circuit
      in the same order, plus the ``{pauli_string: c_P}`` map the groups came
      from.
    - ``"shadows"``: ``(mpo_terms, bases)``, where ``mpo_terms`` holds
      ``(label, coefficient, sign, mpo)`` per Hamiltonian term and ``bases`` has
      one basis string per circuit.
    """

    method: str
    circuits: tuple
    shots: tuple
    recombination: object
    constant: float
    truncation_l1: float | None
    truncation_l2: float


def build_pauli_plan(
    head_circuit,
    nqubits: int,
    coefficients: dict,
    n_shots: int | None,
    epsilon: float | None,
    constant: float = 0.0,
    truncation_l1: float = 0.0,
    truncation_l2: float = 0.0,
) -> MeasurementPlan:
    """
    Build the ``"pauli"`` route's plan: one circuit per QWC group.

    Args:
        head_circuit: the resynthesised head, without measurements.
        nqubits: circuit width.
        coefficients: sampled, tail-folded ``{pauli_string: coefficient}``.
        n_shots: fixed total shot budget, or ``None`` to size it from ``epsilon``.
        epsilon: target standard error, or ``None`` when ``n_shots`` is given.
        constant: observable offset, added back by the estimator.
        truncation_l1: rigorous discarded-Pauli-mass bound to report.
        truncation_l2: typical-case discarded-Pauli-mass estimate to report.
    """
    plan = build_measurement_plan(coefficients)
    if epsilon is not None:
        n_shots = int(np.ceil(plan.shots_for_precision(epsilon)))
    shots = allocate_shots_by_variance(plan, n_shots)

    circuits = tuple(
        head_circuit + _basis_measurement(group.setting, nqubits)
        for group in plan.groups
    )
    return MeasurementPlan(
        method="pauli",
        circuits=circuits,
        shots=shots,
        recombination=(plan.groups, plan.coefficients),
        constant=constant,
        truncation_l1=truncation_l1,
        truncation_l2=truncation_l2,
    )


def build_shadow_plan(
    head_circuit,
    nqubits: int,
    mpo_terms: list,
    n_shots: int | None,
    epsilon: float | None,
    shots_per_setting: int = 1,
    seed: int | None = None,
    constant: float = 0.0,
    truncation_l2: float = 0.0,
) -> MeasurementPlan:
    """
    Build the ``"shadows"`` route's plan: one circuit per random single-qubit
    Pauli basis.

    Args:
        head_circuit: the resynthesised head, without measurements.
        nqubits: circuit width.
        mpo_terms: ``(label, coefficient, sign, mpo)`` per Hamiltonian term; a
            single-entry list for a Pauli-string observable.
        n_shots: fixed total shot budget, or ``None`` to size it from ``epsilon``.
        epsilon: target standard error, sized from the exact snapshot variance of
            :func:`~mpstab.quantum_hardware.pauli_expansion.shadow_variance_from_mpo`.
        shots_per_setting: shots reused per random basis. Larger values cut the
            number of distinct circuits at the price of correlated snapshots.
        seed: RNG seed for the random bases.
        constant: observable offset, added back by the estimator.
        truncation_l2: MPO bond-truncation estimate to report.
    """
    predicted_variance = sum(
        coeff**2 * shadow_variance_from_mpo(mpo) for _, coeff, _, mpo in mpo_terms
    )
    if epsilon is not None:
        n_shots = int(np.ceil(predicted_variance / epsilon**2))

    n_settings = int(np.ceil(n_shots / shots_per_setting))
    rng = np.random.default_rng(seed)
    basis_indices = rng.integers(0, 3, size=(n_settings, nqubits), dtype=np.uint8)

    remaining = n_shots
    circuits, bases, shots = [], [], []
    for setting in range(n_settings):
        basis = "".join("XYZ"[b] for b in basis_indices[setting])
        this_shots = min(shots_per_setting, remaining)
        remaining -= this_shots
        circuits.append(head_circuit + _basis_measurement(basis, nqubits))
        bases.append(basis)
        shots.append(this_shots)

    return MeasurementPlan(
        method="shadows",
        circuits=tuple(circuits),
        shots=tuple(shots),
        recombination=(tuple(mpo_terms), tuple(bases)),
        constant=constant,
        truncation_l1=None,
        truncation_l2=truncation_l2,
    )


def _basis_measurement(basis: str, nqubits: int) -> Circuit:
    """A single ``gates.M`` layer measuring each qubit in its given Pauli basis."""
    circuit = Circuit(nqubits)
    circuit.add(
        gates.M(*range(nqubits), basis=[_BASIS_GATES[label] for label in basis])
    )
    return circuit
