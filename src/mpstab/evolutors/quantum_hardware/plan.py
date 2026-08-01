"""
Pauli terms -> circuits: qubit-wise-commuting grouping, shot allocation and
circuit construction. Everything here is classical and evolutor-free: it never
touches an MPO or a tensor-network engine, only the already-sampled/folded
Pauli coefficients (:mod:`~mpstab.evolutors.quantum_hardware.tail`) or the
already-built tail MPOs for the shadows route. Circuits are decided here, in
full, before anything runs on a backend; :mod:`~mpstab.evolutors.quantum_hardware.estimate`
only ever reads the frequencies a backend returns.

Basis rotation is qibo's job: ``gates.M(*qubits, basis=[...])`` measures each
qubit in the given single-qubit Pauli basis directly, so no separate
basis-change circuit is built here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from qibo import gates

from mpstab.evolutors.quantum_hardware.tail import (
    hamming_weight,
    shadow_variance_from_mpo,
)

_BASIS_GATES = {"X": gates.X, "Y": gates.Y, "Z": gates.Z}


def _qwc(a: str, b: str) -> bool:
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
    Greedy, largest-coefficient-first grouping of Pauli strings into
    qubit-wise-commuting (QWC) sets: only this restricted notion of commuting
    is grouped (not the more general "fully commuting" partition), since that
    would need an entangling Clifford basis change, adding two-qubit gates to
    the very head circuit this scheme exists to keep cheap. Finding the
    minimum number of QWC groups is NP-hard, so this greedy count is an upper
    bound on the true number of settings.

    Args:
        strings: Pauli strings to group.
        weights: optional per-string sort key, e.g. ``|c_P|``.

    Returns:
        A tuple of :class:`QWCGroup`, leftover ``I`` in the merged setting
        resolved to ``Z`` (an arbitrary but fixed choice).
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
        for g, setting in enumerate(settings):
            if _qwc(setting, string):
                settings[g] = _merge_setting(setting, string)
                members[g].append(string)
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
        return max((hamming_weight(string) for string in self.coefficients), default=0)

    @property
    def group_variances(self) -> tuple:
        """
        Per-group a-priori variance, ``sum_{P in group} |c_P|**2``: each
        member's single-shot parity variance is bounded by 1 (saturated when
        ``<P> = 0``), neglecting covariance between members of the same group.
        Needs no sampling and no quantum state; this is what shot allocation
        and shot-budget sizing scale with.
        """
        return tuple(
            float(sum(abs(self.coefficients[m]) ** 2 for m in group.members))
            for group in self.groups
        )

    def shots_for_precision(self, epsilon: float) -> float:
        """
        Predicted total shots for standard error ``epsilon``, under
        Neyman-optimal (variance-proportional) allocation across groups:
        ``(sum_G sqrt(V_G))**2 / epsilon**2``.

        Deliberately *not* ``||c||_1**2 / epsilon**2`` (:meth:`shots_upper_bound`):
        by Cauchy-Schwarz, ``sqrt(V_G) <= sum_{P in G} |c_P|`` for every group,
        so the L1 estimate always overestimates, and the gap is exactly the
        cost win grouping several terms under one setting buys. Using the L1
        bound to compare routes can get the comparison backwards: on a J1-J2
        Hamiltonian at moderate size it predicted the Pauli route losing to
        shadows by 3.6x, while the true cost had it winning by 3.4x.
        """
        return float(sum(np.sqrt(v) for v in self.group_variances)) ** 2 / epsilon**2

    def shots_upper_bound(self, epsilon: float) -> float:
        """Worst-case predicted shots, ``||c||_1**2 / epsilon**2``: what
        :meth:`shots_for_precision` would need if every group's terms were
        perfectly correlated. Kept for comparison, not for method selection."""
        return self.l1_norm**2 / epsilon**2


def build_measurement_plan(coefficients: dict) -> PauliMeasurementPlan:
    """QWC-group ``coefficients`` (``pauli_string -> c_P``) into a plan."""
    strings = list(coefficients)
    weights = [abs(coefficients[s]) for s in strings]
    return PauliMeasurementPlan(
        groups=group_qwc(strings, weights=weights), coefficients=dict(coefficients)
    )


def allocate_shots_by_variance(plan: PauliMeasurementPlan, n_shots: int) -> tuple:
    """
    Neyman-optimal per-group shot allocation, proportional to ``sqrt(V_G))``,
    matching the variance model :meth:`PauliMeasurementPlan.shots_for_precision`
    sizes its total against (unlike an L1-weighted split, which would
    allocate shots by the same worst-case assumption that mis-ranks methods).
    """
    variances = np.asarray(plan.group_variances)
    weights = np.sqrt(variances)
    if weights.sum() == 0:
        weights = np.ones(len(plan.groups))
    raw = n_shots * weights / weights.sum()
    return tuple(int(x) for x in np.maximum(1, np.round(raw)))


@dataclass(frozen=True)
class MeasurementPlan:
    """
    Circuits and their shot allocation, decided classically with no backend
    touched. ``recombination`` is route-specific:

    - ``"pauli"``: ``(groups, coefficients)`` where ``groups`` has one
      :class:`QWCGroup` per circuit (same order) and ``coefficients`` is the
      ``pauli_string -> c_P`` map the groups were built from, read by
      :func:`~mpstab.evolutors.quantum_hardware.estimate.estimate_pauli`.
    - ``"shadows"``: ``(mpo_terms, bases)`` where ``mpo_terms`` is a tuple of
      ``(label, coefficient, sign, mpo)`` and ``bases`` has one basis string
      per circuit, read by
      :func:`~mpstab.evolutors.quantum_hardware.estimate.estimate_shadows`.
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
    """Build the ``"pauli"`` route's :class:`MeasurementPlan` from a (sampled,
    tail-folded) ``{pauli_string: coefficient}`` map."""
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
    Build the ``"shadows"`` route's :class:`MeasurementPlan`: random
    single-qubit-Pauli bases, one circuit per setting.

    Args:
        mpo_terms: list of ``(label, coefficient, sign, mpo)``, one per
            Hamiltonian term (a single-entry list for a Pauli-string observable).
        shots_per_setting: shots reused per random basis. Larger values cut the
            number of distinct circuits at the price of correlated snapshots.
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


def _basis_measurement(basis: str, nqubits: int):
    """One-gate measurement layer: ``gates.M`` rotates each qubit into its
    given Pauli basis internally, so no separate basis-change circuit is built."""
    from qibo import Circuit

    circuit = Circuit(nqubits)
    circuit.add(
        gates.M(*range(nqubits), basis=[_BASIS_GATES[label] for label in basis])
    )
    return circuit


class QiboSimulator:
    """
    The default backend: runs circuits with qibo's own simulator.

    One duck-typed method is the entire backend contract -- no ``Protocol``,
    no ABC. Any object with an ``execute_circuits(circuits, nshots)`` matching
    this signature works, real hardware included: qibo and qibolab both
    produce frequency dictionaries, and
    :mod:`~mpstab.evolutors.quantum_hardware.estimate` consumes exactly that.
    """

    def execute_circuits(self, circuits, nshots: int) -> list:
        n = circuits[0].nqubits
        if n > 30:
            warnings.warn(f"{n} qubits: exact simulation is impractical above ~30.")
        return [circuit(nshots=nshots).frequencies() for circuit in circuits]
