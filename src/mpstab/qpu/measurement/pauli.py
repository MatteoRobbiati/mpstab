"""
The ``"pauli"`` method: measure the sampled Pauli strings of the tail-folded
observable, grouped into qubit-wise-commuting settings.

The point estimate comes from qibo's own
:meth:`qibo.hamiltonians.SymbolicHamiltonian.expectation_from_samples`; only the
standard error, which qibo does not report, is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.pauli import weight
from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementPlan,
    basis_measurement,
)


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
    Split ``n_shots`` across groups proportionally to ``sqrt(V_G)``, respecting
    ``n_shots`` exactly.

    This is the Neyman-optimal allocation, and matches the variance model
    :meth:`PauliMeasurementPlan.shots_for_precision` sizes the total against.
    Every group needs at least one shot to be measured at all, so one shot is
    reserved for each group first; the remainder is allocated by weight and
    rounded with the largest-remainder method, so the returned shot counts
    always sum to exactly ``n_shots`` (rounding a raw fractional allocation
    per group, as a naive ``round()`` would, can drift the total away from
    ``n_shots`` in either direction).

    Raises:
        ValueError: if ``n_shots`` is smaller than the number of groups, since
            every group needs at least one shot.
    """
    n_groups = len(plan.groups)
    if n_shots < n_groups:
        raise ValueError(
            f"n_shots={n_shots} is smaller than the number of measurement "
            f"settings ({n_groups}); every setting needs at least one shot. "
            "Raise n_shots, or the epsilon target that sizes it."
        )

    weights = np.sqrt(np.asarray(plan.group_variances))
    if weights.sum() == 0:
        weights = np.ones(n_groups)

    remaining = n_shots - n_groups
    raw = remaining * weights / weights.sum()
    base = np.floor(raw).astype(int)
    leftover = remaining - int(base.sum())
    if leftover > 0:
        # Largest-remainder method: give the leftover shots, one each, to the
        # groups whose floor() discarded the largest fraction.
        remainder = raw - base
        for index in np.argsort(-remainder)[:leftover]:
            base[index] += 1

    return tuple(int(x) + 1 for x in base)  # +1: the one shot reserved above


def build_pauli_plan(
    head_circuit,
    nqubits: int,
    coefficients: dict,
    n_shots: int | None,
    epsilon: float | None,
    constant: float = 0.0,
    truncation_l1: float = 0.0,
    truncation_l2: float = 0.0,
    retained_weight: float | None = None,
    bond_truncation_l2: float = 0.0,
    antihermitian_l2: float = 0.0,
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
        truncation_l1: heuristic discarded-Pauli-mass estimate to report (see
            :func:`~mpstab.qpu.measurement.pauli_expansion.truncation_error_estimate`
            -- not a rigorous bound).
        truncation_l2: typical-case discarded-Pauli-mass estimate to report.
        retained_weight: see :attr:`MeasurementPlan.retained_weight`.
        bond_truncation_l2: see :attr:`MeasurementPlan.bond_truncation_l2`.
        antihermitian_l2: see :attr:`MeasurementPlan.antihermitian_l2`.
    """
    plan = build_measurement_plan(coefficients)
    if epsilon is not None:
        n_shots = int(np.ceil(plan.shots_for_precision(epsilon)))
    shots = allocate_shots_by_variance(plan, n_shots)

    circuits = tuple(
        head_circuit + basis_measurement(group.setting, nqubits)
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
        retained_weight=retained_weight,
        bond_truncation_l2=bond_truncation_l2,
        antihermitian_l2=antihermitian_l2,
    )


def _variance_from_frequencies(freq: dict, weighted_supports: list) -> float:
    """
    Sample variance of a *single* shot's value of ``sum_i c_i parity_i(bitstring)``
    over one measurement setting. Divide by the setting's shot count to get the
    variance of the mean.

    Exact rather than a sum of per-member variances, since the joint frequency
    table already carries the members' full covariance.
    """
    total = sum(freq.values())
    if total <= 1:
        return 0.0
    values, weights = [], []
    for bitstring, count in freq.items():
        bits = [int(b) for b in bitstring]
        values.append(
            sum(
                coeff * (-1) ** sum(bits[q] for q in support)
                for support, coeff in weighted_supports
            )
        )
        weights.append(count)
    values = np.asarray(values)
    weights = np.asarray(weights)
    mean = float(np.sum(weights * values) / total)
    return float(np.sum(weights * (values - mean) ** 2) / (total - 1))


def estimate_pauli(plan, frequencies) -> ExpectationResult:
    """Recombine a ``"pauli"`` plan's frequencies into an :class:`ExpectationResult`."""
    groups, coefficients = plan.recombination
    nqubits = len(next(iter(coefficients)))

    value = plan.constant
    variance = 0.0
    n_shots = 0
    for group, freq in zip(groups, frequencies):
        shots = sum(freq.values())
        n_shots += shots

        weighted_supports = []
        form = 0
        for member in group.members:
            coeff = float(np.real(coefficients[member]))
            support = tuple(q for q, label in enumerate(member) if label != "I")
            if not support:
                value += coeff  # identity member: parity is always 1, no shot noise
                continue
            weighted_supports.append((support, coeff))
            term = coeff
            for qubit in support:
                term *= symbols.Z(qubit)
            form += term

        if form != 0:
            value += SymbolicHamiltonian(
                form=form, nqubits=nqubits
            ).expectation_from_samples(freq)
        if shots > 1:
            variance += _variance_from_frequencies(freq, weighted_supports) / shots

    return ExpectationResult(
        value=float(value),
        stderr=float(np.sqrt(variance)),
        truncation_l1=plan.truncation_l1,
        truncation_l2=plan.truncation_l2,
        n_settings=len(frequencies),
        n_shots=n_shots,
        retained_weight=plan.retained_weight,
        bond_truncation_l2=plan.bond_truncation_l2,
        antihermitian_l2=plan.antihermitian_l2,
    )
