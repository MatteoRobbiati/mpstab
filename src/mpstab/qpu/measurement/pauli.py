"""
The ``"pauli"`` method: measure the sampled Pauli strings of the tail-folded
observable, grouped into qubit-wise-commuting settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import stim

from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
    basis_measurement,
    weighted_mean_and_variance,
)
from mpstab.qpu.measurement.pauli_expansion import (
    fold_pool_through_tableau,
    fold_truncation_l2,
    pool_pauli_terms,
    sample_pauli_strings,
    truncation_error_estimate,
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

        By Cauchy-Schwarz ``sqrt(V_G) <= sum_{P in G} |c_P|``, so this is never
        above the ungrouped ``||c||_1**2 / epsilon**2``: the difference is what
        grouping saves.
        """
        return float(sum(np.sqrt(v) for v in self.group_variances)) ** 2 / epsilon**2


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


@dataclass(frozen=True)
class PauliSampling(MeasurementMethod):
    """
    The ``"pauli"`` method: sample Pauli strings from each term's tail MPO,
    pool them, fold the Clifford residual into them, and measure them grouped
    into qubit-wise-commuting settings.

    Args:
        n_string_samples: Pauli strings sampled per Hamiltonian term.
    """

    n_string_samples: int = 200

    name: ClassVar[str] = "pauli"

    def build_plan(
        self,
        model,
        terms,
        constant,
        cut_index,
        resynthesis,
        max_bond_dimension,
        n_shots,
        epsilon,
        seed,
    ) -> MeasurementPlan:
        """
        Sample each term's tail MPO, pool the strings, fold the residual, group.

        Two truncations are reported, of different origin. ``truncation_l1`` /
        ``truncation_l2`` count the Pauli strings of each folded MPO that
        ``n_string_samples`` left out. ``bond_truncation_l2`` is the bias of
        folding at a finite bond dimension, estimated from each folded MPO's
        norm (see
        :func:`~mpstab.qpu.measurement.pauli_expansion.fold_fidelity`) as in the
        ``"shadows"`` method: the string truncation is measured against the
        already truncated MPO, so it cannot see it. ``antihermitian_l2`` is the
        size of the anti-Hermitian part that truncation leaves in the pooled
        coefficients, projected out by :func:`_real_coefficients`.
        """
        truncated = max_bond_dimension is not None
        rng = np.random.default_rng(seed)
        ensembles, diagnostics = [], []
        bond_truncation_l2 = 0.0
        for pauli, coefficient in terms.items():
            operator, sign = model.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension
            )
            if truncated:
                bond_truncation_l2 += fold_truncation_l2(
                    coefficient, operator, model.nqubits
                )
            ensemble = sample_pauli_strings(
                operator,
                n_samples=self.n_string_samples,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            ensembles.append((coefficient * sign, ensemble))
            l1, l2 = truncation_error_estimate(ensemble)
            diagnostics.append((abs(coefficient), l1, l2, ensemble.retained_weight))

        pooled = pool_pauli_terms(ensembles)
        if resynthesis.residual_tableau != stim.Tableau(model.nqubits):
            pooled = fold_pool_through_tableau(pooled, resynthesis.residual_tableau)

        truncation_l1, truncation_l2, retained_weight = _aggregate_truncation(
            diagnostics
        )
        grouping = build_measurement_plan(
            _real_coefficients(pooled, check=not truncated)
        )
        if epsilon is not None:
            n_shots = int(np.ceil(grouping.shots_for_precision(epsilon)))

        return MeasurementPlan(
            method=self.name,
            circuits=tuple(
                resynthesis.circuit + basis_measurement(group.setting, model.nqubits)
                for group in grouping.groups
            ),
            shots=allocate_shots_by_variance(grouping, n_shots),
            recombination=(grouping.groups, grouping.coefficients),
            constant=constant,
            truncation_l1=truncation_l1,
            truncation_l2=truncation_l2,
            retained_weight=retained_weight,
            bond_truncation_l2=float(bond_truncation_l2),
            antihermitian_l2=_antihermitian_norm(pooled),
        )

    def estimate(self, plan, frequencies, seed=None) -> ExpectationResult:
        """
        Sum each setting's mean of ``sum_P c_P parity_P(bitstring)``, with its
        standard error from the same per-bitstring values, which carry the
        members' full covariance. ``seed`` is unused, nothing is random.
        """
        groups, coefficients = plan.recombination
        value, variance, n_shots = plan.constant, 0.0, 0
        for group, freq in zip(groups, frequencies):
            supports = [
                (
                    [q for q, label in enumerate(member) if label != "I"],
                    coefficients[member],
                )
                for member in group.members
            ]
            values = [
                sum(
                    c * (-1) ** sum(int(bitstring[q]) for q in support)
                    for support, c in supports
                )
                for bitstring in freq
            ]
            counts = list(freq.values())
            mean, per_shot_variance = weighted_mean_and_variance(values, counts)
            shots = sum(counts)
            value += mean
            if shots > 1:
                variance += per_shot_variance / shots
            n_shots += shots
        return plan.result(value, np.sqrt(variance), n_shots, len(frequencies))


def _aggregate_truncation(diagnostics: list) -> tuple:
    """
    Combine per-term ``(|coeff|, l1, l2, retained_weight)`` truncation
    diagnostics.

    L1 adds by the triangle inequality, rigorous for any combination of terms;
    L2 adds in quadrature, assuming the terms' errors are independent;
    retained_weight is the ``|coeff|``-weighted average of each term's own
    :attr:`~mpstab.qpu.measurement.pauli_expansion.PauliEnsemble.retained_weight`
    -- a diagnostic for whether ``n_string_samples`` covered enough of the
    observable, not a rigorous combination of the per-term fractions.
    """
    if not diagnostics:
        return 0.0, 0.0, 1.0
    total_weight = sum(weight for weight, _, _, _ in diagnostics)
    l1 = sum(weight * term_l1 for weight, term_l1, _, _ in diagnostics)
    l2 = float(
        np.sqrt(sum((weight * term_l2) ** 2 for weight, _, term_l2, _ in diagnostics))
    )
    retained_weight = (
        sum(weight * retained for weight, _, _, retained in diagnostics) / total_weight
        if total_weight > 0
        else 1.0
    )
    return l1, l2, retained_weight


def _antihermitian_norm(pooled: dict) -> float:
    """
    ``sqrt(sum_P Im(c_P)**2)`` of a pooled ``{pauli: complex coefficient}`` map:
    the Frobenius norm, per ``2**(n/2)``, of the anti-Hermitian part of the
    operator the map describes. It is what :func:`_real_coefficients` drops
    when the tail was truncated, restricted to the sampled strings.
    """
    return float(np.sqrt(sum(np.imag(c) ** 2 for c in pooled.values())))


def _real_coefficients(pooled: dict, check: bool = True, tolerance: float = 1e-8):
    """
    The real parts of a pooled ``{pauli: complex coefficient}`` map.

    The tail-folded, Clifford-backpropagated observable is Hermitian, so with an
    exact fold every coefficient is real up to rounding, and with ``check`` a
    larger imaginary part raises: it points to a bug upstream (backpropagation,
    tableau folding, or a non-Hermitian input observable) that silently dropping
    it would hide.

    A tail truncated to a finite bond dimension is not Hermitian (the
    intermediate ``R^dag O`` is not, and Pauli-structured operators have
    degenerate singular values, so a compression picks an arbitrary subspace).
    There, pass ``check=False``: the real part is exactly the projection
    ``(O + O^dag) / 2``, the closest Hermitian operator, and the size of what is
    dropped is :func:`_antihermitian_norm`.

    Raises:
        ValueError: with ``check``, if an imaginary part exceeds ``tolerance``
            relative to its coefficient's magnitude.
    """
    for pauli, coefficient in pooled.items():
        if check and abs(np.imag(coefficient)) > tolerance * max(abs(coefficient), 1.0):
            raise ValueError(
                f"Pooled coefficient for {pauli!r} is not real within tolerance "
                f"({coefficient!r}); the tail-folded observable should be "
                "Hermitian, so this points to a bug upstream rather than "
                "floating-point noise."
            )
    return {pauli: float(np.real(c)) for pauli, c in pooled.items()}
