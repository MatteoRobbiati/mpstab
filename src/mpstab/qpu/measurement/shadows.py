"""
The ``"shadows"`` method: classical shadows from random single-qubit Pauli
bases, contracted with the tail-folded observable MPO through the canonical dual.

Random local Pauli-basis measurements are the six-effect-per-qubit POVM

``Pi_k in {|0><0|/3, |1><1|/3, |+><+|/3, |-><-|/3, |+i><+i|/3, |-i><-i|/3}``

indexed ``k = 0..5`` in that order (:func:`outcome_index`). Every operator is
handled as the MPS of its Pauli coefficients (``O = sum_P c_P P``, as
:func:`~mpstab.qpu.measurement.pauli_expansion.mpo_to_pauli_mps` returns it).
Because each effect is a fixed combination of single-qubit Paulis, the
canonical dual is a site-local map (:data:`LOCAL_CANONICAL_DUAL_MAP`): applied
to ``O``'s Pauli-coefficient MPS it gives, at ``O``'s own bond dimension, the
MPS ``omega`` of per-outcome estimator values (:func:`canonical_omega_seed`).
Estimating is then a batched lookup of ``omega`` at every measured outcome
(:func:`evaluate_omega`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
    basis_measurement,
    weighted_mean_and_variance,
)
from mpstab.qpu.measurement.pauli_expansion import (
    fold_truncation_l2,
    mpo_to_pauli_mps,
    shadow_variance_from_mpo,
)
from mpstab.qpu.synthesis.utils import head_to_qibo_circuit

#: Outcome index ``0..5`` of each (basis, bit).
_BASIS_TO_OUTCOME_PAIR = {"Z": (0, 1), "X": (2, 3), "Y": (4, 5)}


def outcome_index(basis: str, bit: int) -> int:
    """The outcome index ``0..5`` for measuring ``basis`` and getting ``bit``."""
    return _BASIS_TO_OUTCOME_PAIR[basis][bit]


def _local_effect_map() -> np.ndarray:
    """
    ``(6, 4)`` real matrix ``E`` with ``Pi_k = sum_p E[k, p] P_p``, the six
    sub-normalised Pauli-eigenstate effects expanded in the ``(I, X, Y, Z)``
    basis. Every row is ``Pi_k = (I +- P_axis) / 6``.
    """
    axis_column = {"X": 1, "Y": 2, "Z": 3}
    effect_map = np.zeros((6, 4))
    effect_map[:, 0] = 1 / 6
    for row, (axis, sign) in enumerate(
        [("Z", 1), ("Z", -1), ("X", 1), ("X", -1), ("Y", 1), ("Y", -1)]
    ):
        effect_map[row, axis_column[axis]] = sign / 6
    return effect_map


#: ``(6, 4)``: ``Pi_k = sum_p LOCAL_EFFECT_MAP[k, p] P_p``.
LOCAL_EFFECT_MAP = _local_effect_map()

#: ``(6, 4)``: the canonical dual ``D_k = 9 Pi_k - I``, scaled so that applying
#: it site-wise to a Pauli-coefficient MPS gives the canonical-dual ``omega``,
#: see :func:`canonical_omega_seed`.
LOCAL_CANONICAL_DUAL_MAP = 6 * LOCAL_EFFECT_MAP @ np.diag([1.0, 3.0, 3.0, 3.0])


def real_pauli_mps(pauli_mps: list) -> list:
    """
    An exactly equivalent, real-valued copy of a Pauli-coefficient MPS.

    A Hermitian operator's Pauli coefficients are real, but the MPS
    :func:`~mpstab.qpu.measurement.pauli_expansion.mpo_to_pauli_mps` returns
    can still carry complex gauge phases on its bonds -- and the real part of a
    product is not the product of the real parts, so taking ``site.real`` site
    by site would describe a *different* operator. Instead every site
    ``A = R + iI`` becomes the real block ``[[R, -I], [I, R]]`` (the 2x2 real
    representation of a complex number, bond dimension doubled), and the
    boundary sites keep only the row/column that selects the real part of the
    full contraction. The doubled train is then recompressed exactly by
    :func:`_compress_real_mps`: the coefficients being real, the rank of every
    unfolding is the same over the reals as over the complexes, so the bond
    dimension comes back to at most the original one. Sites that are already
    real are returned as they are, at their own bond dimension.
    """
    if all(
        np.abs(np.imag(site)).max() <= 1e-12 * max(1.0, np.abs(site).max())
        for site in pauli_mps
    ):
        return [np.real(site) for site in pauli_mps]

    n = len(pauli_mps)
    real_mps = []
    for index, site in enumerate(pauli_mps):
        re, im = np.real(site), np.imag(site)
        block = np.concatenate(
            [np.concatenate([re, -im], axis=2), np.concatenate([im, re], axis=2)],
            axis=1,
        )
        if index == 0:
            block = block[:, :1, :]
        if index == n - 1:
            block = block[:, :, :1]
        real_mps.append(block)
    return _compress_real_mps(real_mps)


def _compress_real_mps(mps: list, rtol: float = 1e-12) -> list:
    """
    Exact recompression of a real MPS (site tensors ``(p, left, right)``).

    A left-to-right QR sweep puts the train in left-canonical form; a
    right-to-left SVD sweep then drops only the singular values below
    ``rtol`` times the largest one on each bond, so the represented tensor is
    unchanged and every bond shrinks to the rank of its unfolding.
    """
    mps = [site.copy() for site in mps]
    for k in range(len(mps) - 1):
        p, left, right = mps[k].shape
        q, r = np.linalg.qr(mps[k].transpose(1, 0, 2).reshape(left * p, right))
        mps[k] = q.reshape(left, p, -1).transpose(1, 0, 2)
        mps[k + 1] = np.einsum("ab,pbc->pac", r, mps[k + 1])
    for k in range(len(mps) - 1, 0, -1):
        p, left, right = mps[k].shape
        u, s, vt = np.linalg.svd(
            mps[k].transpose(1, 0, 2).reshape(left, p * right), full_matrices=False
        )
        keep = max(1, int((s > rtol * s[0]).sum()))
        mps[k] = vt[:keep].reshape(keep, p, right).transpose(1, 0, 2)
        mps[k - 1] = np.einsum("pab,bc->pac", mps[k - 1], u[:, :keep] * s[:keep])
    return mps


def canonical_omega_seed(pauli_mps: list) -> list:
    """
    The canonical-dual ``omega`` MPS of an operator: site tensors shaped
    ``(6, left, right)`` whose value at an outcome string is the classical-shadow
    estimate of the operator from that single shot.

    Args:
        pauli_mps: the operator's Pauli-coefficient MPS, as
            :func:`~mpstab.qpu.measurement.pauli_expansion.mpo_to_pauli_mps`
            returns it; made real by :func:`real_pauli_mps` first.
    """
    return [
        np.einsum("kp,plr->klr", LOCAL_CANONICAL_DUAL_MAP, site)
        for site in real_pauli_mps(pauli_mps)
    ]


def shots_to_outcomes(bases: tuple, frequencies: list) -> tuple:
    """
    Flatten a shadows plan's ``(bases, frequencies)`` into ``(outcomes,
    counts)``: one row of per-qubit outcome indices per distinct
    ``(circuit, bitstring)`` pair, and its measured shot count.

    Args:
        bases: one basis string per circuit, as the plan's ``recombination``
            carries it.
        frequencies: one ``{bitstring: count}`` dict per circuit, in the same
            order.

    Returns:
        ``(outcomes, counts)``: an ``(M, n)`` int array and an ``(M,)`` float
        array.
    """
    nqubits = len(bases[0]) if bases else 0
    outcomes, counts = [], []
    for basis, freq in zip(bases, frequencies):
        pairs = [_BASIS_TO_OUTCOME_PAIR[label] for label in basis]
        for bitstring, count in freq.items():
            outcomes.append([pairs[q][int(bitstring[q])] for q in range(nqubits)])
            counts.append(count)
    return (
        np.asarray(outcomes, dtype=np.int64).reshape(-1, nqubits),
        np.asarray(counts, dtype=np.float64),
    )


def evaluate_omega(omega_mps: list, outcomes: np.ndarray) -> np.ndarray:
    """
    ``omega[k_1, ..., k_n]`` at every row of ``outcomes``, batched: a chained
    bond-matrix lookup costing ``O(n_rows * n * chi**2)``.
    """
    left = np.ones((outcomes.shape[0], 1))
    for site, tensor in enumerate(omega_mps):
        left = np.einsum("ml,mlr->mr", left, tensor[outcomes[:, site]])
    return left[:, 0]


def estimate_with_omega(
    plan, frequencies, outcomes, weights, omega_of, n_shots
) -> ExpectationResult:
    """
    Sum each term's ``omega`` estimate over the shots ``(outcomes, weights)``.

    Args:
        plan, frequencies: a shadows-shaped plan and its measured frequencies.
        outcomes, weights: the shots the estimate is averaged over.
        omega_of: maps a term's Pauli-coefficient MPS to its ``omega`` MPS.
        n_shots: the shots to report, including any not averaged over.
    """
    mpo_terms, _ = plan.recombination
    n = float(weights.sum())
    value, variance = plan.constant, 0.0
    for _, coefficient, sign, mpo in mpo_terms:
        omega = omega_of(mpo_to_pauli_mps(mpo))
        mean, per_shot_variance = weighted_mean_and_variance(
            evaluate_omega(omega, outcomes), weights
        )
        value += coefficient * sign * mean
        if n > 1:
            variance += coefficient**2 * per_shot_variance / n
    return plan.result(value, np.sqrt(variance), n_shots, len(frequencies))


#: What :class:`ClassicalShadows` can do with a non-trivial Clifford residual.
TAIL_HANDLINGS = ("append_to_head_circuit", "fold")


@dataclass(frozen=True)
class ClassicalShadows(MeasurementMethod):
    """
    The ``"shadows"`` method: measure random single-qubit Pauli bases and
    contract each snapshot with the tail-folded observable MPO.

    Args:
        shots_per_setting: shots reused per random basis. Larger values cut
            the number of distinct circuits at the price of correlated
            snapshots.
        tail_handling: what to do with a non-trivial Clifford residual left by
            the resynthesis. ``"append_to_head_circuit"`` (default) runs it as
            extra gates after the head; ``"fold"`` reabsorbs it into the tail
            MPO (see
            :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.build_post_head_mpo`),
            so the device runs the head alone at a possibly much higher shadow
            variance.

    Raises:
        ValueError: on an unknown ``tail_handling``.
    """

    shots_per_setting: int = 1
    tail_handling: str = "append_to_head_circuit"

    name: ClassVar[str] = "shadows"

    def __post_init__(self):
        if self.tail_handling not in TAIL_HANDLINGS:
            raise ValueError(
                f"Unknown tail_handling {self.tail_handling!r}, expected one of "
                f"{TAIL_HANDLINGS}."
            )

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
        Fold each term's tail MPO and pick one random basis per setting.

        With ``epsilon`` the shots are sized from the exact snapshot variance of
        :func:`~mpstab.qpu.measurement.pauli_expansion.shadow_variance_from_mpo`.
        ``truncation_l2`` is estimated from each folded MPO's norm (see
        :func:`~mpstab.qpu.measurement.pauli_expansion.fold_truncation_l2`), so
        no reference fold and no head state are built.
        """
        nqubits = model.nqubits
        trivial_residual = model.stab_engine.is_identity(resynthesis.residual_circuit)
        residual = None
        if self.tail_handling == "fold" and not trivial_residual:
            residual = resynthesis.residual_circuit

        mpo_terms = []
        truncation_l2 = 0.0
        for pauli, coefficient in terms.items():
            operator, sign = model.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension, residual=residual
            )
            # From here on the tail is plain arrays, whatever the engine.
            operator = model.tn_engine.operator_site_arrays(operator)
            mpo_terms.append((pauli, coefficient, sign, operator))
            if max_bond_dimension is not None:
                truncation_l2 += fold_truncation_l2(coefficient, operator, nqubits)

        head = resynthesis.circuit
        if not trivial_residual and self.tail_handling == "append_to_head_circuit":
            head = head + head_to_qibo_circuit(
                list(resynthesis.residual_gates), nqubits
            )

        if epsilon is not None:
            variance = sum(
                c**2 * shadow_variance_from_mpo(mpo) for _, c, _, mpo in mpo_terms
            )
            n_shots = int(np.ceil(variance / epsilon**2))

        n_settings = int(np.ceil(n_shots / self.shots_per_setting))
        rng = np.random.default_rng(seed)
        basis_indices = rng.integers(0, 3, size=(n_settings, nqubits), dtype=np.uint8)
        bases = ["".join("XYZ"[b] for b in row) for row in basis_indices]
        shots = [self.shots_per_setting] * n_settings
        if n_settings:
            shots[-1] = n_shots - self.shots_per_setting * (n_settings - 1)

        # At most 3^n distinct bases: settings that draw the same one share its
        # circuit object instead of each building a copy of the head.
        circuits = {}
        for basis in bases:
            if basis not in circuits:
                circuits[basis] = head + basis_measurement(basis, nqubits)

        return MeasurementPlan(
            method=self.name,
            circuits=tuple(circuits[basis] for basis in bases),
            shots=tuple(shots),
            recombination=(tuple(mpo_terms), tuple(bases)),
            constant=constant,
            truncation_l1=None,
            truncation_l2=float(truncation_l2),
        )

    def estimate(self, plan, frequencies, seed=None) -> ExpectationResult:
        """
        The canonical-dual estimate over every shot; ``seed`` is unused, nothing
        is random.
        """
        _, bases = plan.recombination
        outcomes, counts = shots_to_outcomes(bases, frequencies)
        return estimate_with_omega(
            plan, frequencies, outcomes, counts, canonical_omega_seed, counts.sum()
        )
