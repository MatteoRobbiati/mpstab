"""
The ``"shadows"`` method: classical shadows from random single-qubit Pauli
bases, contracted with the tail-folded observable MPO through the canonical dual.
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
)
from mpstab.qpu.measurement.pauli_expansion import (
    fold_fidelity,
    mpo_site_arrays,
    shadow_variance_from_mpo,
)
from mpstab.qpu.synthesis.utils import head_to_qibo_circuit


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
            :func:`~mpstab.qpu.measurement.pauli_expansion.shadow_variance_from_mpo`.
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
        circuits.append(head_circuit + basis_measurement(basis, nqubits))
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


_BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _snapshot_factors() -> np.ndarray:
    """
    ``3 u^dag |b><b| u - I`` for the six (basis, outcome) pairs, indexed
    ``[basis, outcome]``.

    ``u`` is the single-qubit Clifford rotating the given Pauli into the Z frame,
    so this is the single-site classical-shadow inverse channel.
    """
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    s_dagger = np.array([[1, 0], [0, -1j]], dtype=complex)
    rotations = {"X": hadamard, "Y": hadamard @ s_dagger, "Z": np.eye(2, dtype=complex)}

    factors = np.empty((3, 2, 2, 2), dtype=complex)
    for label, index in _BASIS_INDEX.items():
        for outcome in (0, 1):
            ket = rotations[label].conj().T[:, outcome : outcome + 1]
            factors[index, outcome] = 3.0 * (ket @ ket.conj().T) - np.eye(2)
    return factors


_SNAPSHOT_FACTORS = _snapshot_factors()


def _term_setting_stats(mpo_arrays, basis: str, freq: dict):
    """``(sum_v, sum_v2, n)`` of ``Tr[sigma_hat . mpo]`` over one setting's shots."""
    site_blocks = [
        np.einsum(
            "lrkb,obk->olr",
            array,
            _SNAPSHOT_FACTORS[_BASIS_INDEX[label]],
            optimize=True,
        )
        for array, label in zip(mpo_arrays, basis)
    ]
    sum_v = sum_v2 = 0.0
    n = 0
    for bitstring, count in freq.items():
        acc = site_blocks[0][int(bitstring[0])]
        for site in range(1, len(bitstring)):
            acc = acc @ site_blocks[site][int(bitstring[site])]
        v = float(acc[0, 0].real)
        sum_v += v * count
        sum_v2 += v * v * count
        n += count
    return sum_v, sum_v2, n


def estimate_shadows(plan, frequencies) -> ExpectationResult:
    """Recombine a ``"shadows"`` plan's frequencies into an :class:`ExpectationResult`."""
    mpo_terms, bases = plan.recombination
    # Every term's inner loop below sums over the same (bases, frequencies),
    # so its own shot count is always this same total -- computed once here
    # rather than reassigned (redundantly, to an identical value) each pass.
    n_shots = sum(sum(freq.values()) for freq in frequencies)

    value = plan.constant
    variance = 0.0
    for _, coeff, sign, mpo in mpo_terms:
        arrays = mpo_site_arrays(mpo)
        sum_v = sum_v2 = 0.0
        n = 0
        for basis, freq in zip(bases, frequencies):
            setting_v, setting_v2, setting_n = _term_setting_stats(arrays, basis, freq)
            sum_v += setting_v
            sum_v2 += setting_v2
            n += setting_n
        if n == 0:
            continue  # no shots for this term: nothing to add to value/variance
        if n > 1:
            per_shot_variance = (sum_v2 - n * (sum_v / n) ** 2) / (n - 1)
            variance += coeff**2 * per_shot_variance / n
        value += coeff * sign * sum_v / n

    return ExpectationResult(
        value=float(value),
        stderr=float(np.sqrt(variance)),
        truncation_l1=None,
        truncation_l2=plan.truncation_l2,
        n_settings=len(frequencies),
        n_shots=n_shots,
        retained_weight=None,
    )


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
            :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.build_post_head_mpo`),
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
        Fold each term's tail MPO and pick random measurement bases.

        ``truncation_l2`` is estimated from each folded MPO's norm (see
        :func:`~mpstab.qpu.measurement.pauli_expansion.fold_fidelity`), so no
        reference fold and no head state are built.
        """
        trivial_residual = resynthesis.residual_tableau == stim.Tableau(model.nqubits)
        residual = None
        if self.tail_handling == "fold" and not trivial_residual:
            residual = resynthesis.residual_tableau

        mpo_terms = []
        truncation_l2 = 0.0
        for pauli, coefficient in terms.items():
            operator, sign = model.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension, residual=residual
            )
            mpo_terms.append((pauli, coefficient, sign, operator))
            if max_bond_dimension is not None:
                truncation_l2 += abs(coefficient) * np.sqrt(
                    max(0.0, 1.0 - fold_fidelity(operator, model.nqubits))
                )

        circuit = resynthesis.circuit
        if not trivial_residual and self.tail_handling == "append_to_head_circuit":
            circuit = circuit + head_to_qibo_circuit(
                list(resynthesis.residual_gates), model.nqubits
            )

        return build_shadow_plan(
            circuit,
            model.nqubits,
            mpo_terms,
            n_shots,
            epsilon,
            shots_per_setting=self.shots_per_setting,
            seed=seed,
            constant=constant,
            truncation_l2=float(truncation_l2),
        )

    def estimate(self, plan, frequencies, seed=None) -> ExpectationResult:
        """:func:`estimate_shadows`; ``seed`` is unused, nothing is random."""
        return estimate_shadows(plan, frequencies)
