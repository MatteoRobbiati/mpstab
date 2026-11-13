"""
The ``"shadows"`` method: classical shadows from random single-qubit Pauli
bases, contracted with the tail-folded observable MPO through the canonical dual.
"""

from __future__ import annotations

import numpy as np

from mpstab.qpu.measurement.abstract import (
    ExpectationResult,
    MeasurementPlan,
    basis_measurement,
)
from mpstab.qpu.measurement.pauli_expansion import (
    mpo_site_arrays,
    shadow_variance_from_mpo,
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
