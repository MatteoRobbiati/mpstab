"""
Frequencies to a result: post-processing of what a backend measured.

Nothing here touches a backend. The point estimate for the ``"pauli"`` route
comes from qibo's own
:meth:`qibo.hamiltonians.SymbolicHamiltonian.expectation_from_samples`; only the
standard error, which qibo does not report, is computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.quantum_hardware.pauli_expansion import mpo_site_arrays

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


@dataclass(frozen=True)
class ExpectationResult:
    """
    A measured expectation value, its shot noise and its truncation budget.

    Attributes:
        value: the (real) expectation value.
        stderr: standard error from shot noise alone.
        truncation_l1: rigorous discarded-Pauli-mass bound; ``None`` for the
            ``"shadows"`` route, whose bond truncation has no L1/L2 split.
        truncation_l2: typical-case truncation estimate -- Pauli-set truncation
            for ``"pauli"``, MPO bond truncation for ``"shadows"``.
        n_settings: number of distinct circuits the shots came from.
        n_shots: total shots used.
    """

    value: float
    stderr: float
    truncation_l1: object
    truncation_l2: float
    n_settings: int
    n_shots: int

    @property
    def total_error(self) -> float:
        """
        ``sqrt(stderr**2 + truncation_l2**2)``: the number that belongs in a
        results table, since ``stderr`` alone omits the truncation bias.
        """
        return float(np.sqrt(self.stderr**2 + self.truncation_l2**2))

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"ExpectationResult({self.value:+.6f}, total_error={self.total_error:.6f}, "
            f"n_shots={self.n_shots}, n_settings={self.n_settings})"
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
    )


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
    value = plan.constant
    variance = 0.0
    n_shots = 0
    for _, coeff, sign, mpo in mpo_terms:
        arrays = mpo_site_arrays(mpo)
        sum_v = sum_v2 = 0.0
        n = 0
        for basis, freq in zip(bases, frequencies):
            setting_v, setting_v2, setting_n = _term_setting_stats(arrays, basis, freq)
            sum_v += setting_v
            sum_v2 += setting_v2
            n += setting_n
        if n > 1:
            per_shot_variance = (sum_v2 - n * (sum_v / n) ** 2) / (n - 1)
            variance += coeff**2 * per_shot_variance / n
        value += coeff * sign * sum_v / n
        n_shots = n  # identical across terms: same circuits, same shots

    return ExpectationResult(
        value=float(value),
        stderr=float(np.sqrt(variance)),
        truncation_l1=None,
        truncation_l2=plan.truncation_l2,
        n_settings=len(frequencies),
        n_shots=n_shots,
    )


def estimate(plan, frequencies) -> ExpectationResult:
    """Dispatch to :func:`estimate_pauli` or :func:`estimate_shadows` by ``plan.method``."""
    if plan.method == "pauli":
        return estimate_pauli(plan, frequencies)
    if plan.method == "shadows":
        return estimate_shadows(plan, frequencies)
    raise ValueError(
        f"Unknown plan method {plan.method!r}, expected 'pauli' or 'shadows'."
    )
