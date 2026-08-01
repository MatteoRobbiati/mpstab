"""
Frequencies -> the measurement result. Never touches a backend: everything
here is pure post-processing of the frequency dictionaries a
:class:`~mpstab.evolutors.quantum_hardware.plan.QiboSimulator`-like backend
returned for a :class:`~mpstab.evolutors.quantum_hardware.plan.MeasurementPlan`'s
circuits.

The point-estimate value comes from qibo's own
:meth:`qibo.hamiltonians.SymbolicHamiltonian.expectation_from_samples`, which
already implements the diagonal-observable-from-frequencies computation
correctly; only the standard error (which that method does not report) is
computed here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.evolutors.quantum_hardware.tail import mpo_site_arrays

#: 3 u^dag |b><b| u - I for the six (basis, outcome) pairs, ``u`` the
#: single-qubit Clifford that rotates the given Pauli into the Z frame.
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_BASIS_ROTATIONS = {"X": _H, "Y": _H @ _SDG, "Z": np.eye(2, dtype=complex)}
_BASIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
_SNAPSHOT_FACTORS = np.empty((3, 2, 2, 2), dtype=complex)
for _label, _i in _BASIS_INDEX.items():
    _u = _BASIS_ROTATIONS[_label]
    for _b in (0, 1):
        _ket = _u.conj().T[:, _b : _b + 1]
        _SNAPSHOT_FACTORS[_i, _b] = 3.0 * (_ket @ _ket.conj().T) - np.eye(2)


@dataclass(frozen=True)
class ExpectationResult:
    """
    The measurement result: a value, its shot-noise standard error, and the
    systematic truncation budget that goes with it.

    Attributes:
        value: the (real) expectation value.
        stderr: standard error from shot noise alone.
        truncation_l1: rigorous discarded-Pauli-mass bound (``"pauli"``
            route only; ``None`` for ``"shadows"``, which has no meaningful
            L1/L2 split for its bond-truncation error).
        truncation_l2: typical-case truncation estimate (Pauli-set truncation
            for ``"pauli"``, MPO bond truncation for ``"shadows"``).
        n_settings: distinct circuits the shots came from.
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
        """``sqrt(stderr**2 + truncation_l2**2)``: the number that belongs in
        a results table, since ``stderr`` alone omits the systematic
        truncation bias."""
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
    Exact per-shot sample variance of ``sum_i coeff_i * parity_i(bitstring)``
    over one measurement setting, i.e. the variance of a *single* shot's
    value, not yet of the mean (divide by the setting's shot count for that).

    Not something qibo's ``expectation_from_samples`` provides (it returns the
    mean only), and exact rather than a per-member sum (as an
    independent-terms approximation would give) since the joint frequency
    table already carries the members' full covariance for free.
    """
    total = sum(freq.values())
    if total <= 1:
        return 0.0
    values, weights = [], []
    for bitstring, count in freq.items():
        bits = [int(b) for b in bitstring]
        v = sum(
            coeff * (-1) ** sum(bits[q] for q in support)
            for support, coeff in weighted_supports
        )
        values.append(v)
        weights.append(count)
    values = np.asarray(values)
    weights = np.asarray(weights)
    mean = float(np.sum(weights * values) / total)
    return float(np.sum(weights * (values - mean) ** 2) / (total - 1))


def estimate_pauli(plan, frequencies) -> ExpectationResult:
    """Turn a ``"pauli"`` :class:`~mpstab.evolutors.quantum_hardware.plan.MeasurementPlan`'s
    frequencies into an :class:`ExpectationResult`."""
    groups, coefficients = plan.recombination
    nqubits = len(next(iter(coefficients)))

    value = plan.constant
    variance = 0.0
    n_shots = 0
    for group, freq in zip(groups, frequencies):
        n_shots += sum(freq.values())
        weighted_supports = []
        form = 0
        for member in group.members:
            coeff = (
                coefficients[member].real
                if hasattr(coefficients[member], "real")
                else coefficients[member]
            )
            support = tuple(q for q, label in enumerate(member) if label != "I")
            if not support:
                value += coeff  # identity member: parity is always 1, no shot noise
                continue
            weighted_supports.append((support, coeff))
            term = coeff
            for q in support:
                term *= symbols.Z(q)
            form += term
        if form != 0:
            value += SymbolicHamiltonian(
                form=form, nqubits=nqubits
            ).expectation_from_samples(freq)
        n = sum(freq.values())
        if n > 1:
            variance += _variance_from_frequencies(freq, weighted_supports) / n

    stderr = float(np.sqrt(variance))
    return ExpectationResult(
        value=float(value),
        stderr=stderr,
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
    """Turn a ``"shadows"`` :class:`~mpstab.evolutors.quantum_hardware.plan.MeasurementPlan`'s
    frequencies into an :class:`ExpectationResult`."""
    mpo_terms, bases = plan.recombination
    value = plan.constant
    variance = 0.0
    n_shots = 0
    for label, coeff, sign, mpo in mpo_terms:
        arrays = mpo_site_arrays(mpo)
        sum_v = sum_v2 = 0.0
        n = 0
        for basis, freq in zip(bases, frequencies):
            v, v2, m = _term_setting_stats(arrays, basis, freq)
            sum_v += v
            sum_v2 += v2
            n += m
        mean = sign * sum_v / n
        if n > 1:
            per_shot_variance = (sum_v2 - n * (sum_v / n) ** 2) / (n - 1)
            variance += coeff**2 * per_shot_variance / n
        value += coeff * mean
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
