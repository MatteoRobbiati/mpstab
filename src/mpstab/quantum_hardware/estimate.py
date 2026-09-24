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

from mpstab.quantum_hardware import tnice
from mpstab.quantum_hardware.pauli_expansion import mpo_site_arrays, mpo_to_pauli_mps

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
        truncation_l1: heuristic discarded-Pauli-mass estimate (see
            :func:`~mpstab.quantum_hardware.pauli_expansion.truncation_error_estimate`
            -- an extrapolation, not a rigorous bound); ``None`` for the
            ``"shadows"``/``"tnice"`` routes, whose bond truncation has no
            L1/L2 split.
        truncation_l2: typical-case truncation estimate -- Pauli-set truncation
            for ``"pauli"``, MPO bond truncation for ``"shadows"``/``"tnice"``.
        n_settings: number of distinct circuits the shots came from.
        n_shots: total shots used.
        retained_weight: ``"pauli"`` only, ``None`` otherwise -- see
            :attr:`~mpstab.quantum_hardware.plan.MeasurementPlan.retained_weight`.
    """

    value: float
    stderr: float
    truncation_l1: object
    truncation_l2: float
    n_settings: int
    n_shots: int
    retained_weight: float | None = None

    @property
    def total_error(self) -> float:
        """
        A practical accuracy *bound*, not a standard error itself:
        ``sqrt(stderr**2 + truncation_l2**2)``, combining two contributions of
        different nature in quadrature. ``stderr`` is a statistical quantity
        (shot noise, shrinks with more shots); ``truncation_l2`` is a
        systematic-bias estimate (bond-dimension or Pauli-coverage
        truncation, independent of shot count -- see ``truncation_l1`` and
        ``retained_weight`` to tell which one is dominating). Combining them
        in quadrature is a convenient, common convention, not a rigorous
        derivation of a single confidence interval -- but it is still the
        number to report in a results table, since ``stderr`` alone omits the
        truncation bias entirely.
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
        retained_weight=plan.retained_weight,
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


def estimate_tnice(
    plan,
    frequencies,
    lam: float = 0.999,
    n_sweeps: int = 4,
    bond_dimension: int | None = None,
    test_fraction: float = 0.5,
    seed: int | None = None,
) -> ExpectationResult:
    """
    Recombine a ``"tnice"`` plan's frequencies -- built identically to a
    ``"shadows"`` plan, see
    :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO._shadow_plan` -- with the
    TN-ICE estimator of
    :mod:`~mpstab.quantum_hardware.tnice` instead of the fixed canonical-dual
    contraction :func:`estimate_shadows` uses, so the two can be compared on
    identical measurement data.

    Splits the shots into a training half, which fits ``omega`` per term via
    :func:`~mpstab.quantum_hardware.tnice.fit_omega_mps`, and a held-out test
    half the final value and standard error come from -- avoiding the
    overfitting the paper's Sec. 7.3 warns an in-sample estimate would show.

    Args:
        plan, frequencies: as :func:`estimate_shadows`.
        lam, n_sweeps, bond_dimension: forwarded to
            :func:`~mpstab.quantum_hardware.tnice.fit_omega_mps`.
        test_fraction: fraction of each circuit's shots held out for the
            final estimate; the rest trains ``omega``.
        seed: RNG seed for the train/test split.
    """
    mpo_terms, bases = plan.recombination
    outcomes, counts = tnice.shots_to_outcomes(bases, frequencies)
    (train_outcomes, train_weights), (test_outcomes, test_weights) = (
        tnice.split_train_test(outcomes, counts, test_fraction, seed)
    )
    n_test = float(test_weights.sum())

    value = plan.constant
    variance = 0.0
    for _, coeff, sign, mpo in mpo_terms:
        pauli_mps = mpo_to_pauli_mps(mpo)
        omega = tnice.fit_omega_mps(
            pauli_mps,
            train_outcomes,
            train_weights,
            lam=lam,
            n_sweeps=n_sweeps,
            bond_dimension=bond_dimension,
            seed=seed,
        )
        term_values = tnice.evaluate_omega(omega, test_outcomes)
        mean, per_shot_variance = tnice.weighted_mean_and_variance(
            term_values, test_weights
        )
        if n_test > 1:
            variance += coeff**2 * per_shot_variance / n_test
        value += coeff * sign * mean

    return ExpectationResult(
        value=float(value),
        stderr=float(np.sqrt(variance)),
        truncation_l1=None,
        truncation_l2=plan.truncation_l2,
        n_settings=len(frequencies),
        n_shots=int(counts.sum()),
        retained_weight=None,
    )


def estimate(plan, frequencies, **kwargs) -> ExpectationResult:
    """
    Dispatch to :func:`estimate_pauli`, :func:`estimate_shadows` or
    :func:`estimate_tnice` by ``plan.method``. ``kwargs`` are forwarded only
    to :func:`estimate_tnice`, the only one of the three that takes any.
    """
    if plan.method == "pauli":
        return estimate_pauli(plan, frequencies)
    if plan.method == "shadows":
        return estimate_shadows(plan, frequencies)
    if plan.method == "tnice":
        return estimate_tnice(plan, frequencies, **kwargs)
    raise ValueError(
        f"Unknown plan method {plan.method!r}, expected 'pauli', 'shadows' or 'tnice'."
    )
