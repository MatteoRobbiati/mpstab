"""
Layered correctness checks for `mpstab.quantum_hardware.tnice`, the TN-ICE
post-processing that replaces the fixed canonical-dual contraction of the
``"tnice"`` route (sharing its circuits/shots with ``"shadows"``, see
`HSynthSMPO._shadow_plan`) with an MPS-optimized dual estimator.

Each test targets one layer in isolation, mirroring `test_pauli_sampling.py`:
the fixed effect/dual maps against a brute-force dense reconstruction, then
the canonical seed against `estimate.py`'s own per-shot snapshot factors (the
"zero sweeps reproduces shadows" sanity check), then the DMRG-like fit against
its own optimization objective. `test_expectation_methods.py` covers the next
layer up (the `HSynthSMPO` integration and cross-route agreement).
"""

import numpy as np
import pytest
from qibo import set_backend

from mpstab.engines import QuimbEngine
from mpstab.pauli import PAULI_LABELS, PAULI_MATRICES
from mpstab.quantum_hardware import tnice
from mpstab.quantum_hardware.estimate import (
    _BASIS_INDEX,
    _SNAPSHOT_FACTORS,
    _term_setting_stats,
)
from mpstab.quantum_hardware.pauli_expansion import mpo_site_arrays, mpo_to_pauli_mps

set_backend("numpy")


def _folded_mpo(base: str, generators, angles, max_bond=None):
    """A tail-folded MPO with nontrivial bond dimension, no HSynthSMPO needed."""
    engine = QuimbEngine()
    mpo = engine.pauli_mpo(base)
    for generator, angle in zip(generators, angles):
        mpo = engine.conjugate_operator(mpo, generator, angle, max_bond)
    return mpo


@pytest.fixture
def scrambled_mpo():
    """A 4-qubit Hermitian operator scrambled enough to have a nontrivial bond
    dimension and several sizeable Pauli terms -- conjugation by real-angle
    Pauli rotations keeps it Hermitian throughout."""
    rng = np.random.default_rng(3)
    generators = ["YYII", "IXZY", "ZIXY", "YZYX"]
    angles = [rng.uniform(0.3, 1.5) for _ in generators]
    return _folded_mpo("ZZZI", generators, angles)


# ---------------------------------------------------------------------------
# 1. The fixed effect/dual maps
# ---------------------------------------------------------------------------


def test_local_effect_map_matches_the_povm_effects():
    for basis in "XYZ":
        for bit in (0, 1):
            k = tnice.outcome_index(basis, bit)
            coeffs = tnice.LOCAL_EFFECT_MAP[k]
            operator = sum(
                c * PAULI_MATRICES[label] for c, label in zip(coeffs, PAULI_LABELS)
            )
            state = _SNAPSHOT_FACTORS[_BASIS_INDEX[basis], bit]
            # The POVM effect Pi_k = (I + 3-scaled dual)/9, recovered from the
            # dual's own known form D_k = 9 Pi_k - I (Eq. 38/41).
            expected_pi = (np.eye(2) + state) / 9
            assert np.allclose(operator, expected_pi)


def test_local_canonical_dual_map_matches_shadows_snapshot_factors():
    """
    `LOCAL_CANONICAL_DUAL_MAP`, applied to a single qubit, must equal
    `estimate.py`'s own `_SNAPSHOT_FACTORS` -- both are the classical-shadow
    inverse-channel operator `3 u^dag |b><b| u - I` for that (basis, outcome).
    """
    for basis in "XYZ":
        for bit in (0, 1):
            k = tnice.outcome_index(basis, bit)
            # divide the multi-site *2 factor back out for a single qubit
            coeffs = tnice.LOCAL_CANONICAL_DUAL_MAP[k] / 2
            operator = sum(
                c * PAULI_MATRICES[label] for c, label in zip(coeffs, PAULI_LABELS)
            )
            reference = _SNAPSHOT_FACTORS[_BASIS_INDEX[basis], bit]
            assert np.allclose(operator, reference)


# ---------------------------------------------------------------------------
# 2. Canonical seed reproduces the "shadows" per-shot estimator exactly
# ---------------------------------------------------------------------------


def test_canonical_seed_reproduces_shadows_per_shot_value(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    omega = tnice.canonical_omega_seed(pauli_mps)
    arrays = mpo_site_arrays(scrambled_mpo)

    rng = np.random.default_rng(0)
    basis = "".join(rng.choice(list("XYZ"), size=4))
    bits = rng.integers(0, 2, size=4)
    bitstring = "".join(str(b) for b in bits)

    outcome_row = np.array(
        [[tnice.outcome_index(b, bit) for b, bit in zip(basis, bits)]]
    )
    value_tnice = tnice.evaluate_omega(omega, outcome_row)[0]

    sum_v, sum_v2, n = _term_setting_stats(arrays, basis, {bitstring: 1})
    value_shadows = sum_v / n

    assert value_tnice == pytest.approx(value_shadows, abs=1e-8)


def test_reconstruction_error_is_near_zero_at_the_seed(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    omega = tnice.canonical_omega_seed(pauli_mps)
    assert tnice.reconstruction_error(pauli_mps, omega) < 1e-6


# ---------------------------------------------------------------------------
# 3. The DMRG-like fit
# ---------------------------------------------------------------------------


def _random_outcomes(nqubits, n_rows, seed):
    rng = np.random.default_rng(seed)
    outcomes = rng.integers(0, 6, size=(n_rows, nqubits))
    weights = np.ones(n_rows)
    return outcomes, weights


def test_fit_keeps_reconstruction_small_at_high_lambda(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    outcomes, weights = _random_outcomes(4, 500, seed=1)

    omega = tnice.fit_omega_mps(
        pauli_mps, outcomes, weights, lam=0.999, n_sweeps=3, seed=1
    )
    assert tnice.reconstruction_error(pauli_mps, omega) < 0.05


def test_fit_lowers_the_second_moment_below_the_canonical_seed(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    outcomes, weights = _random_outcomes(4, 2000, seed=2)

    seed_omega = tnice.canonical_omega_seed(pauli_mps)
    seed_second_moment = np.average(
        tnice.evaluate_omega(seed_omega, outcomes) ** 2, weights=weights
    )

    fitted = tnice.fit_omega_mps(
        pauli_mps, outcomes, weights, lam=0.99, n_sweeps=4, seed=2
    )
    fitted_second_moment = np.average(
        tnice.evaluate_omega(fitted, outcomes) ** 2, weights=weights
    )

    assert fitted_second_moment < seed_second_moment


def test_fit_with_zero_sweeps_is_the_canonical_seed(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    outcomes, weights = _random_outcomes(4, 50, seed=3)

    seed_omega = tnice.canonical_omega_seed(pauli_mps)
    fitted = tnice.fit_omega_mps(pauli_mps, outcomes, weights, n_sweeps=0)

    for seed_site, fitted_site in zip(seed_omega, fitted):
        assert np.allclose(seed_site, fitted_site)


# ---------------------------------------------------------------------------
# 4. Train/test split
# ---------------------------------------------------------------------------


def test_split_train_test_preserves_total_counts():
    rng = np.random.default_rng(4)
    outcomes = rng.integers(0, 6, size=(30, 3))
    counts = rng.integers(1, 5, size=30).astype(float)

    (train_o, train_w), (test_o, test_w) = tnice.split_train_test(
        outcomes, counts, test_fraction=0.5, seed=4
    )
    assert train_w.sum() + test_w.sum() == pytest.approx(counts.sum())


def test_split_train_test_zero_fraction_keeps_everything_in_train():
    rng = np.random.default_rng(5)
    outcomes = rng.integers(0, 6, size=(10, 2))
    counts = np.ones(10)

    (train_o, train_w), (test_o, test_w) = tnice.split_train_test(
        outcomes, counts, test_fraction=0.0, seed=5
    )
    assert train_w.sum() == pytest.approx(10.0)
    assert test_w.size == 0
