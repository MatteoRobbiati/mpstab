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
# 2b. Complex gauge phases in the Pauli-coefficient MPS
# ---------------------------------------------------------------------------


def _dense_coefficients(pauli_mps):
    vector = pauli_mps[0]
    for site in pauli_mps[1:]:
        vector = np.einsum("...l,plr->...pr", vector, site)
    return vector.reshape(-1)


def _with_complex_gauge(pauli_mps, seed):
    """The same coefficients, with a random complex gauge on the widest bond."""
    rng = np.random.default_rng(seed)
    gauged = [site.astype(complex) for site in pauli_mps]
    bond = int(np.argmax([site.shape[2] for site in gauged[:-1]]))
    chi = gauged[bond].shape[2]
    gauge = rng.normal(size=(chi, chi)) + 1j * rng.normal(size=(chi, chi))
    gauged[bond] = np.einsum("plr,rs->pls", gauged[bond], gauge)
    gauged[bond + 1] = np.einsum("sr,prq->psq", np.linalg.inv(gauge), gauged[bond + 1])
    return gauged


def test_real_pauli_mps_is_exact_under_a_complex_gauge(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    gauged = _with_complex_gauge(pauli_mps, seed=0)
    expected = _dense_coefficients(pauli_mps).real

    # The gauge changes nothing physical...
    assert np.allclose(_dense_coefficients(gauged), expected)
    # ...but the site-by-site real part is a different operator...
    naive = _dense_coefficients([site.real for site in gauged])
    assert not np.allclose(naive, expected, atol=1e-3)
    # ...while the real rewrite is exact.
    assert np.allclose(_dense_coefficients(tnice.real_pauli_mps(gauged)), expected)


def test_real_pauli_mps_does_not_grow_the_bond_dimension(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    gauged = _with_complex_gauge(pauli_mps, seed=2)
    real_mps = tnice.real_pauli_mps(gauged)

    assert all(np.isrealobj(site) for site in real_mps)
    for original, rewritten in zip(gauged, real_mps):
        assert rewritten.shape[1] <= original.shape[1]
        assert rewritten.shape[2] <= original.shape[2]


def test_canonical_seed_does_not_depend_on_the_gauge(scrambled_mpo):
    pauli_mps = mpo_to_pauli_mps(scrambled_mpo)
    gauged = _with_complex_gauge(pauli_mps, seed=1)
    outcomes, _ = _random_outcomes(4, 200, seed=7)

    plain = tnice.evaluate_omega(tnice.canonical_omega_seed(pauli_mps), outcomes)
    rotated = tnice.evaluate_omega(tnice.canonical_omega_seed(gauged), outcomes)

    assert np.allclose(plain, rotated)
    assert tnice.reconstruction_error(gauged, tnice.canonical_omega_seed(gauged)) < 1e-6


def test_tnice_canonical_dual_equals_shadows_on_the_same_shots():
    """
    Regression test: with no fit (``n_sweeps=0``) and no held-out split
    (``test_fraction=1.0``), the ``"tnice"`` estimator is the canonical dual
    on every shot, so it must equal ``"shadows"`` on the same frequencies --
    here at a cut whose tail MPS genuinely carries complex gauge phases.
    """
    from mpstab.evolutors.hsynthsmpo import HSynthSMPO
    from mpstab.models.ansatze import FloquetAnsatz
    from mpstab.quantum_hardware.backend import QiboSimulator, execute_plan
    from mpstab.quantum_hardware.estimate import estimate_shadows, estimate_tnice

    ansatz = FloquetAnsatz(nqubits=4, nlayers=1, target_qubit=1, decompose_rzz=True)
    hs = HSynthSMPO.rotations_only(ansatz)
    hs.max_bond_dimension = 16
    cut = len(hs.magic_gates) // 2
    observable = "IZII"

    tail, _ = hs.tail_operator(observable, cut)
    assert (
        max(np.abs(site.imag).max() for site in mpo_to_pauli_mps(tail)) > 1e-3
    ), "this cut no longer exercises a complex gauge; pick another one"

    plan = hs._shadow_plan(
        terms={observable: 1.0},
        constant=0.0,
        cut_index=cut,
        resynthesis=hs.resynthesize_head(cut),
        max_bond_dimension=16,
        n_shots=300,
        epsilon=None,
        seed=3,
        tail_handling="append",
    )
    frequencies = execute_plan(QiboSimulator(), plan)

    shadows = estimate_shadows(plan, frequencies)
    canonical = estimate_tnice(plan, frequencies, n_sweeps=0, test_fraction=1.0)
    assert canonical.value == pytest.approx(shadows.value, abs=1e-10)


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
