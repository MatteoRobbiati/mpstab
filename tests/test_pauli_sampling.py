"""
Layered correctness checks for `mpstab.quantum_hardware.pauli_expansion`, the
Pauli-sampling half of the head/tail measurement split (the `"pauli"` route of
`HSynthSMPO.expectation_at_cut`).

Each test targets one layer in isolation so a failure localizes: coefficient
extraction first (no randomness at all), then the perfect sampler against
that same extraction, then QWC grouping as pure combinatorics, then the
retained-weight bias bound that ties truncation back to the reported
diagnostic. `test_expectation_methods.py` covers the next layer up (the
`HSynthSMPO` integration and cross-method agreement).
"""

from itertools import product

import numpy as np
import pytest
from qibo import set_backend

from mpstab.engines import QuimbEngine
from mpstab.pauli import PAULI_ARRAY as PAULIS
from mpstab.pauli import PAULI_LABELS as LABELS
from mpstab.quantum_hardware.pauli_expansion import (
    enumerate_pauli_coefficients,
    mpo_to_pauli_mps,
    sample_pauli_strings,
    shadow_variance_from_mpo,
    top_k_pauli_strings,
    truncation_error_estimate,
)
from mpstab.quantum_hardware.plan import build_measurement_plan, group_qwc

set_backend("numpy")


def _folded_mpo(base: str, generators, angles, max_bond=None):
    """A tail-folded MPO with nontrivial bond dimension, no HSynthSMPO needed."""
    engine = QuimbEngine()
    mpo = engine.pauli_mpo(base)
    for generator, angle in zip(generators, angles):
        mpo = engine.conjugate_operator(mpo, generator, angle, max_bond)
    return mpo


def _dense_pauli_matrix(label: str) -> np.ndarray:
    matrix = PAULIS[LABELS.index(label[0])]
    for char in label[1:]:
        matrix = np.kron(matrix, PAULIS[LABELS.index(char)])
    return matrix


@pytest.fixture
def scrambled_mpo():
    """A 4-qubit operator scrambled enough to have several sizeable Pauli terms."""
    rng = np.random.default_rng(3)
    generators = ["YYII", "IXZY", "ZIXY", "YZYX"]
    angles = [rng.uniform(0.3, 1.5) for _ in generators]
    return _folded_mpo("ZZZI", generators, angles)


# ---------------------------------------------------------------------------
# 1. Coefficient extraction vs brute-force dense trace
# ---------------------------------------------------------------------------


def test_coefficients_match_dense_trace(scrambled_mpo):
    dense = scrambled_mpo.to_dense()
    nqubits = 4
    for label, coefficient in enumerate_pauli_coefficients(scrambled_mpo):
        reference = np.trace(dense @ _dense_pauli_matrix(label)) / 2**nqubits
        assert (
            abs(coefficient - reference) < 1e-14
        ), f"[{label}] mps={coefficient} brute-force={reference}"


def test_coefficients_sum_of_squares_is_one_for_pauli_observable():
    # Conjugating a single Pauli string preserves the Frobenius norm exactly,
    # so sum|c_P|^2 must equal 1 (||P||_F^2 / 2^n = 1) to machine precision.
    engine = QuimbEngine()
    mpo = engine.pauli_mpo("XZXI")
    rng = np.random.default_rng(1)
    for generator in ["YYII", "IXZY", "ZIXY", "YZYX", "XYZI", "IIXY"]:
        mpo = engine.conjugate_operator(mpo, generator, rng.uniform(0.3, 1.5), None)

    tensors = mpo_to_pauli_mps(mpo)
    total = sum(
        abs(coefficient) ** 2 for _, coefficient in enumerate_pauli_coefficients(mpo)
    )
    assert total == pytest.approx(1.0, abs=1e-12)
    assert len(tensors) == 4


# ---------------------------------------------------------------------------
# 2. Perfect sampling vs exact marginals and vs the deterministic top-k path
# ---------------------------------------------------------------------------


def test_sampler_frequencies_match_exact_marginals(scrambled_mpo):
    exact = enumerate_pauli_coefficients(scrambled_mpo)
    total_weight = sum(abs(c) ** 2 for _, c in exact)
    exact_probabilities = {label: abs(c) ** 2 / total_weight for label, c in exact}

    rng = np.random.default_rng(0)
    counts: dict = {}
    n_draws = 20000
    from mpstab.quantum_hardware.pauli_expansion import (
        _draw_one_string,
        _right_environments,
    )

    tensors = mpo_to_pauli_mps(scrambled_mpo)
    envs = _right_environments(tensors)
    for _ in range(n_draws):
        label = _draw_one_string(tensors, envs, rng)
        counts[label] = counts.get(label, 0) + 1

    # Only the handful of strings with non-negligible weight are checked:
    # the rest have exact probability ~0 and would need a huge sample to
    # resolve, which is not what this test is for.
    for label, exact_p in exact_probabilities.items():
        if exact_p < 1e-3:
            continue
        empirical_p = counts.get(label, 0) / n_draws
        assert (
            abs(empirical_p - exact_p) < 0.02
        ), f"[{label}] exact={exact_p:.4f} empirical={empirical_p:.4f}"


def test_sampler_recovers_top_k_strings(scrambled_mpo):
    top = top_k_pauli_strings(scrambled_mpo, k=4)
    ensemble = sample_pauli_strings(scrambled_mpo, n_samples=5000, seed=42)

    assert set(top.strings) <= set(ensemble.strings)
    for string, coefficient in zip(top.strings, top.coefficients):
        recovered = dict(zip(ensemble.strings, ensemble.coefficients))[string]
        assert recovered == pytest.approx(coefficient, abs=1e-12)


def test_sample_pauli_strings_deduplicates_and_reports_retained_weight(scrambled_mpo):
    ensemble = sample_pauli_strings(scrambled_mpo, n_samples=3000, seed=7)
    assert len(ensemble.strings) == len(set(ensemble.strings))
    assert 0.0 < ensemble.retained_weight <= 1.0 + 1e-9
    assert ensemble.total_weight == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# 3. QWC grouping
# ---------------------------------------------------------------------------


def _is_qwc(a: str, b: str) -> bool:
    return all(x == "I" or y == "I" or x == y for x, y in zip(a, b))


@pytest.mark.parametrize(
    "strings",
    [
        ["XZXI", "IIYZ", "XYIX", "XYXZ", "YZIY"],
        ["ZZZZ", "ZZII", "IIZZ", "XXXX", "IXXI"],
        ["".join(labels) for labels in product("IXYZ", repeat=3)],
    ],
)
def test_qwc_groups_are_genuinely_qwc(strings):
    groups = group_qwc(strings)
    for group in groups:
        for member in group.members:
            assert _is_qwc(group.setting, member), (group.setting, member)
        # Every pair of members within a group must also be pairwise QWC,
        # not just individually QWC with the (possibly more specific) merged
        # setting.
        for i, a in enumerate(group.members):
            for b in group.members[i + 1 :]:
                assert _is_qwc(a, b), (a, b)


@pytest.mark.parametrize(
    "strings",
    [
        ["XZXI", "IIYZ", "XYIX", "XYXZ", "YZIY"],
        ["ZZZZ", "ZZII", "IIZZ", "XXXX", "IXXI", "ZZZZ"],
    ],
)
def test_qwc_grouping_covers_every_input_string(strings):
    groups = group_qwc(strings)
    covered = [member for group in groups for member in group.members]
    assert sorted(covered) == sorted(strings)


def test_qwc_setting_has_no_leftover_identity():
    groups = group_qwc(["XIII", "IIIX"])
    for group in groups:
        assert "I" not in group.setting


def test_measurement_plan_diagnostics(scrambled_mpo):
    ensemble = sample_pauli_strings(scrambled_mpo, n_samples=3000, seed=11)
    coefficients = dict(zip(ensemble.strings, ensemble.coefficients))
    plan = build_measurement_plan(coefficients)

    assert plan.n_settings == len(plan.groups)
    assert plan.l1_norm == pytest.approx(sum(abs(c) for c in coefficients.values()))
    assert plan.max_weight == max(sum(l != "I" for l in s) for s in coefficients)

    # shots_upper_bound keeps the old (worst-case, perfectly-correlated) scaling.
    assert plan.shots_upper_bound(1e-2) == pytest.approx(plan.l1_norm**2 / 1e-2**2)

    # shots_for_precision now uses the Neyman-optimal, per-group-variance
    # scaling instead, which is never larger than the worst case.
    expected = sum(np.sqrt(v) for v in plan.group_variances) ** 2 / 1e-2**2
    assert plan.shots_for_precision(1e-2) == pytest.approx(expected)
    assert plan.shots_for_precision(1e-2) <= plan.shots_upper_bound(1e-2) + 1e-6


# ---------------------------------------------------------------------------
# 4. Truncation-error estimate (A1) and the exact shadow-variance predictor (A3)
# ---------------------------------------------------------------------------


def test_truncation_error_estimate_matches_exact_l2_and_bounds_zero_at_full_retention(
    scrambled_mpo,
):
    # scrambled_mpo happens to have only 5 Pauli terms with non-negligible
    # weight (the rest are ~1e-30, floating-point noise), so truncating to
    # the top 2 discards a genuine chunk of the Pauli mass, unlike sampling
    # enough strings to reach ~full retention, which would leave only
    # floating-point noise to compare an L1/L2 ordering against.
    ensemble = top_k_pauli_strings(scrambled_mpo, k=2)
    assert ensemble.retained_weight < 0.9

    l1, l2 = truncation_error_estimate(ensemble)
    discarded_mass = ensemble.total_weight * (1.0 - ensemble.retained_weight)
    assert l2 == pytest.approx(np.sqrt(max(0.0, discarded_mass)), abs=1e-10)
    assert l1 >= l2 - 1e-9  # L1 mass is never smaller than L2 mass for >=1 term


def test_truncation_error_estimate_is_zero_when_nothing_is_discarded(scrambled_mpo):
    ensemble = top_k_pauli_strings(scrambled_mpo, k=4**4)
    assert ensemble.retained_weight == pytest.approx(1.0, abs=1e-8)
    l1, l2 = truncation_error_estimate(ensemble)
    assert l1 == pytest.approx(0.0, abs=1e-6)
    assert l2 == pytest.approx(0.0, abs=1e-6)


def test_shadow_variance_from_mpo_matches_brute_force_enumeration(scrambled_mpo):
    assert scrambled_mpo.max_bond() > 1

    enumerated = sum(
        abs(coefficient) ** 2 * 3 ** sum(label != "I" for label in string)
        for string, coefficient in enumerate_pauli_coefficients(scrambled_mpo)
    )
    from_mpo = shadow_variance_from_mpo(scrambled_mpo)
    assert from_mpo == pytest.approx(enumerated, abs=1e-10)
