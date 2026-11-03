import numpy as np
import pytest
import stim
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hsynthsmpo import HSynthSMPO, _assert_real_coefficients
from mpstab.models.ansatze import (
    QAE,
    QFT,
    QPE,
    CircuitAnsatz,
    Grover,
    HardwareEfficient,
    QFTPhaseKernel,
    TrotterIsing,
)
from mpstab.quantum_hardware.plan import (
    allocate_shots_by_variance,
    build_measurement_plan,
)
from mpstab.quantum_hardware.synthesis import build_cnot_ladder_head_and_residual

set_backend("numpy")
set_rng_seed()


def _small_entangled_circuit(n=4):
    """H layer + CZ chain + RY layer: yields multi-qubit dressed rotations."""
    set_rng_seed()
    circ = Circuit(n)
    for q in range(n):
        circ.add(gates.H(q))
    for q in range(n - 1):
        circ.add(gates.CZ(q, q + 1))
    for q in range(n):
        circ.add(gates.RY(q, theta=float(np.random.uniform(0.1, 1.2))))
    return circ


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
@pytest.mark.parametrize("cut_index", [0, 2, 4])
def test_simulation_matches_qibo(observable, cut_index):
    circ = _small_entangled_circuit(4)
    ansatz = CircuitAnsatz(qibo_circuit=circ)
    hs = HSynthSMPO(ansatz)

    simulated = hs.expectation_at_cut(observable, cut_index, method="simulation").value
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(simulated, reference, atol=1e-6), (
        f"[obs={observable}, cut={cut_index}] simulated={simulated:+.6f} "
        f"reference={reference:+.6f}"
    )


@pytest.mark.parametrize("cut_index", [0, 3, 6])
def test_simulation_matches_hsmpo_expectation(cut_index):
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    hs = HSynthSMPO(ansatz)
    observable = "ZIZI"

    base = hs.expectation(observable)
    simulated = hs.expectation_at_cut(observable, cut_index, method="simulation").value

    assert np.allclose(base, simulated, atol=1e-6)


def test_simulation_result_fields():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    result = hs.expectation_at_cut("ZZZZ", 2, method="simulation")
    assert np.isreal(result.value)
    assert 0.0 <= result.state_fidelity <= 1.0 + 1e-6
    assert result.stderr == 0.0
    assert result.n_shots == result.n_settings == 0
    # Untruncated tail: no truncation budget.
    assert result.truncation_l2 == 0.0


def test_simulation_reports_tail_truncation():
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HSynthSMPO(ansatz)
    result = hs.expectation_at_cut(
        "Z" * 6, 0, method="simulation", max_bond_dimension=1
    )
    assert result.truncation_l2 > 1e-6
    assert result.total_error == pytest.approx(result.truncation_l2)


def test_simulation_matches_hamiltonian_sum():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    terms = {"ZZZZ": 0.5, "XIII": -1.5}
    total = hs.expectation_at_cut(terms, 2, method="simulation").value
    expected = sum(
        coefficient * hs.expectation_at_cut(pauli, 2, method="simulation").value
        for pauli, coefficient in terms.items()
    )
    assert total == pytest.approx(expected, abs=1e-10)


def test_simulation_rejects_shot_budget():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation", n_shots=100)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation", epsilon=0.01)


def test_native_engine_not_supported():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    hs.set_engines(tn_engine=NativeTensorNetworkEngine())
    with pytest.raises(NotImplementedError):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation")


def test_tail_truncation_lossless_when_untruncated():
    # No bond cap => the folded tail MPO is exact => zero error.
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)  # max_bond_dimension defaults to None
    info = hs.tail_truncation("ZZZZ", cut_index=0, with_expval=True)
    assert info.relative_frobenius_error == pytest.approx(0.0, abs=1e-7)
    assert info.expval_abs_error == pytest.approx(0.0, abs=1e-7)


def test_tail_truncation_skips_expval_by_default():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    info = hs.tail_truncation("ZZZZ", cut_index=0)
    assert info.relative_frobenius_error is not None
    assert info.expval_abs_error is None
    with pytest.raises(ValueError):
        hs.tail_truncation("ZZZZ", cut_index=0, exact=False, with_expval=True)


def test_tail_truncation_max_bond_dimension_overrides_default():
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HSynthSMPO(ansatz)  # untruncated by default
    assert hs.tail_truncation("Z" * 6, cut_index=0).fidelity_estimate == (
        pytest.approx(1.0)
    )
    info = hs.tail_truncation("Z" * 6, cut_index=0, max_bond_dimension=1)
    assert info.fidelity_estimate < 1.0 - 1e-6
    assert info.relative_frobenius_error > 1e-6


def test_tail_truncation_empty_tail():
    # cut_index == number of dressed rotations => empty tail => exact operator.
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz, max_bond_dimension=2)
    n_dressed = len(hs.magic_gates)
    info = hs.tail_truncation("ZZZZ", cut_index=n_dressed)
    assert info.relative_frobenius_error == pytest.approx(0.0, abs=1e-10)


def test_tail_truncation_reports_error_when_truncated():
    # Aggressive bond cap on a full tail => nonzero operator error; the exact
    # (untruncated) tail fold still reproduces the true expectation, since with
    # cut_index=0 the state is exactly |0...0>.
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HSynthSMPO(ansatz, max_bond_dimension=1)
    info = hs.tail_truncation(
        "Z" * 6, cut_index=0, reference_max_bond=None, with_expval=True
    )
    assert info.relative_frobenius_error >= 0.0
    assert info.expval_abs_error >= 0.0

    reference_operator, sign = hs.build_post_head_mpo(
        "Z" * 6, cut_index=0, max_bond_dimension=None
    )
    state_mps = hs._head_state_mps([])
    reference_expval = (
        np.real(
            hs.tn_engine.expval(state_circuit=state_mps, operator=reference_operator)
        )
        * sign
    )
    exact = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="Z" * 6)
    assert np.allclose(reference_expval, exact, atol=1e-6)


def test_resynthesize_head_is_qibo_circuit():
    pytest.importorskip("rustiq")
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    resynth = hs.resynthesize_head(n_dressed)
    assert isinstance(resynth.circuit, Circuit)
    assert resynth.circuit.nqubits == hs.nqubits
    assert resynth.method == "rustiq"
    assert resynth.cut_index == n_dressed


def test_resynthesize_head_gate_counts_full_cut():
    pytest.importorskip("rustiq")

    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    resynth = hs.resynthesize_head(n_dressed)
    assert resynth.n_gates >= resynth.n_two_qubit_gates >= 0


def test_resynthesize_head_is_cached():
    # Same cut, same rotations: the second call returns the cached head.
    hs = HSynthSMPO(CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4)))
    first = hs.resynthesize_head(2)
    assert hs.resynthesize_head(2) is first
    assert hs.resynthesize_head(3) is not first


def test_resynthesize_head_cache_follows_parameters():
    # New parameters give new dressed angles, hence a new key: no stale head.
    hs = HSynthSMPO(CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4)))
    first = hs.resynthesize_head(2)
    hs.set_parameters([(theta + 0.1,) for (theta,) in hs.get_parameters()])
    assert hs.resynthesize_head(2) is not first


def test_resynthesize_head_cnot_ladder_fallback_has_identity_tail():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)
    dressed = hs._dressed_rotations()

    head, residual_tableau, residual_gates = build_cnot_ladder_head_and_residual(
        [p for p, _ in dressed], [a for _, a in dressed]
    )
    assert residual_gates == []
    assert residual_tableau == stim.Tableau(hs.nqubits)
    assert len(head) > 0
    assert n_dressed > 0


# ---------------------------------------------------------------------------
# Broad coverage over the circuit-library ansatze
# ---------------------------------------------------------------------------
def _observables(n):
    """A spread of Pauli observables of length n, including Y-heavy ones."""
    return [
        "Z" * n,
        "X" * n,
        "Y" * n,  # odd/even #Y -- exercises the expval convention
        ("XYZ" * n)[:n],
        ("IY" * n)[:n],
        "Y" + "I" * (n - 1),  # single Y on qubit 0 (was the bug trigger)
        "I" * (n - 1) + "Y",
    ]


_LIBRARY_ANSATZE = [
    ("QFT", lambda: QFT(nqubits=4)),
    ("QFTPhaseKernel", lambda: QFTPhaseKernel(nqubits=4)),
    ("Grover", lambda: Grover(nqubits=3)),
    ("QPE", lambda: QPE(n_counting=3)),
    ("QAE", lambda: QAE(n_counting=3)),
    ("TrotterIsing", lambda: TrotterIsing(nqubits=4, n_steps=2)),
]


@pytest.mark.parametrize(
    "label,factory", _LIBRARY_ANSATZE, ids=[a[0] for a in _LIBRARY_ANSATZE]
)
def test_library_ansatze_simulation_matches_qibo(label, factory):
    ansatz = factory()
    hs = HSynthSMPO(ansatz)
    n = ansatz.nqubits
    n_dressed = len(hs.magic_gates)
    cuts = sorted({0, 1, n_dressed // 2, n_dressed - 1, n_dressed})

    for observable in _observables(n):
        reference = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        # base HSMPO and every cut must agree with the statevector.
        assert np.allclose(
            hs.expectation(observable), reference, atol=1e-6
        ), f"[{label} base / {observable}]"
        for cut in cuts:
            simulated = hs.expectation_at_cut(observable, cut, method="simulation").value
            assert np.allclose(simulated, reference, atol=1e-6), (
                f"[{label} / {observable} / cut={cut}] "
                f"simulated={simulated:+.6f} reference={reference:+.6f}"
            )


@pytest.mark.parametrize("nqubits", [6, 8, 10])
def test_simulation_matches_qibo_more_qubits(nqubits):
    # Larger systems (statevector still tractable). We sweep cuts with modest
    # tails (cut_index >= n_dressed // 2): with no bond cap those folds are
    # analytically exact and cheap. The opposite extreme (cut=0, the *entire*
    # scrambled observable folded into a single MPO) needs exponentially large
    # bond and is the intended approximation regime -- it is exercised exactly at
    # small n by test_library_ansatze_simulation_matches_qibo instead.
    ansatz = TrotterIsing(nqubits=nqubits, n_steps=2)
    hs = HSynthSMPO(ansatz)
    n_dressed = len(hs.magic_gates)
    cuts = sorted({n_dressed // 2, n_dressed - 3, n_dressed - 1, n_dressed})

    for observable in ["Z" * nqubits, "Y" * nqubits, ("XYZ" * nqubits)[:nqubits]]:
        reference = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        for cut in cuts:
            simulated = hs.expectation_at_cut(observable, cut, method="simulation").value
            assert np.allclose(simulated, reference, atol=1e-6), (
                f"[n={nqubits} / {observable} / cut={cut}] "
                f"simulated={simulated:+.6f} reference={reference:+.6f}"
            )


# ---------------------------------------------------------------------------
# Shot budget exactness (plan.allocate_shots_by_variance)
# ---------------------------------------------------------------------------


def _multi_group_coefficients():
    """
    Pauli strings chosen to QWC-group into several distinct settings (each
    pair below disagrees on a shared qubit, so none of them merge), with
    unequal weights so the Neyman allocation isn't trivially uniform.
    """
    return {"XIII": 3.0, "ZIII": 1.0, "IXII": 2.0, "IZII": 0.5, "IIXI": 1.5}


def test_allocate_shots_by_variance_respects_the_budget_exactly():
    plan = build_measurement_plan(_multi_group_coefficients())
    n_groups = len(plan.groups)
    assert n_groups > 1, "fixture must actually produce multiple QWC groups"

    for n_shots in (n_groups, n_groups + 1, n_groups + 7, 10_000, 10_007):
        shots = allocate_shots_by_variance(plan, n_shots)
        assert sum(shots) == n_shots, (n_shots, shots)
        assert all(s >= 1 for s in shots), shots


def test_allocate_shots_by_variance_raises_below_group_count():
    plan = build_measurement_plan(_multi_group_coefficients())
    n_groups = len(plan.groups)
    with pytest.raises(ValueError):
        allocate_shots_by_variance(plan, n_groups - 1)


def test_expectation_at_cut_pauli_respects_shot_budget():
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    hs = HSynthSMPO(ansatz)
    cut_index = len(hs.magic_gates) // 2

    result = hs.expectation_at_cut(
        "XZIZ", cut_index, method="pauli", n_shots=257, seed=3, n_string_samples=50
    )
    assert result.n_shots == 257


# ---------------------------------------------------------------------------
# _assert_real_coefficients (hsynthsmpo._pauli_plan's Hermiticity guard)
# ---------------------------------------------------------------------------


def test_assert_real_coefficients_passes_through_real_values():
    pooled = {"XI": 1.5, "ZZ": -0.25 + 1e-12j}  # negligible float noise
    real = _assert_real_coefficients(pooled)
    assert real == pytest.approx({"XI": 1.5, "ZZ": -0.25})
    assert all(isinstance(v, float) for v in real.values())


def test_assert_real_coefficients_raises_on_non_negligible_imaginary_part():
    pooled = {"XI": 1.0, "ZZ": 0.5 + 0.3j}
    with pytest.raises(ValueError, match="ZZ"):
        _assert_real_coefficients(pooled)
