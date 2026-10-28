import numpy as np
import pytest
import stim
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z
from utils import expectation_with_qibo, small_entangled_circuit

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hqsmpo import HQSMPO, _resynthesize
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
from mpstab.qpu.measurement.pauli import (
    _real_coefficients,
    allocate_shots_by_variance,
    build_measurement_plan,
)
from mpstab.qpu.synthesis import CnotLadderSynthesis


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
@pytest.mark.parametrize("cut_index", [0, 2, 4])
def test_simulation_matches_qibo(observable, cut_index):
    circ = small_entangled_circuit(4)
    ansatz = CircuitAnsatz(qibo_circuit=circ)
    hs = HQSMPO(ansatz)

    simulated = hs.expectation_at_cut(observable, cut_index, method="simulation").value
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(simulated, reference, atol=1e-6), (
        f"[obs={observable}, cut={cut_index}] simulated={simulated:+.6f} "
        f"reference={reference:+.6f}"
    )


@pytest.mark.parametrize("cut_index", [0, 3, 6])
def test_simulation_matches_hsmpo_expectation(cut_index):
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    hs = HQSMPO(ansatz)
    observable = "ZIZI"

    base = hs.expectation(observable)
    simulated = hs.expectation_at_cut(observable, cut_index, method="simulation").value

    assert np.allclose(base, simulated, atol=1e-6)


def test_simulation_result_fields():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    result = hs.expectation_at_cut("ZZZZ", 2, method="simulation")
    assert np.isreal(result.value)
    assert 0.0 <= result.state_fidelity <= 1.0 + 1e-6
    assert result.stderr == 0.0
    assert result.n_shots == result.n_settings == 0
    # Untruncated tail: no truncation budget.
    assert result.truncation_l2 == 0.0


def test_simulation_reports_tail_truncation():
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HQSMPO(ansatz)
    result = hs.expectation_at_cut(
        "Z" * 6, 0, method="simulation", max_bond_dimension=1
    )
    assert result.truncation_l2 > 1e-6
    assert result.total_error == pytest.approx(result.truncation_l2)


def _ising_hamiltonian(n):
    from qibo import hamiltonians, symbols

    return hamiltonians.SymbolicHamiltonian(
        sum(symbols.Z(i) * symbols.Z(i + 1) for i in range(n - 1))
        + 0.7 * sum(symbols.X(i) for i in range(n)),
        nqubits=n,
    )


def test_simulation_reports_svd_cutoff_truncation_without_bond_cap():
    # A long tail of small-angle rotations leaves Schmidt values below the
    # default relative cutoff (1e-10): the fold truncates with no bond cap, and
    # truncation_l2 must say so rather than read 0.
    n = 10
    ansatz = TrotterIsing(nqubits=n, n_steps=6, dt=0.2)
    observable = _ising_hamiltonian(n)
    reference = HQSMPO(ansatz)
    reference.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
    exact = float(np.real(reference.expectation(observable)))

    result = HQSMPO(ansatz, max_bond_dimension=None).expectation_at_cut(
        observable, 0, method="simulation"
    )
    error = abs(result.value - exact)
    assert error > 1e-5  # the cutoff really does truncate here
    assert result.truncation_l2 > 0.0
    assert error / 10 <= result.truncation_l2 <= 10 * error


def test_simulation_exact_fold_reports_no_truncation():
    # No cutoff and no bond cap: the fold is exact up to rounding, and its
    # drift (|1 - F| ~ 1e-15) must stay below the noise floor of
    # fold_truncation_l2 instead of reading as ~1e-7.
    n = 6
    ansatz = TrotterIsing(nqubits=n, n_steps=4, dt=0.2)
    hs = HQSMPO(ansatz, max_bond_dimension=None)
    hs.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
    result = hs.expectation_at_cut(_ising_hamiltonian(n), 0, method="simulation")
    assert result.truncation_l2 == 0.0


def test_pauli_skips_hermiticity_check_when_only_the_cutoff_truncates():
    # An aggressive SVD cutoff with no bond cap leaves an anti-Hermitian part
    # far above the check's tolerance: it is truncation, not a bug upstream,
    # so the plan must project it out rather than raise.
    n = 4
    ansatz = TrotterIsing(nqubits=n, n_steps=3, dt=0.2)
    hs = HQSMPO(ansatz, max_bond_dimension=None)
    hs.set_engines(tn_engine=QuimbEngine(svd_cutoff=1e-2))
    result = hs.expectation_at_cut(
        _ising_hamiltonian(n), 0, method="pauli", n_shots=100, seed=1
    )
    assert result.bond_truncation_l2 > 0.0
    assert result.antihermitian_l2 > 1e-8


def test_simulation_matches_hamiltonian_sum():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    terms = {"ZZZZ": 0.5, "XIII": -1.5}
    total = hs.expectation_at_cut(terms, 2, method="simulation").value
    expected = sum(
        coefficient * hs.expectation_at_cut(pauli, 2, method="simulation").value
        for pauli, coefficient in terms.items()
    )
    assert total == pytest.approx(expected, abs=1e-10)


def test_simulation_rejects_shot_budget():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation", n_shots=100)
    with pytest.raises(ValueError):
        hs.expectation_at_cut("ZZZZ", 2, method="simulation", epsilon=0.01)


def test_tail_truncation_lossless_when_untruncated():
    # No bond cap => the folded tail MPO is exact => zero error.
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)  # max_bond_dimension defaults to None
    info = hs.tail_truncation("ZZZZ", cut_index=0, with_expval=True)
    assert info.relative_frobenius_error == pytest.approx(0.0, abs=1e-7)
    assert info.expval_abs_error == pytest.approx(0.0, abs=1e-7)


def test_tail_truncation_skips_expval_by_default():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    info = hs.tail_truncation("ZZZZ", cut_index=0)
    assert info.relative_frobenius_error is not None
    assert info.expval_abs_error is None
    with pytest.raises(ValueError):
        hs.tail_truncation("ZZZZ", cut_index=0, exact=False, with_expval=True)


def test_tail_truncation_max_bond_dimension_overrides_default():
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HQSMPO(ansatz)  # untruncated by default
    assert hs.tail_truncation("Z" * 6, cut_index=0).fidelity_estimate == (
        pytest.approx(1.0)
    )
    info = hs.tail_truncation("Z" * 6, cut_index=0, max_bond_dimension=1)
    assert info.fidelity_estimate < 1.0 - 1e-6
    assert info.relative_frobenius_error > 1e-6


def test_tail_truncation_empty_tail():
    # cut_index == number of dressed rotations => empty tail => exact operator.
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz, max_bond_dimension=2)
    n_dressed = len(hs.magic_gates)
    info = hs.tail_truncation("ZZZZ", cut_index=n_dressed)
    assert info.relative_frobenius_error == pytest.approx(0.0, abs=1e-10)


def test_tail_truncation_reports_error_when_truncated():
    # Aggressive bond cap on a full tail => nonzero operator error; the exact
    # (untruncated) tail fold still reproduces the true expectation, since with
    # cut_index=0 the state is exactly |0...0>.
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    hs = HQSMPO(ansatz, max_bond_dimension=1)
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


@pytest.mark.parametrize(
    "observable", ["ZZZZZZ", "IXIYIZ", {"ZZIIII": 0.7, "IXIYIZ": -1.2, "XIIIIX": 0.4}]
)
def test_split_scan_matches_simulation(observable):
    # A tight bond cap makes every cut different; the incremental scan must
    # still reproduce the single-cut simulation at each one, all fields.
    hs = HQSMPO(HardwareEfficient(nqubits=6, nlayers=2), max_bond_dimension=2)
    scan = hs.split_scan(observable)
    assert [point.cut_index for point in scan] == list(
        range(len(hs._dressed_rotations()) + 1)
    )

    for point in scan:
        reference = hs.expectation_at_cut(
            observable, point.cut_index, method="simulation"
        )
        assert point.result.value == pytest.approx(reference.value, abs=1e-10)
        assert point.result.state_fidelity == pytest.approx(
            reference.state_fidelity, abs=1e-10
        )
        # sqrt(1 - fidelity) turns ~1e-15 rounding in the fold's norm into
        # ~1e-8, and the scan folds the tail in segments, so it rounds
        # differently from a single fold: compare at that scale.
        assert point.result.truncation_l2 == pytest.approx(
            reference.truncation_l2, abs=1e-6
        )


def test_split_scan_light_cone_matches_simulation():
    # With light_cone the tail skips the rotations each term cannot see; the
    # backward pass must skip the same ones.
    observable = {"ZZIIII": 0.7, "IXIYIZ": -1.2}
    hs = HQSMPO(
        HardwareEfficient(nqubits=6, nlayers=2), max_bond_dimension=2, light_cone=True
    )
    for point in hs.split_scan(observable):
        reference = hs.expectation_at_cut(
            observable, point.cut_index, method="simulation"
        )
        assert point.result.value == pytest.approx(reference.value, abs=1e-10)


def test_split_scan_light_cone_override():
    # light_cone=True on the scan alone matches an instance built with it, and
    # untruncated it changes nothing: the skipped rotations cancel exactly.
    observable = {"ZZIIII": 0.7, "IXIYIZ": -1.2}
    ansatz = HardwareEfficient(nqubits=6, nlayers=2)
    on = HQSMPO(ansatz, max_bond_dimension=2, light_cone=True).split_scan(observable)
    override = HQSMPO(ansatz, max_bond_dimension=2).split_scan(
        observable, light_cone=True
    )
    for a, b in zip(on, override):
        assert a.result.value == pytest.approx(b.result.value, abs=1e-12)

    exact = HQSMPO(ansatz)
    for a, b in zip(
        exact.split_scan(observable, light_cone=True),
        exact.split_scan(observable, light_cone=False),
    ):
        assert a.result.value == pytest.approx(b.result.value, abs=1e-10)
    assert exact.best_split(observable, light_cone=True).cut_index in range(
        len(exact._dressed_rotations()) + 1
    )


def test_split_scan_hamiltonian_constant_and_l1():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    hamiltonian = SymbolicHamiltonian(
        form=2.0 + 0.5 * Z(0) * Z(1) - 1.5 * X(2), nqubits=4
    )
    for point in hs.split_scan(hamiltonian, cuts=[0, 3]):
        reference = hs.expectation_at_cut(
            hamiltonian, point.cut_index, method="simulation"
        )
        assert point.result.value == pytest.approx(reference.value, abs=1e-10)
        # The identity offset carries no truncation, so it is left out.
        assert point.coefficient_l1 == pytest.approx(2.0)


def test_split_scan_untruncated_is_exact():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="IZIX")
    for point in hs.split_scan("IZIX"):
        assert point.result.value == pytest.approx(reference, abs=1e-6)
        assert point.error_estimate == pytest.approx(0.0, abs=1e-6)


def test_split_scan_max_bond_dimension_overrides_default():
    hs = HQSMPO(HardwareEfficient(nqubits=6, nlayers=2))  # untruncated
    scan = hs.split_scan("Z" * 6, cuts=[0], max_bond_dimension=1)
    reference = hs.expectation_at_cut(
        "Z" * 6, 0, method="simulation", max_bond_dimension=1
    )
    assert scan[0].result.truncation_l2 > 1e-6
    assert scan[0].result.truncation_l2 == pytest.approx(reference.truncation_l2)


def test_split_point_error_estimate():
    hs = HQSMPO(HardwareEfficient(nqubits=6, nlayers=2), max_bond_dimension=2)
    for point in hs.split_scan({"ZZIIII": 0.7, "IXIYIZ": -1.2}):
        result = point.result
        expected = result.truncation_l2 + 1.9 * np.sqrt(
            max(0.0, 1.0 - result.state_fidelity)
        )
        assert point.error_estimate == pytest.approx(expected)


def test_split_scan_cuts_sorted_and_deduplicated():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    n_rotations = len(hs._dressed_rotations())
    scan = hs.split_scan("ZZZZ", cuts=[n_rotations, 0, 2, 2])
    assert [point.cut_index for point in scan] == [0, 2, n_rotations]


def test_split_scan_accepts_range_and_numpy_cuts():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    n_rotations = len(hs._dressed_rotations())
    expected = hs.split_scan("ZZZZ", cuts=list(range(0, n_rotations + 1, 2)))
    for cuts in (range(0, n_rotations + 1, 2), np.arange(0, n_rotations + 1, 2)):
        scan = hs.split_scan("ZZZZ", cuts=cuts)
        assert scan == expected
        assert all(type(point.cut_index) is int for point in scan)


def test_split_scan_without_rotations():
    # A Clifford-only circuit has K = 0: the only cut is 0, the whole circuit
    # sits in the Clifford part, and the split is exact.
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.H(q))
    for q in range(3):
        circuit.add(gates.CZ(q, q + 1))
    circuit.add(gates.S(1))
    ansatz = CircuitAnsatz(qibo_circuit=circuit)
    hs = HQSMPO(ansatz)
    assert len(hs._dressed_rotations()) == 0

    scan = hs.split_scan("IZXZ")
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str="IZXZ")
    assert [point.cut_index for point in scan] == [0]
    assert scan[0].result.value == pytest.approx(reference, abs=1e-6)
    assert hs.best_split("IZXZ").cut_index == 0


def test_split_scan_rejects_bad_cuts():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    n_rotations = len(hs._dressed_rotations())
    for bad_cuts in ([0, -1], [0, n_rotations + 1], []):
        with pytest.raises(ValueError):
            hs.split_scan("ZZZZ", cuts=bad_cuts)


def test_split_scan_native_engine_not_supported():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    hs.set_engines(tn_engine=NativeTensorNetworkEngine())
    with pytest.raises(NotImplementedError):
        hs.split_scan("ZZZZ")


def test_best_split_minimises_error_estimate():
    hs = HQSMPO(HardwareEfficient(nqubits=6, nlayers=2), max_bond_dimension=2)
    scan = hs.split_scan("ZZZZZZ")
    best = hs.best_split(scan=scan)
    assert best.error_estimate == min(point.error_estimate for point in scan)
    assert best == hs.best_split("ZZZZZZ")  # same result without a cached scan


def test_best_split_forwards_max_bond_dimension():
    # The object is untruncated, so any tail truncation must come from the
    # forwarded cap: picking the most truncated cut exposes it.
    hs = HQSMPO(HardwareEfficient(nqubits=6, nlayers=2))
    most_truncated = hs.best_split(
        "Z" * 6, key=lambda point: -point.result.truncation_l2, max_bond_dimension=1
    )
    assert most_truncated.result.truncation_l2 > 1e-6


def test_best_split_ties_go_to_shorter_head():
    # A constant key ties every cut, so the shortest head wins.
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    assert hs.best_split("ZZZZ", key=lambda point: 0.0).cut_index == 0
    assert hs.best_split("ZZZZ", cuts=[3, 1, 2], key=lambda point: 0.0).cut_index == 1


def test_best_split_custom_key():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    n_rotations = len(hs._dressed_rotations())
    best = hs.best_split("ZZZZ", key=lambda point: -point.cut_index)
    assert best.cut_index == n_rotations


def test_best_split_needs_observable_or_scan():
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    with pytest.raises(ValueError):
        hs.best_split()
    with pytest.raises(ValueError):
        hs.best_split(scan=[])


def test_resynthesize_head_is_qibo_circuit():
    pytest.importorskip("rustiq")
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    resynth = hs.resynthesize_head(n_dressed)
    assert isinstance(resynth.circuit, Circuit)
    assert resynth.circuit.nqubits == hs.nqubits
    assert resynth.method == "rustiq"
    assert resynth.cut_index == n_dressed


def test_resynthesize_head_gate_counts_full_cut():
    pytest.importorskip("rustiq")

    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    n_dressed = len(hs.magic_gates)

    resynth = hs.resynthesize_head(n_dressed)
    assert resynth.n_gates >= resynth.n_two_qubit_gates >= 0


def test_resynthesize_head_is_cached():
    # Same cut, same rotations: the second call is served by the cache.
    _resynthesize.cache_clear()
    hs = HQSMPO(CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)))
    first = hs.resynthesize_head(2)
    second = hs.resynthesize_head(2)
    assert _resynthesize.cache_info().hits == 1
    assert second.circuit.get_parameters() == first.circuit.get_parameters()
    assert second.residual_gates == first.residual_gates
    hs.resynthesize_head(3)
    assert _resynthesize.cache_info().misses == 2


def test_resynthesize_head_cache_follows_parameters():
    # New parameters give new dressed angles, hence a new key: no stale head.
    _resynthesize.cache_clear()
    hs = HQSMPO(
        CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)),
        head_synthesis="cnot_ladder",
    )
    first = hs.resynthesize_head(2)
    hs.set_parameters([(theta + 0.1,) for (theta,) in hs.get_parameters()])
    second = hs.resynthesize_head(2)
    assert _resynthesize.cache_info().misses == 2
    assert second.circuit.get_parameters() != first.circuit.get_parameters()


def test_mutating_a_plan_circuit_leaves_the_cached_head_intact():
    # Plan circuits are head + basis measurement, and qibo's circuit addition
    # shares gate objects: changing a plan circuit must not reach the cache.
    hs = HQSMPO(
        CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)),
        head_synthesis="cnot_ladder",
    )
    original = hs.resynthesize_head(2).circuit.get_parameters()
    assert original

    plan = hs.plan("ZZZZ", 2, method="pauli", n_shots=200, seed=0, out_folder=False)
    circuit = plan.circuits[0]
    circuit.set_parameters([(0.9,) for _ in circuit.get_parameters()])

    assert hs.resynthesize_head(2).circuit.get_parameters() == original


def test_resynthesize_head_returns_independent_objects():
    hs = HQSMPO(
        CircuitAnsatz(qibo_circuit=small_entangled_circuit(4)),
        head_synthesis="cnot_ladder",
    )
    first, second = hs.resynthesize_head(2), hs.resynthesize_head(2)
    assert first.circuit is not second.circuit
    assert first.residual_tableau is not second.residual_tableau
    assert first.residual_tableau == second.residual_tableau


def test_resynthesize_head_cnot_ladder_fallback_has_identity_tail():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO(ansatz)
    n_dressed = len(hs.magic_gates)
    dressed = hs._dressed_rotations()

    head, residual_tableau, residual_gates = CnotLadderSynthesis().synthesize(
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
    hs = HQSMPO(ansatz)
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
            simulated = hs.expectation_at_cut(
                observable, cut, method="simulation"
            ).value
            assert np.allclose(simulated, reference, atol=1e-6), (
                f"[{label} / {observable} / cut={cut}] "
                f"simulated={simulated:+.6f} reference={reference:+.6f}"
            )


@pytest.mark.parametrize("observable", ["XIX", "YIX"])
def test_long_clifford_t_tail_fold_is_exact(observable):
    # Folding the whole circuit (cut=0) meets exactly degenerate Gram spectra,
    # on which LAPACK's MRRR eigensolver returned non-orthogonal eigenvectors:
    # the fold then lost canonical form and the default SVD cutoff dropped
    # real weight (XIX gave -0.1222 instead of -0.125).
    ansatz = Grover(nqubits=3, marked_state=0, n_iterations=2)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)
    simulated = HQSMPO(ansatz).expectation_at_cut(observable, 0, method="simulation")
    assert np.allclose(simulated.value, reference, atol=1e-6)


@pytest.mark.parametrize("nqubits", [6, 8, 10])
def test_simulation_matches_qibo_more_qubits(nqubits):
    # Larger systems (statevector still tractable). We sweep cuts with modest
    # tails (cut_index >= n_dressed // 2): with no bond cap those folds are
    # analytically exact and cheap. The opposite extreme (cut=0, the *entire*
    # scrambled observable folded into a single MPO) needs exponentially large
    # bond and is the intended approximation regime -- it is exercised exactly at
    # small n by test_library_ansatze_simulation_matches_qibo instead.
    ansatz = TrotterIsing(nqubits=nqubits, n_steps=2)
    hs = HQSMPO(ansatz)
    n_dressed = len(hs.magic_gates)
    cuts = sorted({n_dressed // 2, n_dressed - 3, n_dressed - 1, n_dressed})

    for observable in ["Z" * nqubits, "Y" * nqubits, ("XYZ" * nqubits)[:nqubits]]:
        reference = expectation_with_qibo(
            mpstab_ansatz=ansatz, observable_str=observable
        )
        for cut in cuts:
            simulated = hs.expectation_at_cut(
                observable, cut, method="simulation"
            ).value
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
    hs = HQSMPO(ansatz)
    cut_index = len(hs.magic_gates) // 2

    result = hs.expectation_at_cut(
        "XZIZ", cut_index, method="pauli", n_shots=257, seed=3, n_string_samples=50
    )
    assert result.n_shots == 257


# ---------------------------------------------------------------------------
# _real_coefficients (PauliSampling.build_plan's Hermiticity guard)
# ---------------------------------------------------------------------------


def test_real_coefficients_passes_through_real_values():
    pooled = {"XI": 1.5, "ZZ": -0.25 + 1e-12j}  # negligible float noise
    real = _real_coefficients(pooled)
    assert real == pytest.approx({"XI": 1.5, "ZZ": -0.25})
    assert all(isinstance(v, float) for v in real.values())


def test_real_coefficients_raises_on_non_negligible_imaginary_part():
    pooled = {"XI": 1.0, "ZZ": 0.5 + 0.3j}
    with pytest.raises(ValueError, match="ZZ"):
        _real_coefficients(pooled)


def test_real_coefficients_projects_when_truncated():
    pooled = {"XY": 0.5 - 2e-4j, "ZZ": -0.25 + 0j}
    real = _real_coefficients(pooled, check=False)
    assert real == {"XY": 0.5, "ZZ": -0.25}
