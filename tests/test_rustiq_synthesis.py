"""
Foldable low-level-rustiq resynthesis (`mpstab.evolutors.quantum_hardware`).

Covers:
  - target == tail . head at the unitary level (dense, small n);
  - tail tableaus (not gate lists -- rustiq's fix_clifford=True is
    non-deterministic) are reproducible across repeated calls;
  - head_counts_only matches the (expensive) exact head from
    build_head_and_residual;
  - HSynthSMPO.expectation_from_rustiq_fold matches direct qibo simulation.
"""
import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

pytest.importorskip("rustiq")

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.evolutors.quantum_hardware import (
    build_head_and_residual,
    fold_observable,
    head_counts_only,
    head_to_qibo_circuit,
)
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient

set_backend("numpy")

_PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _pauli_matrix(pauli_str):
    mat = _PAULI_MATS[pauli_str[0]]
    for c in pauli_str[1:]:
        mat = np.kron(mat, _PAULI_MATS[c])
    return mat


def _rotation_unitary(pauli_str, angle):
    dim = 2 ** len(pauli_str)
    return np.cos(angle / 2) * np.eye(dim) - 1j * np.sin(angle / 2) * _pauli_matrix(
        pauli_str
    )


def _target_unitary(paulis, angles):
    dim = 2 ** len(paulis[0])
    U = np.eye(dim, dtype=complex)
    for p, a in zip(paulis, angles):
        U = _rotation_unitary(p, a) @ U
    return U


def _unitary_from_gate_list(gate_list, n):
    """Dense unitary of a rustiq-vocabulary gate list, via qibo circuit columns."""
    circuit = head_to_qibo_circuit(gate_list, n)
    dim = 2 ** n
    columns = []
    for k in range(dim):
        basis_state = np.zeros(dim, dtype=complex)
        basis_state[k] = 1.0
        columns.append(circuit(initial_state=basis_state.copy()).state())
    return np.column_stack(columns)


def _equal_up_to_global_phase(A, B, atol=1e-6):
    idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    phase = B[idx] / A[idx]
    return np.allclose(A * phase, B, atol=atol)


def _small_entangled_circuit(n=4):
    set_rng_seed()
    circ = Circuit(n)
    for q in range(n):
        circ.add(gates.H(q))
    for q in range(n - 1):
        circ.add(gates.CZ(q, q + 1))
    for q in range(n):
        circ.add(gates.RY(q, theta=float(np.random.uniform(0.1, 1.2))))
    return circ


def _dressed_paulis_and_angles(n=4):
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(n))
    hs = HSynthSMPO.rotations_only(ansatz)
    dressed = hs._dressed_rotations()
    paulis = [p for p, _ in dressed]
    angles = [a for _, a in dressed]
    return hs, paulis, angles


def test_target_equals_tail_times_head_dense():
    n = 4
    _, paulis, angles = _dressed_paulis_and_angles(n)
    assert len(paulis) > 3  # non-trivial rotation chain

    head, tail_tableau, tail_gates = build_head_and_residual(paulis, angles)

    U_target = _target_unitary(paulis, angles)
    U_head = _unitary_from_gate_list(head, n)
    U_tail = _unitary_from_gate_list(tail_gates, n)

    assert _equal_up_to_global_phase(U_target, U_tail @ U_head)


def test_tail_tableau_reproducible_across_calls_gate_list_not_necessarily():
    _, paulis, angles = _dressed_paulis_and_angles(4)

    _, tail_tableau_1, tail_gates_1 = build_head_and_residual(paulis, angles)
    _, tail_tableau_2, tail_gates_2 = build_head_and_residual(paulis, angles)

    # The Clifford OPERATOR is unique -- compare tableaus, not gate lists.
    assert tail_tableau_1 == tail_tableau_2


def test_head_counts_only_matches_exact_head():
    _, paulis, angles = _dressed_paulis_and_angles(4)

    head, _, _ = build_head_and_residual(paulis, angles)
    exact_total, exact_two_q = len(head), sum(1 for g in head if len(g[1]) == 2)

    cheap_total, cheap_two_q = head_counts_only(paulis)

    assert cheap_total == exact_total
    assert cheap_two_q == exact_two_q


def test_build_head_and_residual_empty_input():
    head, tail_tableau, tail_gates = build_head_and_residual([], [])
    assert head == []
    assert tail_gates == []
    assert len(tail_tableau) == 0


def test_fold_observable_roundtrip_identity_tableau():
    import stim

    identity = stim.Tableau(3)
    folded, sign = fold_observable(identity, "XYZ", sign=1.0)
    assert folded == "XYZ"
    assert sign == 1.0

    folded, sign = fold_observable(identity, "XYZ", sign=-1.0)
    assert folded == "XYZ"
    assert sign == -1.0


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
def test_expectation_from_rustiq_fold_matches_direct_simulation(observable):
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO.rotations_only(ansatz)

    expval = hs.expectation_from_rustiq_fold(observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_expectation_from_rustiq_fold_library_ansatze():
    ansatz = HardwareEfficient(nqubits=5, nlayers=2)
    hs = HSynthSMPO.rotations_only(ansatz)
    observable = "Z" * 5

    expval = hs.expectation_from_rustiq_fold(observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_foldable_head_and_tail_partial_cut_gate_counts():
    # Partial cuts are only claimed to be correct/useful for gate-count
    # profiling (mirrors the hsmpo4transpilation prototype's usage) -- not for
    # combining with an exact expectation value (see expectation_from_rustiq_fold's
    # docstring for why a partial tail can't be folded exactly as a single Pauli).
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO.rotations_only(ansatz)
    n_dressed = len(hs.magic_gates)

    for cut in (0, n_dressed // 2, n_dressed):
        counts = hs.foldable_head_gate_counts(cut)
        assert counts["n_head_rotations"] == cut
        assert counts["n_tail_rotations"] == n_dressed - cut
        assert counts["synthesized_head_total_gates"] >= 0
        assert counts["synthesized_head_2q_gates"] >= 0
        assert counts["synthesized_head_2q_gates"] <= counts["synthesized_head_total_gates"]
        assert counts["original_circuit_2q_gates"] >= 0

    circuit = hs.foldable_head_circuit(n_dressed)
    assert isinstance(circuit, Circuit)
    assert circuit.nqubits == hs.nqubits


def test_rustiq_import_error_has_install_hint(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rustiq":
            raise ImportError("no rustiq here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="mpstab\\[rustiq\\]"):
        build_head_and_residual(["X"], [0.1])
