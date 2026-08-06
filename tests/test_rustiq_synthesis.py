"""
rustiq resynthesis (`mpstab.quantum_hardware.synthesis`).

Covers:
  - target == residual . head at the unitary level (dense, small n);
  - residual tableaus are reproducible across repeated calls, while gate lists
    need not be (rustiq's fix_clifford=True draws randomly);
  - head_counts_only agrees with the (expensive) exact head;
  - resynthesizing the whole chain and folding the residual into the observable
    reproduces direct qibo simulation.
"""

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

pytest.importorskip("rustiq")

from mpstab.engines import StimEngine
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient
from mpstab.pauli import PAULI_MATRICES as _PAULI_MATS
from mpstab.quantum_hardware import (
    build_head_and_residual,
    head_counts_only,
    head_to_qibo_circuit,
)

set_backend("numpy")


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
    dim = 2**n
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


def test_folding_an_identity_residual_leaves_the_observable_alone():
    import stim

    identity = stim.Tableau(3)
    engine = StimEngine()

    folded, sign = engine.fold_pauli_through_tableau("XYZ", identity, sign=1.0)
    assert folded == "XYZ"
    assert sign == 1.0

    folded, sign = engine.fold_pauli_through_tableau("XYZ", identity, sign=-1.0)
    assert folded == "XYZ"
    assert sign == -1.0


def _expectation_via_full_resynthesis(hs, observable):
    """
    The whole resynthesis path end to end, exactly, with no sampling anywhere.

    Two folds put the observable in the head circuit's frame. The head realises
    only the dressed-rotation chain R, so the observable is first backpropagated
    through the circuit's Clifford part (giving M, as
    :meth:`HSMPO.expectation` does). Then, since ``R = U_residual . U_head``,
    ``R^dag M R = U_head^dag (U_residual^dag M U_residual) U_head``, so M is
    folded through the residual before being evaluated on ``U_head|0>``.
    """
    backpropagated, clifford_sign = hs.stab_engine.backpropagate(
        observable=observable, clifford_circuit=hs.clifford_circuit
    )
    resynthesis = hs.resynthesize_head(cut_index=len(hs.magic_gates))
    folded, residual_sign = hs.stab_engine.fold_pauli_through_tableau(
        backpropagated, resynthesis.tail_tableau, 1.0
    )
    state = resynthesis.circuit().state()
    expval = np.real(np.conj(state) @ _pauli_matrix(folded) @ state)
    return clifford_sign * residual_sign * expval


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
def test_full_resynthesis_and_fold_matches_direct_simulation(observable):
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO.rotations_only(ansatz)

    expval = _expectation_via_full_resynthesis(hs, observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_full_resynthesis_and_fold_on_library_ansatz():
    ansatz = HardwareEfficient(nqubits=5, nlayers=2)
    hs = HSynthSMPO.rotations_only(ansatz)
    observable = "Z" * 5

    expval = _expectation_via_full_resynthesis(hs, observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_resynthesize_head_gate_counts_across_partial_cuts():
    ansatz = CircuitAnsatz(qibo_circuit=_small_entangled_circuit(4))
    hs = HSynthSMPO.rotations_only(ansatz)
    n_dressed = len(hs.magic_gates)

    for cut in (0, n_dressed // 2, n_dressed):
        resynthesis = hs.resynthesize_head(cut)
        assert resynthesis.cut_index == cut
        assert resynthesis.method == "rustiq"
        assert isinstance(resynthesis.circuit, Circuit)
        assert resynthesis.circuit.nqubits == hs.nqubits
        assert 0 <= resynthesis.n_two_qubit_gates <= resynthesis.n_gates
        # One rotation is placed per dressed Pauli in the head.
        assert resynthesis.n_gates >= cut


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
