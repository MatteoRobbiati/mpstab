"""
rustiq resynthesis (`mpstab.qpu.synthesis`).

Covers:
  - target == residual . head at the unitary level (dense, small n);
  - residual tableaus are reproducible across repeated calls, while gate lists
    need not be (rustiq's fix_clifford=True draws randomly);
  - resynthesizing the whole chain and folding the residual into the observable
    reproduces direct qibo simulation.
"""

import numpy as np
import pytest
from qibo import Circuit
from utils import (
    expectation_with_qibo,
    pauli_matrix,
    rotation_unitary,
    small_entangled_circuit,
)

pytest.importorskip("rustiq")

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient
from mpstab.pauli import conjugate
from mpstab.qpu.synthesis import RustiqSynthesis, head_to_qibo_circuit
from mpstab.qpu.synthesis import rustiq as rustiq_synthesis


def _target_unitary(paulis, angles):
    dim = 2 ** len(paulis[0])
    U = np.eye(dim, dtype=complex)
    for p, a in zip(paulis, angles):
        U = rotation_unitary(p, a) @ U
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


def _dressed_paulis_and_angles(n=4):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(n))
    hs = HQSMPO.rotations_only(ansatz)
    dressed = hs._dressed_rotations()
    paulis = [p for p, _ in dressed]
    angles = [a for _, a in dressed]
    return hs, paulis, angles


def test_target_equals_tail_times_head_dense():
    n = 4
    _, paulis, angles = _dressed_paulis_and_angles(n)
    assert len(paulis) > 3  # non-trivial rotation chain

    head, residual_tableau, residual_gates = RustiqSynthesis().synthesize(
        paulis, angles
    )

    U_target = _target_unitary(paulis, angles)
    U_head = _unitary_from_gate_list(head, n)
    U_tail = _unitary_from_gate_list(residual_gates, n)

    assert _equal_up_to_global_phase(U_target, U_tail @ U_head)


def test_residual_tableau_reproducible_across_calls_gate_list_not_necessarily():
    _, paulis, angles = _dressed_paulis_and_angles(4)

    _, residual_tableau_1, residual_gates_1 = RustiqSynthesis().synthesize(
        paulis, angles
    )
    _, residual_tableau_2, residual_gates_2 = RustiqSynthesis().synthesize(
        paulis, angles
    )

    # The Clifford OPERATOR is unique -- compare tableaus, not gate lists.
    assert residual_tableau_1 == residual_tableau_2


def test_build_head_and_residual_empty_input():
    head, residual_tableau, residual_gates = RustiqSynthesis().synthesize([], [])
    assert head == []
    assert residual_gates == []
    assert len(residual_tableau) == 0


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
    folded, residual_sign = conjugate(
        backpropagated, resynthesis.residual_tableau.inverse()
    )
    state = resynthesis.circuit().state()
    expval = np.real(np.conj(state) @ pauli_matrix(folded) @ state)
    return clifford_sign * residual_sign * expval


@pytest.mark.parametrize("observable", ["ZZZZ", "XIII", "IZIX", "YYYY"])
def test_full_resynthesis_and_fold_matches_direct_simulation(observable):
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO.rotations_only(ansatz)

    expval = _expectation_via_full_resynthesis(hs, observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_full_resynthesis_and_fold_on_library_ansatz():
    ansatz = HardwareEfficient(nqubits=5, nlayers=2)
    hs = HQSMPO.rotations_only(ansatz)
    observable = "Z" * 5

    expval = _expectation_via_full_resynthesis(hs, observable)
    reference = expectation_with_qibo(mpstab_ansatz=ansatz, observable_str=observable)

    assert np.allclose(expval, reference, atol=1e-6)


def test_resynthesize_head_gate_counts_across_partial_cuts():
    ansatz = CircuitAnsatz(qibo_circuit=small_entangled_circuit(4))
    hs = HQSMPO.rotations_only(ansatz)
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
    # What the module-level guarded import leaves behind without rustiq.
    monkeypatch.setattr(rustiq_synthesis, "pauli_network_synthesis", None)

    with pytest.raises(ImportError, match="smartiel/rustiq.git"):
        RustiqSynthesis().synthesize(["X"], [0.1])
