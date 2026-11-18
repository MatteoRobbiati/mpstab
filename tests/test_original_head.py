"""
Checks for the original head (`HQSMPO.original_head`, `head="original"`): the
original circuit up to the cut plus its residual prepares the head's dressed
rotations, with or without absorption and light-cone pruning; the expectation
value through the split matches the statevector; and the Clifford gates map to
the right tableaux.
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from utils import dense_expectation, rotation_unitary

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.models.ansatze import HardwareEfficient, ParametricKickedIsing
from mpstab.qpu.original_head import clifford_gate_names
from mpstab.qpu.synthesis.utils import head_to_qibo_circuit

HALF_PI = 0.5 * np.pi


def _kicked_ising():
    return ParametricKickedIsing(
        nqubits=5, n_steps=3, delta_h=0.3, delta_j=0.2, delta_z=0.15
    )


def _state(circuit, initial=None):
    return np.asarray(circuit.copy()(initial_state=initial).state())


def _head_and_residual_state(head, nqubits):
    """``residual . head |0>`` as a statevector."""
    circuit = Circuit(nqubits)
    for gate in head.circuit.queue:
        circuit.add(gate)
    circuit += head_to_qibo_circuit(list(head.residual_gates), nqubits)
    return _state(circuit)


def _rotations_state(rotations, nqubits):
    """The dressed ``rotations`` applied to ``|0>``."""
    psi = np.zeros(2**nqubits, dtype=complex)
    psi[0] = 1.0
    for generator, angle in rotations:
        psi = rotation_unitary(generator, angle) @ psi
    return psi


def _overlap(a, b):
    return abs(np.vdot(a, b))


@pytest.mark.parametrize("absorb", [True, False])
@pytest.mark.parametrize("cut_index", [0, 1, 4, 9, 20, 29])
def test_head_and_residual_prepare_the_dressed_rotations(cut_index, absorb):
    hs = HQSMPO.rotations_only(_kicked_ising())
    head = hs.original_head(cut_index, absorb=absorb)
    expected = _rotations_state(hs._dressed_rotations()[:cut_index], hs.nqubits)
    assert _overlap(_head_and_residual_state(head, hs.nqubits), expected) == (
        pytest.approx(1.0, abs=1e-10)
    )
    assert head.method == "original" and head.cut_index == cut_index


@pytest.mark.parametrize(
    "observable", ["IIZZI", "XIIII", {"ZZIII": 0.7, "IIIXZ": -1.3}]
)
@pytest.mark.parametrize("cut_index", [6, 15, 25])
def test_light_cone_head_keeps_only_relevant_rotations(observable, cut_index):
    hs = HQSMPO.rotations_only(_kicked_ising())
    head = hs.original_head(cut_index, observable=observable)
    dressed = hs._dressed_rotations()
    kept = [dressed[k] for k in hs.relevant_rotations(observable) if k < cut_index]
    expected = _rotations_state(kept, hs.nqubits)
    assert _overlap(_head_and_residual_state(head, hs.nqubits), expected) == (
        pytest.approx(1.0, abs=1e-10)
    )


@pytest.mark.parametrize("cut_index", [5, 15, 25])
def test_split_through_the_original_head_matches_statevector(cut_index):
    """Head + residual, then the tail's dressed rotations and the whole Clifford
    part: the original circuit's expectation values, with the head pruned."""
    ansatz = _kicked_ising()
    hs = HQSMPO.rotations_only(ansatz)
    observable = {"IIZZI": 1.0, "ZIIIX": 0.5}
    head = hs.original_head(cut_index, observable=observable)
    psi = _head_and_residual_state(head, hs.nqubits)
    for generator, angle in hs._dressed_rotations()[cut_index:]:
        psi = rotation_unitary(generator, angle) @ psi
    psi = _state(hs.clifford_circuit, psi)
    reference = _state(ansatz.circuit)
    for pauli in observable:
        assert dense_expectation(psi, pauli) == pytest.approx(
            dense_expectation(reference, pauli), abs=1e-10
        )


def test_absorption_moves_trailing_cliffords_off_the_device():
    hs = HQSMPO.rotations_only(_kicked_ising())
    full = hs.original_head(20, absorb=False)
    absorbed = hs.original_head(20, absorb=True)
    assert absorbed.n_gates < full.n_gates
    assert absorbed.n_two_qubit_gates <= full.n_two_qubit_gates
    assert len(absorbed.residual_gates) > len(full.residual_gates)
    # every magic gate of the prefix stays on the device
    magic = lambda head: sum(not gate.clifford for gate in head.circuit.queue)
    assert magic(absorbed) == magic(full) == 20


def test_selected_by_head_and_through_plan():
    hs = HQSMPO(ansatz=HardwareEfficient(nqubits=4, nlayers=2), head="original")
    cut_index = 3
    head = hs.head_at(cut_index)
    assert isinstance(head.circuit, Circuit) and head.method == "original"
    plan = hs.plan("ZZII", cut_index, n_shots=200, seed=0)
    prefix = [type(g).__name__ for g in head.circuit.queue]
    for circuit in plan.circuits:
        assert [type(g).__name__ for g in circuit.queue[: len(prefix)]] == prefix


@pytest.mark.parametrize("cut_index", [6, 15])
def test_shadows_through_the_original_head_match_statevector(cut_index):
    """End to end on the default simulator: head + appended residual + shadow
    bases, estimated against the exact value within five standard errors."""
    ansatz = _kicked_ising()
    hs = HQSMPO(ansatz=ansatz, head="original", light_cone=True)
    observable = "IIZZI"
    result = hs.expectation_at_cut(
        observable, cut_index, method="shadows", n_shots=20000, seed=1
    )
    exact = dense_expectation(_state(ansatz.circuit), observable)
    assert abs(result.value - exact) < 5 * result.stderr + 1e-9


def test_head_does_not_share_the_ansatz_gates():
    ansatz = _kicked_ising()
    hs = HQSMPO.rotations_only(ansatz)
    head = hs.original_head(10)
    originals = {id(gate) for gate in ansatz.circuit.queue}
    assert not any(id(gate) in originals for gate in head.circuit.queue)


def test_default_head_is_the_synthesis_and_unknown_heads_are_refused():
    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    assert HQSMPO(ansatz=ansatz, head_synthesis="extraction").head_at(3).method == (
        "extraction"
    )
    absorbed = HQSMPO(ansatz=ansatz, head="original").head_at(3)
    unabsorbed = HQSMPO(ansatz=ansatz, head="original", absorb=False).head_at(3)
    assert absorbed.n_gates <= unabsorbed.n_gates
    with pytest.raises(ValueError):
        HQSMPO(ansatz=ansatz, head="magic").head_at(3)


_CLIFFORDS = [
    gates.H(0),
    gates.S(0),
    gates.SDG(0),
    gates.SX(0),
    gates.SXDG(0),
    gates.X(0),
    gates.Y(0),
    gates.Z(0),
    gates.CNOT(0, 1),
    gates.CNOT(1, 0),
    gates.CZ(0, 1),
] + [
    rotation(0, theta=k * HALF_PI)
    for rotation in (gates.RX, gates.RY, gates.RZ)
    for k in range(-3, 5)
]


@pytest.mark.parametrize("gate", _CLIFFORDS, ids=lambda g: f"{g.name}{g.parameters}")
def test_clifford_gate_names_match_up_to_phase(gate):
    expected = Circuit(2)
    expected.add(gate)
    names = head_to_qibo_circuit(clifford_gate_names(gate), 2)
    a, b = expected.unitary(), (names.unitary() if names.queue else np.eye(4))
    phase = np.vdot(a.ravel(), b.ravel())
    assert abs(phase) == pytest.approx(4.0)
    np.testing.assert_allclose(b, a * phase / abs(phase), atol=1e-10)


def test_non_clifford_gate_is_refused():
    with pytest.raises(ValueError):
        clifford_gate_names(gates.RX(0, theta=0.3))
