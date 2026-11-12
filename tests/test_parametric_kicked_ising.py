"""
Checks for the `ParametricKickedIsing` ansatz: it is the `KickedIsing` circuit
at the angles `Clifford point + deviation`, only the deviations are magic for
HSMPO, and its expectation values match the statevector.
"""

from functools import reduce

import numpy as np
import pytest
from qibo import set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import KickedIsing, ParametricKickedIsing

set_backend("numpy")

_PAULIS = {
    "I": np.eye(2),
    "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.diag([1, -1]),
}


def _dense_expectation(state, pauli):
    operator = reduce(np.kron, [_PAULIS[p] for p in pauli])
    return float(np.real(np.vdot(state, operator @ state)))


def _state(ansatz):
    return np.asarray(ansatz.circuit.copy()().state())


def _magic_gates(ansatz):
    (magic_gates, _), _ = ansatz.partitionate_circuit(
        replacement_probability=0.0, replacement_method="closest"
    )
    return magic_gates


@pytest.mark.parametrize(
    "deltas",
    [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, -0.3, 0.0), (0.2, -0.3, 0.15)],
    ids=["clifford", "h", "h_j", "h_j_z"],
)
def test_same_state_as_kicked_ising(deltas):
    """Splitting every rotation into Clifford part + deviation does not change the state."""
    delta_h, delta_j, delta_z = deltas
    parametric = ParametricKickedIsing(
        nqubits=5, n_steps=3, delta_h=delta_h, delta_j=delta_j, delta_z=delta_z
    )
    reference = KickedIsing(
        nqubits=5,
        n_steps=3,
        theta_h=parametric.theta_h,
        theta_z=delta_z,
        theta_j=parametric.theta_j,
    )
    overlap = np.vdot(_state(reference), _state(parametric))
    assert abs(overlap) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize(
    "deltas, magic_per_step",
    [
        ((0.0, 0.0, 0.0), lambda n: 0),
        ((0.2, 0.0, 0.0), lambda n: n),
        ((0.2, 0.3, 0.0), lambda n: 2 * n - 1),
        ((0.2, 0.3, 0.1), lambda n: 3 * n - 1),
    ],
    ids=["clifford", "h", "h_j", "h_j_z"],
)
def test_only_the_deviations_are_magic(deltas, magic_per_step):
    nqubits, n_steps = 5, 2
    delta_h, delta_j, delta_z = deltas
    ansatz = ParametricKickedIsing(
        nqubits=nqubits,
        n_steps=n_steps,
        delta_h=delta_h,
        delta_j=delta_j,
        delta_z=delta_z,
    )
    magic = _magic_gates(ansatz)
    assert len(magic) == magic_per_step(nqubits) * n_steps
    # the magic rotations carry the deviations, never the Clifford angles
    angles = {round(float(gate.parameters[0]), 12) for _, gate in magic}
    assert angles <= {delta for delta in deltas if delta != 0}


def test_clifford_point_has_no_parametrised_magic():
    """At delta = 0 the circuit is the Clifford skeleton: kicks and couplings only."""
    nqubits, n_steps = 4, 2
    ansatz = ParametricKickedIsing(nqubits=nqubits, n_steps=n_steps, delta_h=0.0)
    assert all(gate.clifford for gate in ansatz.circuit.queue)
    assert len(ansatz.circuit.queue) == n_steps * (nqubits + 3 * (nqubits - 1))


def test_clifford_angles_must_be_multiples_of_half_pi():
    with pytest.raises(ValueError):
        ParametricKickedIsing(nqubits=4, clifford_h=0.8)
    with pytest.raises(ValueError):
        ParametricKickedIsing(nqubits=4, clifford_j=1.0)
    # other Clifford points are allowed
    ansatz = ParametricKickedIsing(
        nqubits=4, delta_h=0.0, clifford_h=np.pi, clifford_j=0.0
    )
    assert all(gate.clifford for gate in ansatz.circuit.queue)


@pytest.mark.parametrize("pauli", ["ZIIIII", "IIXIII", "IZZIII", "XYIIZX", "ZZZZZZ"])
def test_hsmpo_matches_statevector(pauli):
    ansatz = ParametricKickedIsing(
        nqubits=6, n_steps=3, delta_h=0.2, delta_j=-0.3, delta_z=0.15
    )
    hsmpo = HSMPO(ansatz=ansatz)
    assert hsmpo.expectation(pauli) == pytest.approx(
        _dense_expectation(_state(ansatz), pauli), abs=1e-10
    )


@pytest.mark.parametrize("cut_index", [0, 7, 18])
def test_head_tail_split_matches_statevector(cut_index):
    ansatz = ParametricKickedIsing(nqubits=6, n_steps=3, delta_h=0.2)
    evolutor = HSynthSMPO.rotations_only(ansatz)
    result = evolutor.expectation_at_cut("IZZIII", cut_index, method="simulation")
    assert result.value == pytest.approx(
        _dense_expectation(_state(ansatz), "IZZIII"), abs=1e-8
    )
