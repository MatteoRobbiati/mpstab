"""
Checks for the `KickedIsing` and `ParametricKickedIsing` ansatze: gate layout,
which gates HSMPO treats as magic, the parametric form being the same circuit at
"Clifford point + deviation" angles, and expectation values against the
statevector.
"""

import numpy as np
import pytest
from utils import dense_expectation, magic_gates

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import KickedIsing, ParametricKickedIsing


def _state(ansatz):
    return np.asarray(ansatz.circuit.copy()().state())


@pytest.mark.parametrize("nqubits, n_steps", [(4, 1), (5, 3)])
def test_kicked_ising_gate_count(nqubits, n_steps):
    """Each step has an RX and an RZ per qubit and a CNOT-RZ-CNOT per bond."""
    ansatz = KickedIsing(nqubits=nqubits, n_steps=n_steps)
    expected = n_steps * (2 * nqubits + 3 * (nqubits - 1))
    assert len(ansatz.circuit.queue) == expected


@pytest.mark.parametrize(
    "theta_h, theta_z, magic_per_step",
    [(np.pi / 2, 0.0, 0), (0.8, 0.0, 1), (0.8, 0.3, 2)],
    ids=["clifford", "rx_kicks", "rx_rz_kicks"],
)
def test_kicked_isingmagic_gates(theta_h, theta_z, magic_per_step):
    """With the default Clifford couplings, only the non-Clifford kicks are magic."""
    nqubits, n_steps = 5, 2
    ansatz = KickedIsing(
        nqubits=nqubits, n_steps=n_steps, theta_h=theta_h, theta_z=theta_z
    )
    assert len(magic_gates(ansatz)) == magic_per_step * nqubits * n_steps


def test_kicked_ising_non_clifford_couplings_are_magic():
    nqubits, n_steps = 4, 2
    ansatz = KickedIsing(nqubits=nqubits, n_steps=n_steps, theta_h=0.8, theta_j=0.3)
    assert len(magic_gates(ansatz)) == (nqubits + nqubits - 1) * n_steps


# ParametricKickedIsing: KickedIsing at "Clifford point + deviation" angles.


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
    magic = magic_gates(ansatz)
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


@pytest.mark.parametrize("cut_index", [0, 7, 18])
def test_head_tail_split_matches_statevector(cut_index):
    ansatz = ParametricKickedIsing(nqubits=6, n_steps=3, delta_h=0.2)
    evolutor = HQSMPO.rotations_only(ansatz)
    result = evolutor.expectation_at_cut("IZZIII", cut_index, method="simulation")
    assert result.value == pytest.approx(
        dense_expectation(_state(ansatz), "IZZIII"), abs=1e-8
    )


@pytest.mark.parametrize(
    "ansatz",
    [
        KickedIsing(nqubits=6, n_steps=3, theta_h=0.8, theta_z=0.3),
        ParametricKickedIsing(
            nqubits=6, n_steps=3, delta_h=0.2, delta_j=-0.3, delta_z=0.15
        ),
    ],
    ids=["kicked", "parametric"],
)
@pytest.mark.parametrize("pauli", ["ZIIIII", "IIXIII", "IZZIII", "XYIIZX", "ZZZZZZ"])
def test_hsmpo_matches_statevector(ansatz, pauli):
    assert HSMPO(ansatz=ansatz).expectation(pauli) == pytest.approx(
        dense_expectation(_state(ansatz), pauli), abs=1e-10
    )
