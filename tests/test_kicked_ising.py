"""
Checks for the `KickedIsing` ansatz: its gate layout, which gates HSMPO treats
as magic for Clifford and non-Clifford angles, and HSMPO expectation values
against the statevector of the same circuit.
"""

from functools import reduce

import numpy as np
import pytest
from qibo import set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import KickedIsing

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


def _magic_gates(ansatz):
    (magic_gates, _), _ = ansatz.partitionate_circuit(
        replacement_probability=0.0, replacement_method="closest"
    )
    return magic_gates


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
def test_kicked_ising_magic_gates(theta_h, theta_z, magic_per_step):
    """With the default Clifford couplings, only the non-Clifford kicks are magic."""
    nqubits, n_steps = 5, 2
    ansatz = KickedIsing(
        nqubits=nqubits, n_steps=n_steps, theta_h=theta_h, theta_z=theta_z
    )
    assert len(_magic_gates(ansatz)) == magic_per_step * nqubits * n_steps


def test_kicked_ising_non_clifford_couplings_are_magic():
    nqubits, n_steps = 4, 2
    ansatz = KickedIsing(nqubits=nqubits, n_steps=n_steps, theta_h=0.8, theta_j=0.3)
    assert len(_magic_gates(ansatz)) == (nqubits + nqubits - 1) * n_steps


@pytest.mark.parametrize("pauli", ["ZIIIII", "IIXIII", "IZZIII", "XYIIZX", "XYIIZY"])
def test_kicked_ising_hsmpo_matches_statevector(pauli):
    ansatz = KickedIsing(nqubits=6, n_steps=3, theta_h=0.8, theta_z=0.3)
    state = ansatz.circuit.copy()().state()
    hsmpo = HSMPO(ansatz=ansatz)
    assert hsmpo.expectation(pauli) == pytest.approx(
        _dense_expectation(state, pauli), abs=1e-10
    )
