"""
Checks for `RampedKickedIsing` and `StarKickedIsing`: without a ramp and without
randomness the chain is `ParametricKickedIsing`, the ramp moves the magic across
the steps, the random draws are reproducible, and HSMPO matches the statevector.
"""

from functools import reduce

import numpy as np
import pytest
from qibo import set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import (
    ParametricKickedIsing,
    RampedKickedIsing,
    StarKickedIsing,
)

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


def _gates(ansatz):
    return [
        (gate.name, gate.qubits, tuple(float(p) for p in gate.parameters))
        for gate in ansatz.circuit.queue
    ]


def _kick_deviations(ansatz):
    """The magic RX angles, step by step (one row per step)."""
    angles = [
        float(gate.parameters[0])
        for gate in ansatz.circuit.queue
        if gate.name == "rx" and not gate.clifford
    ]
    return np.reshape(angles, (ansatz.n_steps, ansatz.nqubits))


@pytest.mark.parametrize(
    "deltas", [(0.2, 0.0, 0.0), (0.2, -0.3, 0.15)], ids=["h", "h_j_z"]
)
def test_without_ramp_is_parametric_kicked_ising(deltas):
    delta_h, delta_j, delta_z = deltas
    kwargs = dict(
        nqubits=5, n_steps=3, delta_h=delta_h, delta_j=delta_j, delta_z=delta_z
    )
    assert _gates(RampedKickedIsing(**kwargs)) == _gates(
        ParametricKickedIsing(**kwargs)
    )


def test_ramp_is_geometric_from_first_to_last_step():
    ansatz = RampedKickedIsing(nqubits=4, n_steps=4, delta_h=0.5, delta_h_last=0.05)
    deviations = _kick_deviations(ansatz)
    np.testing.assert_allclose(deviations[:, 0], np.geomspace(0.5, 0.05, 4))
    # the same deviation on every qubit of a step
    np.testing.assert_allclose(deviations, deviations[:, :1] * np.ones((1, 4)))


def test_ramp_through_zero_is_linear():
    ansatz = RampedKickedIsing(
        nqubits=3, n_steps=3, delta_h=0.4, delta_h_last=-0.2, delta_z=0.1
    )
    rx = [
        float(g.parameters[0])
        for g in ansatz.circuit.queue
        if g.name == "rx" and not g.clifford
    ]
    np.testing.assert_allclose(rx[::3], np.linspace(0.4, -0.2, 3))


def test_ramp_direction_changes_the_state():
    forward = RampedKickedIsing(nqubits=4, n_steps=3, delta_h=0.5, delta_h_last=0.05)
    reverse = RampedKickedIsing(nqubits=4, n_steps=3, delta_h=0.05, delta_h_last=0.5)
    assert len(forward.circuit.queue) == len(reverse.circuit.queue)
    assert abs(np.vdot(_state(forward), _state(reverse))) < 1 - 1e-6


@pytest.mark.parametrize("cls", [RampedKickedIsing, StarKickedIsing])
def test_random_draws_follow_the_seed(cls):
    kwargs = dict(nqubits=6, n_steps=3, delta_h=0.3, disorder=0.4, bond_density=0.6)
    assert _gates(cls(seed=1, **kwargs)) == _gates(cls(seed=1, **kwargs))
    assert _gates(cls(seed=1, **kwargs)) != _gates(cls(seed=2, **kwargs))
    # without a seed, numpy's global generator: the same stream as RandomState(seed)
    np.random.seed(1)
    assert _gates(cls(**kwargs)) == _gates(cls(seed=1, **kwargs))


def test_disorder_spreads_the_kicks_around_the_profile():
    ansatz = RampedKickedIsing(nqubits=8, n_steps=2, delta_h=0.4, disorder=0.3, seed=0)
    deviations = _kick_deviations(ansatz)
    assert np.all(np.abs(deviations / 0.4 - 1) <= 0.3 + 1e-12)
    assert np.ptp(deviations) > 0


def test_bond_density_drops_couplings_only():
    full = RampedKickedIsing(nqubits=8, n_steps=3, delta_h=0.3)
    diluted = RampedKickedIsing(
        nqubits=8, n_steps=3, delta_h=0.3, bond_density=0.5, seed=0
    )
    count = lambda a, name: sum(g.name == name for g in a.circuit.queue)
    assert count(diluted, "cx") < count(full, "cx")
    assert count(diluted, "rx") == count(full, "rx")
    with pytest.raises(ValueError):
        RampedKickedIsing(nqubits=4, bond_density=0.0)


def test_star_couples_the_hub_to_every_leaf():
    nqubits, n_steps = 5, 2
    ansatz = StarKickedIsing(nqubits=nqubits, n_steps=n_steps, delta_h=0.2)
    bonds = [g.qubits for g in ansatz.circuit.queue if g.name == "cx"]
    one_step = [(0, leaf) for leaf in range(1, nqubits) for _ in range(2)]
    assert bonds == one_step * n_steps


@pytest.mark.parametrize(
    "ansatz",
    [
        RampedKickedIsing(
            nqubits=6,
            n_steps=3,
            delta_h=0.5,
            delta_h_last=0.1,
            delta_j=0.2,
            delta_z=0.1,
        ),
        RampedKickedIsing(nqubits=6, n_steps=3, delta_h=0.3, disorder=0.3, seed=2),
        StarKickedIsing(
            nqubits=6, n_steps=3, delta_h=0.4, delta_h_last=0.1, delta_j=0.2
        ),
    ],
    ids=["ramped", "disordered", "star"],
)
@pytest.mark.parametrize("pauli", ["ZZIIII", "IIXIII", "ZIIIIZ", "XYIIZX"])
def test_hsmpo_matches_statevector(ansatz, pauli):
    assert HSMPO(ansatz=ansatz).expectation(pauli) == pytest.approx(
        _dense_expectation(_state(ansatz), pauli), abs=1e-10
    )
