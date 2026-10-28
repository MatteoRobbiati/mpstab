"""
Checks for the Meyer-Wallach global entanglement in `mpstab.models.entropies`:
the dense reference against known closed-form values, then the HSMPO estimator
(single-qubit Paulis backpropagated through the Clifford part) against the
dense reference on the same circuit.
"""

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.entropies import meyer_wallach, meyer_wallach_mps

set_backend("numpy")


def _ghz(n):
    circuit = Circuit(n)
    circuit.add(gates.H(0))
    for q in range(1, n):
        circuit.add(gates.CNOT(0, q))
    return circuit


def _kicked_ising(n, steps, theta_h):
    """RX(theta_h) kicks with Clifford RZZ(-pi/2) couplings, as in the benchmarks."""
    circuit = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            circuit.add(gates.RX(q, theta_h))
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                circuit.add(gates.CNOT(q, q + 1))
                circuit.add(gates.RZ(q + 1, -np.pi / 2))
                circuit.add(gates.CNOT(q, q + 1))
    return circuit


def _random_layers(n, layers, seed):
    rng = np.random.default_rng(seed)
    circuit = Circuit(n)
    for _ in range(layers):
        for q in range(n):
            circuit.add(gates.RY(q, rng.uniform(0, 2 * np.pi)))
            circuit.add(gates.RZ(q, rng.uniform(0, 2 * np.pi)))
        for q in range(n - 1):
            circuit.add(gates.CNOT(q, q + 1))
    return circuit


def test_meyer_wallach_product_state_is_zero():
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.RY(q, 0.3 * q + 0.1))
    assert meyer_wallach(np.asarray(circuit().state())) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_meyer_wallach_ghz_is_one(n):
    assert meyer_wallach(np.asarray(_ghz(n)().state())) == pytest.approx(1, abs=1e-12)


def test_meyer_wallach_partial_entanglement():
    """cos(a)|00> + sin(a)|11> has Tr rho^2 = cos^4 + sin^4 on both qubits."""
    a = 0.4
    state = np.array([np.cos(a), 0, 0, np.sin(a)])
    expected = 2 * (1 - (np.cos(a) ** 4 + np.sin(a) ** 4))
    assert meyer_wallach(state) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "circuit",
    [_ghz(4), _kicked_ising(6, 3, 1.2), _random_layers(5, 3, seed=1)],
    ids=["ghz", "kicked_ising", "random_layers"],
)
@pytest.mark.parametrize(
    "tn_engine", [QuimbEngine, NativeTensorNetworkEngine], ids=["quimb", "native"]
)
def test_meyer_wallach_mps_matches_dense(circuit, tn_engine):
    dense = meyer_wallach(np.asarray(circuit.copy()().state()))
    hsmpo = HSMPO(ansatz=circuit.copy())
    hsmpo.set_engines(tn_engine=tn_engine())
    assert meyer_wallach_mps(hsmpo) == pytest.approx(dense, abs=1e-10)
    assert hsmpo.meyer_wallach() == pytest.approx(dense, abs=1e-10)
