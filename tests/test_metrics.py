"""
Checks for the HSMPO estimators in `mpstab.models.metrics` against Qibo's
statevector references. The Meyer-Wallach estimator (single-qubit Paulis
backpropagated through the Clifford part) is also checked on closed-form values;
the sampled stabilizer Renyi entropy only to within its statistical error.
"""

import numpy as np
import pytest
from qibo import Circuit, gates, set_backend
from qibo.quantum_info import meyer_wallach_entanglement, stabilizer_renyi_entropy

from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.models.metrics import meyer_wallach_mps, stabilizer_renyi_entropy_mps

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


def _hardware_efficient(n, nlayers, seed):
    """HardwareEfficient circuit with seeded angles, so the test is reproducible."""
    circuit = HardwareEfficient(nqubits=n, nlayers=nlayers).circuit
    rng = np.random.default_rng(seed)
    circuit.set_parameters(rng.uniform(-np.pi, np.pi, len(circuit.get_parameters())))
    return circuit


def test_meyer_wallach_mps_product_state_is_zero():
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.RY(q, 0.3 * q + 0.1))
    assert meyer_wallach_mps(HSMPO(ansatz=circuit)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_meyer_wallach_mps_ghz_is_one(n):
    assert meyer_wallach_mps(HSMPO(ansatz=_ghz(n))) == pytest.approx(1, abs=1e-12)


def test_meyer_wallach_mps_partial_entanglement():
    """cos(a)|00> + sin(a)|11> has Tr rho^2 = cos^4 + sin^4 on both qubits."""
    a = 0.4
    circuit = Circuit(2)
    circuit.add(gates.RY(0, 2 * a))
    circuit.add(gates.CNOT(0, 1))
    expected = 2 * (1 - (np.cos(a) ** 4 + np.sin(a) ** 4))
    assert meyer_wallach_mps(HSMPO(ansatz=circuit)) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize(
    "circuit",
    [_ghz(4), _kicked_ising(6, 3, 1.2), _hardware_efficient(5, 3, seed=1)],
    ids=["ghz", "kicked_ising", "hardware_efficient"],
)
@pytest.mark.parametrize(
    "tn_engine", [QuimbEngine, NativeTensorNetworkEngine], ids=["quimb", "native"]
)
def test_meyer_wallach_mps_matches_dense(circuit, tn_engine):
    dense = meyer_wallach_entanglement(circuit.copy()().state())
    hsmpo = HSMPO(ansatz=circuit.copy())
    hsmpo.set_engines(tn_engine=tn_engine())
    assert meyer_wallach_mps(hsmpo) == pytest.approx(dense, abs=1e-10)


def test_meyer_wallach_mps_grows_with_hardware_efficient_layers():
    """
    Deeper HardwareEfficient circuits approach the Haar-random average
    (2^n - 2) / (2^n + 1), close to 1. A single random instance need not grow
    monotonically, so Q is averaged over seeds.
    
    """
    n, seeds = 6, range(10)
    nlayers = [1, 3, 5, 10]
    mean_q = [
        np.mean(
            [
                meyer_wallach_mps(HSMPO(ansatz=_hardware_efficient(n, L, s)))
                for s in seeds
            ]
        )
        for L in nlayers
    ]
    assert np.all(np.diff(mean_q) > 0)
    haar = (2**n - 2) / (2**n + 1)
    assert mean_q[-1] == pytest.approx(haar, abs=0.05)


def _t_layer(n):
    """H, T and a generic RY on every qubit, then a CNOT ladder: a magic state."""
    circuit = Circuit(n)
    for q in range(n):
        circuit.add(gates.H(q))
        circuit.add(gates.T(q))
        circuit.add(gates.RY(q, 0.3 + 0.2 * q))
    for q in range(n - 1):
        circuit.add(gates.CNOT(q, q + 1))
    return circuit


@pytest.mark.parametrize("alpha", [2, 3])
def test_stabilizer_renyi_entropy_mps_matches_dense(alpha):
    circuit = _t_layer(2)
    dense = stabilizer_renyi_entropy(circuit.copy()().state(), alpha)
    hsmpo = HSMPO(ansatz=circuit.copy())
    estimate = stabilizer_renyi_entropy_mps(hsmpo, alpha=alpha, n_samples=4000, seed=0)
    assert estimate == pytest.approx(dense, abs=0.1)


def test_stabilizer_renyi_entropy_mps_rejects_alpha_below_two():
    with pytest.raises(ValueError):
        stabilizer_renyi_entropy_mps(HSMPO(ansatz=_ghz(2)), alpha=1)
