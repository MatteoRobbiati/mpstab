"""Tests for the variance and cost reductions of the noisy estimator."""

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab import HSMPO
from mpstab.engines import QuimbEngine
from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.models.noise import (
    AmplitudeDampingError,
    DepolarizingError,
    NoiseModel,
    PauliError,
)
from mpstab.pauli import pauli_matrix

OBSERVABLE = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZI": 0.3}


def _exact(noise, circuit, observable):
    rho = noise.apply(circuit)().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


def _hardware_efficient():
    np.random.seed(3)
    return HardwareEfficient(nqubits=5, nlayers=2)


def _kicked_ising():
    return KickedIsing(nqubits=5, n_steps=2, theta_h=0.8, theta_z=0.3, theta_j=0.4)


# -- Control variates --------------------------------------------------------------------


def test_multiplicative_model_is_exact_on_clifford_circuits():
    """Without rotations every fault only flips signs: the model is the value."""
    circuit = Circuit(4)
    for q in range(4):
        circuit.add(gates.H(q))
    for q in range(3):
        circuit.add(gates.CNOT(q, q + 1))
    circuit.add(gates.S(2))
    noise = NoiseModel().add(DepolarizingError(0.3))
    observable = {"XXYX": 0.6, "XIII": 0.4}
    exact = _exact(noise, circuit, observable)
    hsmpo = HSMPO(circuit)
    result = hsmpo.noisy_expectation(observable, noise, n_samples=500, seed=0)
    assert result.stratum_weights[2] > 0.1
    assert result.mean == pytest.approx(exact, abs=1e-10)
    assert result.stderr < 1e-10
    assert result.variance_reduction > 1e6


@pytest.mark.parametrize("seed", range(4))
def test_control_variates_are_unbiased_and_reduce_variance(seed):
    ansatz = _hardware_efficient()
    noise = (
        NoiseModel()
        .add(DepolarizingError(0.03))
        .add(PauliError({"ZZ": 0.01}), gates.CZ)
    )
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    result = HSMPO(ansatz).noisy_expectation(
        OBSERVABLE, noise, n_samples=400, seed=seed
    )
    assert result.stratum_weights[2] > 0.05
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
    assert result.variance_reduction > 1.0


def test_control_variates_with_coherences_are_unbiased():
    ansatz = _kicked_ising()
    noise = NoiseModel().add(AmplitudeDampingError(0.03)).add(DepolarizingError(0.01))
    exact = _exact(noise, ansatz.circuit, OBSERVABLE)
    result = HSMPO(ansatz).noisy_expectation(OBSERVABLE, noise, n_samples=1500, seed=5)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


# -- Evaluation --------------------------------------------------------------------------


def test_parallel_evaluation_matches_serial():
    ansatz = _kicked_ising()
    noise = NoiseModel().add(AmplitudeDampingError(0.02)).add(DepolarizingError(0.01))
    hsmpo = HSMPO(ansatz)
    serial = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=200, seed=2)
    parallel = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=200, seed=2, n_workers=2
    )
    assert parallel.mean == pytest.approx(serial.mean, abs=1e-13)
    assert parallel.stderr == pytest.approx(serial.stderr, abs=1e-13)
    assert parallel.n_transitions == serial.n_transitions > 0


def test_transition_skips_shared_isometries():
    engine = QuimbEngine()
    rng = np.random.default_rng(0)
    circuit = Circuit(8)
    for layer in range(3):
        for q in range(8):
            circuit.add(gates.RY(q, theta=rng.uniform(0, 3)))
        for q in range(layer % 2, 7, 2):
            circuit.add(gates.CNOT(q, q + 1))
    ket = engine.working_state(engine.build_circuit_mps(8, circuit))
    bra = engine.copy_state(ket)
    engine.apply_rotation(bra, engine.rotation_operator("IIIXZIII", 0.7), None)
    centers = (bra.center, ket.center)
    paulis = ["IIIIIIII", "IIIZIIII", "ZIIIIIIX", "IIYXIIII"]
    batches = [[pauli] for pauli in paulis] + [paulis, paulis[1::2]]
    for batch in batches:
        fast = kernels.pauli_transitions(bra.arrays, ket.arrays, batch, centers=centers)
        slow = kernels.pauli_transitions(bra.arrays, ket.arrays, batch)
        assert fast == pytest.approx(slow, abs=1e-13)
