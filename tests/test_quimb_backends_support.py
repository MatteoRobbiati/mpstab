import jax
import numpy as np
import pytest
import torch
from qibo import Circuit, gates

from mpstab.engines import QuimbEngine, StimEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_backend_support(backend):

    nqubits = 10
    nlayers = 3
    observable = "ZX" * int(nqubits / 2)

    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
    hs = HSMPO(ansatz)

    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend=backend))

    expectation = hs.expectation(observable)

    assert expectation is not None


def _non_clifford_circuit(nqubits, nlayers, rng):
    """Rotations at generic angles, so every one of them is a magic gate."""
    circuit = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            circuit.add(gates.RY(q, theta=rng.uniform(0.3, 1.2)))
            circuit.add(gates.RZ(q, theta=rng.uniform(0.3, 1.2)))
        for q in range(nqubits - 1):
            circuit.add(gates.CNOT(q, q + 1))
    return circuit


def _dense_expectation(circuit, parameters, observable):
    """Exact <observable> from the statevector, in float64."""
    circuit = circuit.copy(deep=True)
    circuit.set_parameters(parameters)
    state = circuit().state(numpy=True)
    paulis = {"Z": np.diag([1.0, -1.0]), "X": np.array([[0.0, 1.0], [1.0, 0.0]])}
    operator = np.array([[1.0]])
    for pauli in observable:
        operator = np.kron(operator, paulis[pauli])
    return float(np.real(state.conj() @ operator @ state))


def test_gradients_jax():
    """
    ``jax.grad`` through ``HSMPO.set_parameters`` + ``expectation`` matches
    central finite differences of the exact statevector expectation.

    The angles are generic (non-Clifford) on purpose: a Clifford gate is folded
    into the stabilizer frame and cannot carry a traced parameter. Only jax is
    covered, since torch's ``backward`` cannot differentiate through the complex
    SVD the MPS evolution uses.
    """
    nqubits, nlayers, observable = 4, 2, "ZXZX"
    circuit = _non_clifford_circuit(nqubits, nlayers, np.random.default_rng(0))
    hs = HSMPO(CircuitAnsatz(qibo_circuit=circuit))
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend="jax"))

    def f(parameters):
        hs.set_parameters(parameters)
        return hs.expectation(observable)

    parameters = np.array(circuit.get_parameters()).ravel()
    gradient = np.array(jax.grad(f)(parameters))

    eps = 1e-5
    expected = np.array(
        [
            (
                _dense_expectation(circuit, parameters + eps * e, observable)
                - _dense_expectation(circuit, parameters - eps * e, observable)
            )
            / (2 * eps)
            for e in np.eye(len(parameters))
        ]
    )
    np.testing.assert_allclose(gradient, expected, atol=1e-4)
