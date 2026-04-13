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


def build_circuit(nqubits, nlayers):
    circ = Circuit(nqubits)
    for layer in range(nlayers):
        for q in range(nqubits):
            circ.add(gates.RY(q=q, theta=0.0))
            circ.add(gates.RZ(q=q, theta=0.0))
            circ.add(gates.RX(q=q, theta=0.0))
        for q in range(nqubits - 1):
            circ.add(gates.CNOT(q, q + 1))
            circ.add(gates.SWAP(q, q + 1))
    circ.add(gates.M(*range(nqubits)))
    return circ


@pytest.mark.parametrize(
    "backend",
    [
        "jax",
    ],
)
def test_gradients(backend):

    nqubits = 10
    nlayers = 3
    observable = "ZX" * int(nqubits / 2)

    circuit = build_circuit(nqubits, nlayers)
    ansatz = CircuitAnsatz(qibo_circuit=circuit)
    hs = HSMPO(ansatz)

    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend=backend))

    expectation = hs.expectation(observable)

    def f(model: HSMPO, observable, params):
        model.ansatz.circuit.set_parameters(params)
        return hs.expectation(observable)

    if backend == "jax":
        parameters = np.random.uniform(
            -np.pi, np.pi, size=len(circuit.get_parameters())
        )
        grad = jax.grad(f, argnums=2)(hs, observable, parameters)

    if backend == "torch":
        parameters = torch.distributions.Uniform(-np.pi, np.pi).sample(
            (len(circuit.get_parameters()),)
        )
        parameters.requires_grad_(True)
        loss = f(hs, observable, parameters)
        loss.backward()

    assert grad is not None
