"""A dummy example of tncdr"""
import numpy as np
import matplotlib.pyplot as plt

from qibo.noise import NoiseModel, PauliError
from qibo import (
    Circuit, 
    gates, 
    hamiltonians,
    symbols
)

from qiboedu.scripts.plotscripts import visualize_states

from tncdr.targets.ansatze import HardwareEfficient
from tncdr.mitigation.methods import tncdr

nqubits = 4
nlayers = 3

ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers, density_matrix=True)
ansatz.circuit.set_parameters(
    np.random.randn(ansatz.nparams)
)

# Initial state
init_circ = Circuit(nqubits=nqubits, density_matrix=True)
for q in range(nqubits):
    init_circ.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))

# Construct the observable
observable = "Z" * nqubits

noise_model = NoiseModel()
for q in range(nqubits):
    noise_model.add(
        PauliError(
        [
            ("X", np.abs(np.random.normal(0., 0.008))), 
            ("Y", np.abs(np.random.normal(0., 0.008))), 
            ("Z", np.abs(np.random.normal(0., 0.008)))
        ]), 
        qubits=q
    )


training_data, params = tncdr(
    observable=observable,
    ansatz=ansatz,
    initial_state=init_circ,
    noise_model=noise_model,
    npartitions=2,
    magic_gates_per_partition=3,
    ncircuits=20,
)

print(training_data)
print(params)
