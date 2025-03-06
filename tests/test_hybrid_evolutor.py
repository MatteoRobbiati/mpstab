import numpy as np

from qibo import hamiltonians, set_backend
from qibo import symbols
from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import HardwareEfficient

set_backend("numpy")

obs = "ZZZZ"

# Construct the ansatz
ansatz = HardwareEfficient(nqubits=4, nlayers=1, entangling=True)
ansatz.circuit.set_parameters(np.random.randn(len(ansatz.circuit.get_parameters())))
ansatz.circuit.draw()

print("Parameters before surrogate sampling: ", ansatz.circuit.get_parameters())

# Construct the evolutor
evo = HybridSurrogate(ansatz)

# Easy test to prove our hybrid evolutor works
result, partitions = evo.expectation_from_partition(
    n_partitions=1,
    magic_gates_per_partition=3,
    observable=obs,
    return_partitions=True,
)

partitions["full_circuit"].draw()

print("Parameters after surrogate sampling: ", partitions["full_circuit"].get_parameters())

print(f"my result: {result}")

print("---------")

form = 1
for i, pauli in enumerate(obs):
    form *= getattr(symbols, pauli)(i)

print(form)

ham = hamiltonians.SymbolicHamiltonian(form=form)
print("EXPVAL CIRC:", ham.expectation(partitions["full_circuit"]().state()))

