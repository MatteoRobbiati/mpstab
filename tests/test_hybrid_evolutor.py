import numpy as np

from qibo import hamiltonians
from qibo.symbols import X, Y, Z
from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import HardwareEfficient


# Construct the ansatz
ansatz = HardwareEfficient(nqubits=4, nlayers=1, entangling=False)

ansatz.circuit.set_parameters(np.random.randn(len(ansatz.circuit.get_parameters())))
ansatz.circuit.draw()

# Construct the evolutor
evo = HybridSurrogate(ansatz)

# Easy test to prove our hybrid evolutor works
result, partitions = evo.expectation_from_partition(
    n_partitions=1,
    magic_gates_per_partition=3,
    observable="ZZZZ",
    return_partitions=True,
)

print(f"my result: {result}")

print("---------")

form = Z(0) * Z(1) * Z(2) * Z(3)
ham = hamiltonians.SymbolicHamiltonian(form=form)
print("EXPVAL CIRC:", ham.expectation(ansatz.circuit().state()))

