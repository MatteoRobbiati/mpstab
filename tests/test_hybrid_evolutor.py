import numpy as np

from qibo import hamiltonians
from qibo.symbols import X, Y, Z
from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import HardwareEfficient


# Construct the ansatz
ansatz = HardwareEfficient(nqubits=4, nlayers=1)

ansatz.circuit.set_parameters(np.random.randn(len(ansatz.circuit.get_parameters())))
ansatz.circuit.draw()

# Partitioning the circuit into magic and stabilizer blocks
part_circ, mag_layers, stab_layers = ansatz.partitionate_circuit(
    n_partitions=1,
    magic_gates_per_partition=3,
)

evo = HybridSurrogate(ansatz)

result = evo.expectation_from_partition(
    n_partitions=1,
    magic_gates_per_partition=2,
    observable="ZZZZ",
)

print(f"my result: {result}")

print("---------")
mag_layers[0].draw()

form = Z(0) * Z(1) * Z(2) * Z(3)
ham = hamiltonians.SymbolicHamiltonian(form=form)
print("EXPVAL CIRC:", ham.expectation(mag_layers[0]().state()))


p = evo.backpropagate_pauli(observable="ZZZZ", stabilizer_circuit=stab_layers[0])
print("XYZ", p)