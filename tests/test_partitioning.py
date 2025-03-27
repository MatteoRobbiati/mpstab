import numpy as np

from qibo import set_backend

from tncdr.targets.ansatze import HardwareEfficient, TranspiledAnsatz

set_backend("numpy")
np.random.seed(42)

# Use the Hardware efficient ansataz to extract the circuit structure
ansatz = HardwareEfficient(
    nqubits=5,
    nlayers=5
)


circuit = ansatz.circuit
ansatz = TranspiledAnsatz(original_circuit=circuit)
ansatz.circuit.draw()

partitioned_circ, magic_layers, stabilizer_layers = ansatz.partitionate_circuit(replacement_probability=0.5)

for i in range(len(stabilizer_layers)):
    stabilizer_layers[i].draw()
    print(stabilizer_layers[i].get_parameters())