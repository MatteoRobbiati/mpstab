import json
from copy import deepcopy

import numpy as np
import qiskit

from tncdr.targets.ansatze import FloquetAnsatz, TranspiledAnsatz


# Function to convert from Qibo to Qiskit
def qibo_to_qiskit(qibocirc):
    """Convert a Qibo circuit into a Qiskit QuantumCircuit."""
    qasm_circuit = qibocirc.to_qasm()
    qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(qasm_circuit)
    return qiskit_circuit


# Set the number of circuits we want to load
ncircuits = 10

# Loading the configuration
with open("circuits/results.json") as f:
    config = json.load(f)

# Initializing Floquet ansatz, transpiling it and constructing a Qibo circuit
ansatz = FloquetAnsatz(
    nqubits=config["nqubits"],
    nlayers=config["nlayers"],
    b=config["b"],
    theta=config["theta"],
)
ansatz_circuit = ansatz.circuit
original_qibo_circuit = ansatz.circuit
# Setting the original parameters
original_qibo_circuit.set_parameters(np.load("circuits/original_circuit_params.npy"))

# Constructing the equivalent Qiskit circuit
original_qiskit_circuit = qibo_to_qiskit(original_qibo_circuit)

# Load all the training circuits required for TNCDR
qiskit_circuits = []
for i in range(ncircuits):
    params = np.load(f"circuits/params_circuit{i}.npy")
    qibo_circ = deepcopy(original_qibo_circuit)
    qibo_circ.set_parameters(params)
    qiskit_circuits.append(qibo_to_qiskit(qibo_circ))

print(qiskit_circuits)
