import json
from copy import deepcopy
import time

import numpy as np
import qiskit
import json
from qiskit_ibm_runtime import RuntimeEncoder

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

#################### IBM RUN PREPARATION
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

INSTANCE = "cern/internal/tncdr"  # proper project instance
service = QiskitRuntimeService(instance=INSTANCE, channel="ibm_quantum")

BACKEND = service.least_busy(operational=True)
print('the selected BACKEND is: ', BACKEND.name)



q = int(config["nqubits"]/2)
observable = {"I" * (q) + "X" + "I" * (config["nqubits"] - (q + 1)): 1.0}

#################### QISKIT RUN 
# Instantiate Estimator with your real hardware backend. Specify shots if desired.
estimator = Estimator(mode=BACKEND)
###
from qiskit.quantum_info import SparsePauliOp

obs_dict = observable

# Convert to a list of (label, coeff) tuples:
pauli_list = list(obs_dict.items())

# Build a SparsePauliOp:
qiskit_obs = SparsePauliOp.from_list(pauli_list)

# Now `observable` is a proper Qiskit operator you can pass into Estimator.run(...)

###
observables = [qiskit_obs] * ncircuits
print('observables: ', observables)

#### transpilation and pub definition
from qiskit.transpiler import generate_preset_pass_manager
# Get ISA circuits
pm = generate_preset_pass_manager(optimization_level=0, backend=BACKEND)

pubs = []

for qc, obs in zip(qiskit_circuits, observables):
    isa_circuit = pm.run(qc)
    isa_obs = obs.apply_layout(isa_circuit.layout)
    pubs.append((isa_circuit, isa_obs))
 
estimator = Estimator(BACKEND)

####TIME
it = time.time()
job = estimator.run(pubs)


print('TNCDR JOB ID : ',job.job_id())

# When the job finishes, retrieve the expectation‐value array:
result_tncdr = job.result() 
print('TNCDR time: ', time.time() - it)

#expvals = result_tncdr.values   # a NumPy array of length N

# You can print them in order:
#for idx, val in enumerate(expvals):
   # print(f"Circuit {idx}  →  ⟨observable⟩ = {val:.6f}")
## Saving them
with open("result_tncdr.json", "w") as file:
    json.dump(result_tncdr, file, cls=RuntimeEncoder)

## #################    Algorithmiq TEM part - original qiskit circuit
#from qiskit_ibm_catalog import QiskitFunctionsCatalog

#tem_function_name = "algorithmiq/tem"

#catalog = QiskitFunctionsCatalog()

# Load your function
#tem = catalog.load(tem_function_name)

#tem_options = {
  #  "tem_max_bond_dimension": 64,
  #  "max_layers_to_learn": 6,
  #  "default_precision": 0.01,
#}
#,"compute_shadows_bias_from_observable": True

#pub = (original_qiskit_circuit, [observable])

#it = time.time()
#job_tem = tem.run(
 #   pubs=[pub], instance=INSTANCE, backend_name=BACKEND, options=tem_options
#)
#print('TEM time: ', time.time() - it)

## SAVING TEM RESULTS
#result_tem = job_tem.result()
#evs_tem = result[0].data.evs

 
#with open("result_tem.json", "w") as file:
 #   json.dump(result_tem, file, cls=RuntimeEncoder)

