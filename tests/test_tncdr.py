"""A dummy example of tncdr"""
import numpy as np
import matplotlib.pyplot as plt

from qibo.noise import NoiseModel, PauliError
from qibo import (
    Circuit, 
    gates, 
    symbols,
    hamiltonians,
    set_backend
)
from qibo.models.error_mitigation import (
    CDR,
    vnCDR,
)

from tncdr.targets.ansatze import HardwareEfficient
from tncdr.mitigation.methods import TNCDR, density_matrix_circuit

nqubits = 5
nlayers = 3
nshots = 10000
ncircuits = 20

set_backend("numpy")

ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
ansatz.circuit.set_parameters(
    np.random.randn(ansatz.nparams)
)

# Initial state
init_circ = Circuit(nqubits=nqubits)
for q in range(nqubits):
    init_circ.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))

# Construct the observable
observable = "Z" * nqubits

noise_model = NoiseModel()
for q in range(nqubits):
    noise_model.add(
        PauliError(
        [
            ("X", np.abs(np.random.normal(0, 0.009))), 
            ("Y", np.abs(np.random.normal(0, 0.009))), 
            ("Z", np.abs(np.random.normal(0, 0.009)))
        ]), 
        qubits=q
    )


training_data, params = TNCDR(
    observable=observable,
    ansatz=ansatz,
    initial_state=init_circ,
    noise_model=noise_model,
    npartitions=2,
    nshots=nshots,
    magic_gates_per_partition=1,
    ncircuits=ncircuits,
    max_bond_dimension=None,
)

np.save(arr=training_data["noisy_expvals"], file="noisy_expvals")
np.save(arr=training_data["exact_expvals"], file="exact_expvals")

print(training_data)
print(params)

x = np.linspace(
    min(training_data["noisy_expvals"]), 
    max(training_data["noisy_expvals"]),
    100,
)

y = params[0] * x + params[1]

plt.scatter(training_data["noisy_expvals"], training_data["exact_expvals"], color="purple")
plt.plot(x, y, color="black")
plt.xlabel("Noisy")
plt.ylabel("Exact")
plt.grid(True)
plt.savefig("test_tncdr")

# Construct the symbolic form from the observable pauli operators
form = 1
for i, pauli in enumerate(observable):
    form *= getattr(symbols, pauli)(i)

# Compute the expectation value using the symbolic Hamiltonian
ham = hamiltonians.SymbolicHamiltonian(form=form)

exact_value = ham.expectation((init_circ + ansatz.circuit)().state())


noisy_init_circ = noise_model.apply(density_matrix_circuit(init_circ))
noisy_main_circ = noise_model.apply(density_matrix_circuit(ansatz.circuit))
noisy_outcome = (noisy_init_circ + noisy_main_circ)() 
noisy_value = ham.expectation(noisy_outcome.state())

(init_circ + ansatz.circuit).draw()
print((init_circ + ansatz.circuit).get_parameters())

cdr_mit_val, _, _, _ = CDR(
    circuit=density_matrix_circuit(init_circ + ansatz.circuit),
    observable=ham,
    noise_model=noise_model,
    nshots=nshots,
    n_training_samples=ncircuits,
    replacement_gates=[(gates.RY, {"theta": n * np.pi / 2}) for n in range(4)],
    target_non_clifford_gates=[gates.RY],
    full_output=True
)

print("################################\n\n")
print(f"Exact value: {exact_value}")
print(f"Noisy value: {noisy_value}")
print(f"TNCDR mitigated value: {noisy_value * params[0] + params[1]}")
print(f"CDR mitigated value: {cdr_mit_val}")