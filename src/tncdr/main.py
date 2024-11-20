from qibo import set_backend

from targets.ansatze import HardwareEfficient
from targets.entropies import stabilizer_renyi_entropy

# qibojit if we will use more than 20 qubits
set_backend("numpy")

ansatz = HardwareEfficient(nqubits=5, nlayers=3)

# draw the circuit
ansatz.circuit.draw()

# execute it and collect final state
# first execution with initialized circuit (params are zeros)
state = ansatz.circuit(nshots=1000).state()

# test my SRE
sre = stabilizer_renyi_entropy(state=state, alpha=2)
print(f"\nSRE for circuit with all zero as parameters (pure clifford): {sre}")

# sample a random state from ansatz and compute it again
random_state = ansatz.sample_random_state(random_seed=42)
sre = stabilizer_renyi_entropy(state=random_state, alpha=2)
print(f"SRE for circuit with all random parameters (non-clifford): {sre}")

# sample a random state but quasi-clifford and compute it again
random_state = ansatz.sample_random_quasi_clifford_state(cliff_fraction=0.7, random_seed=42)
sre = stabilizer_renyi_entropy(state=random_state, alpha=2)
print(f"SRE for circuit for a quasi-clifford circuit: {sre}")