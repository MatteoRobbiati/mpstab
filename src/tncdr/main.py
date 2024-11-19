from qibo import set_backend

from targets.ansatze import HardwareEfficient
from targets.entropies import scan_renyi_entropy

# qibojit if we will use more than 20 qubits
set_backend("numpy")

ansatz = HardwareEfficient(nqubits=3, nlayers=3)

# draw the circuit
ansatz.circuit.draw()

# execute it and collect final state
# first execution with initialized circuit (params are zeros)
state = ansatz.circuit(nshots=1000).state()
print(state)

renyi_entropies = scan_renyi_entropy(state=state, n_moments=2)
print(renyi_entropies)

# sample a random state from ansatz and compute it again
random_state = ansatz.sample_random_state(random_seed=42)
renyi_entropies = scan_renyi_entropy(state=random_state, n_moments=2)
print(renyi_entropies)