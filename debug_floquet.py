import numpy as np
from qibo import hamiltonians, set_backend, symbols

from mpstab.evolutors.models import HybridSurrogate
from mpstab.targets.ansatze import FloquetAnsatz, HardwareEfficient, TranspiledAnsatz

set_backend("numpy")

# ------------- construct surrogate -------------
nqubits = 5

ans = FloquetAnsatz(
    nqubits=nqubits, nlayers=3, target_qubit=int(nqubits / 2), b=np.pi / 2
)

hs = HybridSurrogate(ansatz=ans)

# ------------- construct hamiltonian -------------


def generate_obs(nqubits):
    obs = "".join("X" if i == int(nqubits / 2) else "I" for i in range(nqubits))
    form = 1
    for i, pauli in enumerate(obs):
        form *= getattr(symbols, pauli)(i)
    ham = hamiltonians.SymbolicHamiltonian(form=form)
    return obs, ham


# ----------------- compute expectation value -------------

obs, ham = generate_obs(nqubits)

exact_expval, partitions = hs.expectation_from_partition(
    replacement_probability=0.7,
    observable=obs,
    return_partitions=True,
    replacement_method="random",
)
print(f"HS expectation value: {exact_expval}")
qibo_expval = ham.expectation(partitions["full_circuit"]().state())
print(f"Qibo expectation value: {qibo_expval}")
