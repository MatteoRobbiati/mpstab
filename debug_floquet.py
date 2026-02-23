import matplotlib.pyplot as plt
import numpy as np
from qibo import hamiltonians, set_backend, symbols
from tqdm import tqdm

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import FloquetAnsatz, HardwareEfficient, TranspiledAnsatz

set_backend("numpy")

# ------------- construct surrogate -------------
nqubits = 20
samples = 20

ans = FloquetAnsatz(
    nqubits=nqubits,
    nlayers=2,
    target_qubit=int(nqubits / 2),
    b=np.pi / 2,
    theta=0.25 * np.pi,
)

hs = HSMPO(ansatz=ans)

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

values = []
for _ in tqdm(range(samples)):
    # hs = HSMPO(ansatz=ans)
    exact_expval, partitions = hs.expectation_from_partition(
        replacement_probability=0.75,
        observable=obs,
        return_partitions=True,
        replacement_method="random",
    )
    print(exact_expval)
    values.append(exact_expval)

plt.hist(values)
plt.savefig("expvals_floquet.png")


# print(f"HS expectation value: {exact_expval}")
# qibo_expval = ham.expectation(partitions["full_circuit"]().state())
# print(f"Qibo expectation value: {qibo_expval}")
