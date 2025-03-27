import numpy as np

from qibo import set_backend

from tncdr.targets.ansatze import HardwareEfficient
from tncdr.targets.entropies import stabilizer_renyi_entropy

set_backend("numpy")

np.random.seed(42)

ansatz = HardwareEfficient(
    nqubits=5,
    nlayers=5
)

part_circ, mag_layers, stab_layers = ansatz.partitionate_circuit(
    n_partitions=4,
    magic_gates_per_partition=2,
)


print(f"SRE for initial ansatz: {stabilizer_renyi_entropy(state=ansatz.circuit().state(), alpha=2)}")
print(f"SRE for partitioned ansatz: {stabilizer_renyi_entropy(state=part_circ().state(), alpha=2)}")
for i, magic_l in enumerate(mag_layers):
    print(f"SRE for magic partition {i+1}: {stabilizer_renyi_entropy(state=magic_l().state(), alpha=2)}")
for i, stab_l in enumerate(stab_layers):
    print(f"SRE for clifford partition {i+1}: {stabilizer_renyi_entropy(state=stab_l().state(), alpha=2)}")
