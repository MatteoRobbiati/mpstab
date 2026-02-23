import numpy as np
import tqdm
from qibo import Circuit, gates, set_backend

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient, TranspiledAnsatz
from mpstab.models.utils import obs_string_to_qibo_hamiltonian

nqubits = 20
obs_str = "Z" * nqubits
gate_sampling_prob = 0.5
cut_idx = nqubits
nruns = 10


def compute_commutator(c1: Circuit, c2: Circuit) -> float:
    """Compute difference of AB aqnd BA expvals."""

    evo_12 = HSMPO(ansatz=TranspiledAnsatz(original_circuit=c1 + c2))
    evo_21 = HSMPO(ansatz=TranspiledAnsatz(original_circuit=c2 + c1))

    expv_12 = evo_12.expectation(observable=obs_str)
    expv_21 = evo_21.expectation(observable=obs_str)

    return np.abs(expv_12 - expv_21)


def construct_circuit(
    nqubits: int,
    nlayers: int = 2,
    ry_sampling_prob: float = 0.5,
    cz_sampling_prob: float = 0.8,
) -> Circuit:
    """Construct circuit with random RY and CZ."""
    c = Circuit(nqubits)
    for _ in range(nlayers):
        for q in range(nqubits):
            r1 = np.random.uniform(0, 1)
            if r1 <= ry_sampling_prob:
                c.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            r2 = np.random.uniform(0, 1)
            if r2 <= cz_sampling_prob:
                c.add(gates.CZ(q0=q % nqubits, q1=(q + 1) % nqubits))
    return c


def split_circuit(circuit: Circuit, cut_index: int) -> tuple:
    """Split a circuit into two sub-circuits."""
    c1 = Circuit(circuit.nqubits)
    c2 = Circuit(circuit.nqubits)

    for i, gate in enumerate(circuit.queue):
        if i <= cut_index:
            c1.add(gate)
        else:
            c2.add(gate)

    return (c1, c2)


qubits_averages = []
for iq in [5, 10, 15]:
    ave_values = []

    for prob_value in [0.0, 0.25, 0.5, 0.75, 0.99]:
        print(
            f"Executing experiment with {iq} qubits and CZ sampling prob: {prob_value}."
        )
        commutators = []
        for _ in tqdm.tqdm(range(nruns)):
            c = construct_circuit(
                nqubits=nqubits,
                nlayers=2,
                ry_sampling_prob=0.4,
                cz_sampling_prob=prob_value,
            )
            c1, c2 = split_circuit(circuit=c, cut_index=int(len(c.queue) / 3.0))
            commutators.append(compute_commutator(c1, c2))
        ave_values.append(np.mean(commutators))
        print(f"Obtained commutator value: {ave_values[-1]}\n\n")
    qubits_averages.append(ave_values)

np.save(arr=np.asarray(qubits_averages), file="averages_on_qubits")
