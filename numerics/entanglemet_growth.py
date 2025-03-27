import numpy as np

from tncdr.evolutors.stabilizer.random_clifford import random_clifford_circuit
from tncdr.evolutors.stabilizer.tableaus import CNOT, SWAP, CZ, X, Z, Y, H, S

from mps import CircuitMPS
from stab_mps import StabilizerMPS

from tqdm import tqdm
from time import time

def apply_to_circ(circ, tableau, i):

    if tableau is None:
        return

    qubits = [i+q for q in tableau.qubits]

    if type(tableau) is CNOT:
        return circ.cnot(*qubits)
    if type(tableau) is CZ:
        return circ.cz(*qubits)
    if type(tableau) is SWAP:
        return circ.swap(*qubits)
    if type(tableau) is H:
        return circ.h(*qubits)
    if type(tableau) is S:
        return circ.s(*qubits)
    if type(tableau) is X:
        return circ.x(*qubits)
    if type(tableau) is Y:
        return circ.y(*qubits)
    if type(tableau) is Z:
        return circ.z(*qubits)
    
    raise ValueError(f'Unsupported tableau "{tableau.name}"')

def random_brickwall_clifford(q, D):

    gate_list = []
    for d in range(2*D):
        for i in range(0+d%2,q-1,2):
            for tab in random_clifford_circuit(n=2):
                gate_list.append((tab, i))

    return gate_list

def less_random_brickwall_clifford(q, D):

    single_qubit = [X, Y, Z, H, S]
    two_qubit = [CNOT, CZ]

    gate_list = []
    for d in range(2*D):
        for i in range(q-1):
            gate = single_qubit[np.random.randint(0,len(single_qubit)-1)]
            gate_list.append((gate(0) if gate is not None else None,i))

            gate = single_qubit[np.random.randint(0,len(single_qubit)-1)]
            gate_list.append((gate(0) if gate is not None else None,i))

        for i in range(0+d%2,q-1,2):
            gate = two_qubit[np.random.randint(0,len(two_qubit)-1)]
            gate_list.append((gate(0,1) if gate is not None else None,i))

    return gate_list

def compute_entropies(n, chi, M, D, circ_class):

    entropies = [0]
    circ = circ_class(n, max_bond_dimension=chi)

    for m in range(M):
        for tab, i in less_random_brickwall_clifford(n,D):
            apply_to_circ(circ, tab, i)
        circ.t(np.random.randint(0,n))
        entropies.append(circ.bipartite_entanglement_entropy((n-1)//2))

    return entropies

def main(n, chis, M, D, R):

    print(f'Test parameters: N={n}, D={D} ({R} samples per point)')
    for circ_type in [CircuitMPS, StabilizerMPS]:

        print(f'Testing circuit type: {circ_type.__name__}')
        n_le = 0
        for chi in chis:
            entropies = []
            for r in tqdm(range(R), desc=f'Bond dimension X={chi}', unit="circuit", ncols=80, ascii=" #"):
                while True:
                    try:
                        experiment = compute_entropies(n,chi, M, D, circ_type)
                        break
                    except np.linalg.LinAlgError:
                        n_le += 1

                entropies.append(experiment)

            name = 'simple-mps' if circ_type is CircuitMPS else 'stab-mps'
            np.save(f'numerics/data/less_random/entropies_N{n}_X{chi}_D{D}_{name}', np.mean(entropies, axis=0))
            np.save(f'numerics/data/less_random/errors_N{n}_X{chi}_D{D}_{name}', np.std(entropies, axis=0)/np.sqrt(R))

        print(f'Finished. LinAlgErrors encountered during SVD: {n_le}')
        print(f'---------------------------------')

if __name__ == '__main__':
    
    n = 40
    chis = [16, 32, 64, 128]
    M = 20
    D = 1
    R = 50

    main(n, chis, M, D, R)