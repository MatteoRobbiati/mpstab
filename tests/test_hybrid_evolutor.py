import click
import json
import time
from pathlib import Path
import numpy as np

from qibo import hamiltonians, set_backend, symbols, Circuit, gates
from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import HardwareEfficient

@click.command()
@click.option('--nqubits', default=6, type=int, help='Number of qubits.')
@click.option('--nlayers', default=3, type=int, help='Number of layers in the ansatz.')
@click.option('--npartitions', default=2, type=int, help='Number of partitions.')
@click.option('--magic_gates_per_partition', default=1, type=int, help='Number of magic gates per partition.')
@click.option('--random_seed', default=42, type=int, help='Random number generator seed.')
def main(nqubits, nlayers, npartitions, magic_gates_per_partition, random_seed):

    # Automatically capture all function arguments
    out_results = locals().copy()
    
    # Set backend to numpy
    set_backend("numpy")

    np.random.seed(random_seed)

    folder_path = Path(f"results/{nqubits}q_{nlayers}l_{npartitions}p_{magic_gates_per_partition}magic")
    folder_path.mkdir(parents=True, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Define the observable string
    obs = "Z" * nqubits

    # Construct the ansatz with the given number of qubits and layers
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers, entangling=True)

    # Initial state
    init_circ = Circuit(nqubits=nqubits)
    for q in range(nqubits):
        init_circ.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))


    # Set random parameters for the circuit
    params = np.random.randn(len(ansatz.circuit.get_parameters()))
    ansatz.circuit.set_parameters(params)
        
    # Construct the hybrid surrogate evolutor using the ansatz
    evo = HybridSurrogate(ansatz=ansatz, initial_state=init_circ)

    # Compute the expectation value using the given number of partitions and magic gates per partition
    result, partitions = evo.expectation_from_partition(
        n_partitions=npartitions,
        magic_gates_per_partition=magic_gates_per_partition,
        observable=obs,
        return_partitions=True,
    )

    partitions["full_circuit"].draw()

    # Construct the symbolic form from the observable pauli operators
    form = 1
    for i, pauli in enumerate(obs):
        form *= getattr(symbols, pauli)(i)

    # Compute the expectation value using the symbolic Hamiltonian
    ham = hamiltonians.SymbolicHamiltonian(form=form)
    exact_expval = ham.expectation((init_circ + partitions["full_circuit"])().state())

    # Compute elapsed time
    elapsed_time = time.time() - start_time

    # Store results
    out_results.update({
        "total_time": elapsed_time,
        "exact_expval": exact_expval,
        "my_result": result.tolist() if isinstance(result, np.ndarray) else result
    })

    # Save results to JSON file
    json_path = folder_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(out_results, f, indent=4)

    print(f"Results saved to {json_path}")

if __name__ == '__main__':
    main()