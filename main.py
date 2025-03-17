import click
import json
import time

from pathlib import Path
import numpy as np
from scipy.stats import median_abs_deviation

from qibo import (
    hamiltonians, 
    set_backend, 
    symbols,
    Circuit,
    gates,
)
from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import HardwareEfficient
# from tncdr.evolutors.stabilizer.random_clifford import random_pauli

@click.command()
@click.option('--nqubits', default=7, type=int, help='Number of qubits.')
@click.option('--nlayers', default=3, type=int, help='Number of layers in the ansatz.')
@click.option('--npartitions', default=2, type=int, help='Number of partitions.')
@click.option('--magic_gates_per_partition', default=1, type=int, help='Number of magic gates per partition.')
@click.option('--random_seed', default=42, type=int, help='Random number generator seed.')
@click.option('--n_runs', default=10, type=int, help='Number of runs computed.')
def main(nqubits, nlayers, npartitions, magic_gates_per_partition, random_seed, n_runs):

    # Automatically capture all function arguments
    out_results = locals().copy()
    
    # Set backend to numpy
    set_backend("numpy")

    np.random.seed(random_seed)

    # Create the result folder
    folder_path = Path(
        f"results/{nqubits}q_{nlayers}l_{npartitions}p_{magic_gates_per_partition}magic/{random_seed}rng_{n_runs}runs"
    )
    folder_path.mkdir(parents=True, exist_ok=True)

    # Construct the ansatz with the given number of qubits and layers
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers, entangling=True)

    # Initial state preparation
    init_state = Circuit(nqubits=nqubits)
    [init_state.add(gates.RY(q=q, theta=0.)) for q in range(nqubits)]

    times, circuit_expvals, surrogate_expvals = [], [], []
    # TODO: optimize this loop
    for i in range(n_runs):
        # Define the observable string
        obs = "Z" * nqubits      
        print(f"Run {i+1}/{n_runs}")
        # Start timing
        start_time = time.time()

        # Set random parameters for the circuit
        params = np.random.randn(len(ansatz.circuit.get_parameters()))
        ansatz.circuit.set_parameters(params)

        # New random params in the state
        init_state.set_parameters(np.random.uniform(-np.pi, np.pi, nqubits))
            
        # Construct the hybrid surrogate evolutor using the ansatz
        evo = HybridSurrogate(ansatz=ansatz, initial_state=init_state)

        # Compute the expectation value using the given number of partitions and magic gates per partition
        result, partitions = evo.expectation_from_partition(
            n_partitions=npartitions,
            magic_gates_per_partition=magic_gates_per_partition,
            observable=obs,
            return_partitions=True,
        )

        # Construct the symbolic form from the observable pauli operators
        form = 1
        for i, pauli in enumerate(obs):
            form *= getattr(symbols, pauli)(i)

        # Compute the expectation value using the symbolic Hamiltonian
        ham = hamiltonians.SymbolicHamiltonian(form=form)
        exact_expval = ham.expectation((init_state + partitions["full_circuit"])().state())

        # Compute elapsed time
        elapsed_time = time.time() - start_time

        # Save results
        times.append(elapsed_time)
        circuit_expvals.append(exact_expval)
        surrogate_expvals.append(result)
    

    # Store results
    out_results.update({
        "median_time": np.median(times),
        "error_times": median_abs_deviation(times),
        "exact_expvals_list": circuit_expvals,
        "surrogate_expvals_list": surrogate_expvals,
    })

    # Save results to JSON file
    json_path = folder_path / "results.json"
    with open(json_path, "w") as f:
        json.dump(out_results, f, indent=4)


if __name__ == '__main__':
    main()
