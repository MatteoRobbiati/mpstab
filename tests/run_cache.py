import statistics
import time

import numpy as np
from utils import construct_symbolic_hamiltonian

from mpstab.engines import QuimbEngine, StimEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient


def execute_benchmark(qubit_sizes, iterations=5):
    """
    Measures median execution times across a range of qubit counts.
    Run this on your 'unoptimized' branch, save, then switch branches and run again.
    """
    results = []

    for n in qubit_sizes:
        print(f"Testing nqubits: {n}...")

        # Consistent setup for both branches
        ansatz = HardwareEfficient(nqubits=n, nlayers=2)
        hamiltonian = construct_symbolic_hamiltonian(nqubits=n, rng_seed=42)

        hs = HSMPO(ansatz=ansatz)
        hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine())

        run_times = []
        for i in range(iterations):
            t0 = time.perf_counter()
            _ = hs.expectation(hamiltonian)
            t1 = time.perf_counter()

            elapsed = t1 - t0
            run_times.append(elapsed)
            print(f"  Iter {i+1}: {elapsed:.4f}s")

        results.append([n, statistics.median(run_times)])

    return np.array(results)


if __name__ == "__main__":
    # Scaling up to 32 qubits
    qubit_range = [4, 8, 12, 16, 20, 24, 28, 32]

    data = execute_benchmark(qubit_range)

    # Save with a name reflecting your current branch
    # e.g., np.save("bench_optimized.npy", data)
    filename = "bench_results.npy"
    np.save(filename, data)
    print(f"\nExecution finished. Data saved to {filename}")
