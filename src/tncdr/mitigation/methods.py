from typing import Optional

import numpy as np
from scipy.optimize import curve_fit

from qibo import (
    Circuit,
    hamiltonians,
    symbols,
    get_backend
)
from qibo.noise import NoiseModel

from tncdr.evolutors.models import HybridSurrogate
from tncdr.targets.ansatze import Ansatz

def tncdr(
        observable: str,
        ansatz: Ansatz,
        initial_state: Circuit,
        noise_model: NoiseModel,
        npartitions: int,
        magic_gates_per_partition: int,
        ncircuits: int = 50,
        nshots: Optional[int] = None,
        random_seed: int = 42,
        fit_map=lambda x, a, b: a * x + b  
    ):

    # Fix the RNG seed for reproducibility
    np.random.seed(random_seed)
    backend = get_backend()
    backend.set_seed(random_seed)

    # Construct the symbolic form from the observable pauli operators
    form = 1
    for i, pauli in enumerate(observable):
        form *= getattr(symbols, pauli)(i)

    # Compute the expectation value using the symbolic Hamiltonian
    ham = hamiltonians.SymbolicHamiltonian(form=form)

    # Update noise into the ansatz
    ansatz.update_noise_model(noise_model)

    # Construct the hybrid surrogate
    evo = HybridSurrogate(ansatz=ansatz, initial_state=initial_state)

    # Here we collect the tncdr results
    training_data = {
        "noisy_expvals": [],
        "exact_expvals": [],
    }

    for i in range(ncircuits):
        # Exact expval from surrogate
        exact_expval, _ = evo.expectation_from_partition(
            n_partitions=npartitions,
            magic_gates_per_partition=magic_gates_per_partition,
            observable=observable,
            return_partitions=False,
        )
        # Noisy expval from noisy simulator
        noisy_result = ansatz.execute(
            nshots=nshots, 
            initial_state=initial_state, 
            with_noise=True
        )
        noisy_expval = ham.expectation(noisy_result.state())

        training_data["exact_expvals"].append(exact_expval)
        training_data["noisy_expvals"].append(noisy_expval)

    # Convert lists to numpy arrays for curve_fit
    noisy_array = np.array(training_data["noisy_expvals"])
    exact_array = np.array(training_data["exact_expvals"])

    # Perform the curve fit using the provided mapping (default: linear)
    popt, _ = curve_fit(fit_map, noisy_array, exact_array)

    return training_data, popt
