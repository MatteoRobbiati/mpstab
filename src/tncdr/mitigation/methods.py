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

def TNCDR(
        observable: str,
        ansatz: Ansatz,
        initial_state: Circuit,
        noise_model: NoiseModel,
        replacement_probability: float,
        ncircuits: int = 50,
        nshots: Optional[int] = None,
        random_seed: int = 42,
        fit_map=lambda x, a, b: a * x + b,
        expval_threshold: float = 1e-7,  
        max_bond_dimension: Optional[int] = None,
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

    # Here we collect the tncdr results
    training_data = {
        "noisy_expvals": [],
        "exact_expvals": [],
    }

    for i in range(ncircuits):
        # Construct the hybrid surrogate
        evo = HybridSurrogate(ansatz=ansatz, initial_state=initial_state)

        # Exact expval from surrogate
        exact_expval, partitions = evo.expectation_from_partition(
            replacement_probability=replacement_probability,
            observable=observable,
            return_partitions=True,
            max_bond_dimension=max_bond_dimension,
        )

        if np.abs(exact_expval) < expval_threshold:
            continue
    
        sampled_circuit = density_matrix_circuit(partitions["full_circuit"])
        initialised_sampled_circuit = density_matrix_circuit(initial_state) + sampled_circuit
        noisy_init_sampled_circuit = noise_model.apply(initialised_sampled_circuit)
        noisy_expval = ham.expectation_from_samples(
            noisy_init_sampled_circuit(
                nshots=nshots
            ).frequencies()
        )

        training_data["exact_expvals"].append(exact_expval)
        training_data["noisy_expvals"].append(noisy_expval)

    # Convert lists to numpy arrays for curve_fit
    noisy_array = np.array(training_data["noisy_expvals"])
    exact_array = np.array(training_data["exact_expvals"])

    # Perform the curve fit using the provided mapping (default: linear)
    popt, _ = curve_fit(fit_map, noisy_array, exact_array)

    import pdb
    pdb.set_trace()

    return training_data, popt


def density_matrix_circuit(circuit):
    circ = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        circ.add(gate)
    return circ