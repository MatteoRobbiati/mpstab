import copy
import random
from typing import Optional

import numpy as np
import tqdm
from qibo import Circuit, get_backend, hamiltonians, symbols
from qibo.noise import NoiseModel
from scipy.optimize import curve_fit

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import Ansatz


def TNCDR(
    observable: str,
    ansatz: Ansatz,
    noise_model: NoiseModel,
    replacement_probability: float,
    initial_state: Circuit = None,
    replacement_method: str = "closest",
    ncircuits: int = 50,
    nshots: Optional[int] = None,  # TODO: discuss it
    random_seed: int = 42,
    fit_map=lambda x, a, b: a * x + b,
    expval_threshold: float = 1e-7,
    max_bond_dimension: Optional[int] = None,
):

    # Fix the RNG seed for reproducibility
    random.seed(random_seed)
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

    for i in tqdm.tqdm(range(ncircuits)):
        # Construct the hybrid surrogate
        evo = HSMPO(
            ansatz=ansatz,
            initial_state=initial_state,
            max_bond_dimension=max_bond_dimension,
        )

        # Exact expval from surrogate
        exact_expval, partitions = evo.expectation_from_partition(
            replacement_probability=replacement_probability,
            observable=observable,
            return_partitions=True,
            replacement_method=replacement_method,
        )

        # TODO: discuss this
        if np.abs(exact_expval) < expval_threshold:
            continue

        # TODO: return the mitigated value as well (as it is done in Qibo)
        sampled_circuit = density_matrix_circuit(partitions["full_circuit"])
        if initial_state is not None:
            density_init_state = density_matrix_circuit(copy.deepcopy(initial_state))
            initialised_sampled_circuit = density_init_state + sampled_circuit
        else:
            initialised_sampled_circuit = sampled_circuit

        noisy_init_sampled_circuit = noise_model.apply(initialised_sampled_circuit)

        noisy_expval = ham.expectation(noisy_init_sampled_circuit().state())

        training_data["exact_expvals"].append(exact_expval)
        training_data["noisy_expvals"].append(noisy_expval)

    # Convert lists to numpy arrays for curve_fit
    noisy_array = np.array(training_data["noisy_expvals"])
    exact_array = np.array(training_data["exact_expvals"])

    # Perform the curve fit using the provided mapping (default: linear)
    popt, _ = curve_fit(fit_map, noisy_array, exact_array)

    return training_data, popt


def density_matrix_circuit(circuit):
    """Helper method to convert a circuit into its correspondent with `density_matrix=True`."""
    circ = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        circ.add(gate)
    return circ
