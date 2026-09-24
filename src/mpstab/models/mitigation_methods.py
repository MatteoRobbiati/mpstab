"""Error mitigation built on the surrogate."""

import copy
import random
from typing import Optional

import numpy as np
import tqdm
from qibo import Circuit, get_backend
from qibo.noise import NoiseModel
from scipy.optimize import curve_fit

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import pauli_string_to_hamiltonian
from mpstab.models.ansatze import Ansatz


def TNCDR(
    observable: str,
    ansatz: Ansatz,
    noise_model: NoiseModel,
    replacement_probability: float,
    initial_state: Circuit = None,
    replacement_method: str = "closest",
    ncircuits: int = 50,
    nshots: Optional[int] = None,
    random_seed: int = 42,
    fit_map=lambda x, a, b: a * x + b,
    expval_threshold: float = 1e-7,
    max_bond_dimension: Optional[int] = None,
):
    """
    Tensor-network Clifford data regression.

    Samples ``ncircuits`` lower-magic circuits from the ansatz, computes each
    one's exact expectation value on the surrogate and its noisy value under
    ``noise_model``, then fits ``fit_map`` from noisy to exact. The fit is what
    later corrects a noisy hardware measurement.

    Args:
        observable: the Pauli string to mitigate.
        ansatz: the circuit to sample lower-magic variants of.
        noise_model: the noise to apply when computing noisy values.
        replacement_probability: chance of replacing each magic gate with a
            Clifford one.
        initial_state: state-preparation circuit; defaults to ``|0...0>``.
        replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
        ncircuits: training circuits to sample.
        nshots: shot budget for the noisy values; ``None`` evaluates them exactly.
        random_seed: RNG seed.
        fit_map: the regression model, linear by default.
        expval_threshold: skip training points whose exact value is smaller than
            this, since they carry almost no signal to fit against.
        max_bond_dimension: MPS truncation cap for the surrogate.

    Returns:
        ``(training_data, popt)``, the sampled noisy/exact pairs and the fitted
        parameters of ``fit_map``.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    backend = get_backend()
    backend.set_seed(random_seed)

    ham = pauli_string_to_hamiltonian(observable)

    training_data = {
        "noisy_expvals": [],
        "exact_expvals": [],
    }

    for _ in tqdm.tqdm(range(ncircuits)):
        surrogate = HSMPO(
            ansatz=ansatz,
            initial_state=initial_state,
            max_bond_dimension=max_bond_dimension,
        )
        exact_expval, partitions = surrogate.expectation_from_partition(
            replacement_probability=replacement_probability,
            observable=observable,
            return_partitions=True,
            replacement_method=replacement_method,
        )
        if np.abs(exact_expval) < expval_threshold:
            continue

        sampled_circuit = density_matrix_circuit(partitions["full_circuit"])
        if initial_state is not None:
            sampled_circuit = (
                density_matrix_circuit(copy.deepcopy(initial_state)) + sampled_circuit
            )
        noisy_circuit = noise_model.apply(sampled_circuit)
        noisy_expval = ham.expectation(noisy_circuit().state())

        training_data["exact_expvals"].append(exact_expval)
        training_data["noisy_expvals"].append(noisy_expval)

    popt, _ = curve_fit(
        fit_map,
        np.array(training_data["noisy_expvals"]),
        np.array(training_data["exact_expvals"]),
    )
    return training_data, popt


def density_matrix_circuit(circuit):
    """The same circuit rebuilt with ``density_matrix=True``, so noise can act on it."""
    density_circuit = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        density_circuit.add(gate)
    return density_circuit
