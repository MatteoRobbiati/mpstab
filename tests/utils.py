import random

import numpy as np
from qibo import Circuit, gates, hamiltonians, symbols
from qibo.backends import Backend, get_backend

DEFAULT_REPLACEMENT_PROBABILITY = 0.75
DEFAULT_MAX_BD = 128
DEFAULT_RNG_SEED = 42
DEFAULT_ATOL = 1e-6


def obs_string_to_qibo_hamiltonian(observable: str) -> hamiltonians.SymbolicHamiltonian:
    """
    Convert a string representation of a Pauli observable to a Qibo symbolic Hamiltonian.

    Args:
        observable (str): A string representing the Pauli observable, e.g., "XZIY".

    Returns:
        hamiltonians.SymbolicHamiltonian: The corresponding Qibo symbolic Hamiltonian.
    """
    form = 1
    for i, pauli in enumerate(observable):
        form *= getattr(symbols, pauli)(i)
    ham = hamiltonians.SymbolicHamiltonian(form=form)
    return ham


def set_rng_seed(seed: int = DEFAULT_RNG_SEED):
    """Set all the RNG seeds."""
    backend = get_backend()
    backend.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def construct_test_circuit(
    nqubits: int = 5,
    rng_seed: int = 42,
    is_clifford: bool = False,
) -> Circuit:
    set_rng_seed(rng_seed)

    circ = Circuit(nqubits)
    [circ.add(gates.H(q)) for q in range(nqubits)]
    for q in range(nqubits):
        if np.random.uniform(0, 1) < 0.5:
            circ.add(gates.CZ(q % nqubits, (q + 1) % nqubits))
        if np.random.uniform(0, 1) < 0.5:
            if is_clifford:
                theta = np.random.choice(np.arange(-2, 3)) * (np.pi / 2)
            else:
                theta = np.random.uniform(-np.pi, np.pi)
            circ.add(gates.RY(q=q, theta=theta))

    return circ


def ghz_circuit(nqubits):
    """Prepare the GHZ circuit."""
    circ = Circuit(nqubits)
    circ.add(gates.H(0))
    for q in range(nqubits - 1):
        circ.add(gates.CNOT(q, q + 1))
    return circ


def expectation_with_qibo(mpstab_ansatz, observable_str):
    """
    Take an mpstab ansatz and an observable string and compute the
    corresponding expectation value using qibo facilities.
    """
    if len(observable_str) >= 20:
        raise ValueError(
            f"Please consider lighten the test, this function is using statevector simulation, which can be really slow for the provided {len(observable_str)} problem"
        )

    qibo_ham = obs_string_to_qibo_hamiltonian(observable_str)
    expval = qibo_ham.expectation_from_state(mpstab_ansatz.circuit().state())

    return expval


def construct_symbolic_hamiltonian(
    nqubits: int, rng_seed: int = DEFAULT_RNG_SEED, n_terms: int = 5
):
    """Construct a random symbolic hamiltonian."""

    set_rng_seed(rng_seed)

    from qibo.symbols import X, Y, Z

    symbols = [X, Y, Z]
    ham_form = 0

    for _ in range(n_terms):
        coeff = np.random.uniform(0.5, 2.0)
        # Pick 1 or 2 random qubits for this term
        target_qubits = np.random.choice(range(nqubits), size=2, replace=False)
        paulis = np.random.choice(symbols, size=2)

        # Construct term: e.g. 1.2 * X(0) * Z(3)
        term = (
            coeff * paulis[0](int(target_qubits[0])) * paulis[1](int(target_qubits[1]))
        )
        ham_form += term

    # Add a constant shift to test completeness
    constant_shift = 0.5
    ham_form += constant_shift

    return hamiltonians.SymbolicHamiltonian(nqubits=nqubits, form=ham_form)
