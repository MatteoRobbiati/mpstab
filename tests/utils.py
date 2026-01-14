from qibo import hamiltonians, symbols

DEFAULT_REPLACEMENT_PROBABILITY = 0.75
DEFAULT_MAX_BD = 128


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
