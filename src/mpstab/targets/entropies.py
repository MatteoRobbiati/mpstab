from itertools import product

import numpy as np
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian


def generate_pauli_strings(n):
    """Generate all possible N-bit Pauli strings."""
    pauli_operators = ["I", "X", "Y", "Z"]
    pauli_strings = ["".join(p) for p in product(pauli_operators, repeat=n)]
    return pauli_strings


def stabilizer_renyi_entropy(state: np.ndarray, alpha: int):
    """
    Compute the Stabilizer Renyi Entropy of at given `order` for a given `state`.
    Implementation inspired by Eq. (1) of https://arxiv.org/pdf/2207.13076.
    """
    nqubits = int(np.log2(len(state)))
    pauli_strings = generate_pauli_strings(nqubits)

    expval_to_2n = 0

    for pauli_string in pauli_strings:
        for i, pauli_op in enumerate(pauli_string):
            if i == 0:
                symbolic_obs = getattr(symbols, pauli_op)(i)
            else:
                symbolic_obs *= getattr(symbols, pauli_op)(i)
        obs = SymbolicHamiltonian(form=symbolic_obs)
        expval_to_2n += (obs.expectation(state) ** (2 * alpha)) / (2**nqubits)

    return (1.0 / (1 - alpha)) * np.log(expval_to_2n)
