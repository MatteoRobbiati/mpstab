import numpy as np
import pytest
from qibo import construct_backend
from utils import (
    DEFAULT_ATOL,
    DEFAULT_RNG_SEED,
    construct_symbolic_hamiltonian,
    construct_test_circuit,
)


@pytest.mark.parametrize("nqubits", [5, 8, 11])
def test_providing_backend_to_qibo(nqubits):

    mpstab_backend = construct_backend("mpstab")
    np_backend = construct_backend("numpy")

    circ = construct_test_circuit(nqubits=nqubits, rng_seed=DEFAULT_RNG_SEED + nqubits)

    # Expectation value with mpstab
    mpstab_ham = construct_symbolic_hamiltonian(
        nqubits=nqubits,
        qibo_backend=mpstab_backend,
        rng_seed=DEFAULT_RNG_SEED + nqubits,
    )
    mpstab_exp = mpstab_ham.expectation(circ)

    # Expectation value with numpy
    np_ham = construct_symbolic_hamiltonian(
        nqubits=nqubits, qibo_backend=np_backend, rng_seed=DEFAULT_RNG_SEED + nqubits
    )
    np_exp = np_ham.expectation(circ)

    assert np.allclose(np_exp, mpstab_exp, atol=DEFAULT_ATOL)
