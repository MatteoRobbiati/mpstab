import numpy as np
import pytest
from qibo import construct_backend
from qibo.hamiltonians import TFIM
from utils import DEFAULT_ATOL, construct_symbolic_hamiltonian, construct_test_circuit


@pytest.mark.parametrize("rng_seed", range(5))
@pytest.mark.parametrize("ham_choice", ["random"])
@pytest.mark.parametrize("nqubits", [8, 12])
def test_providing_backend_to_qibo(rng_seed, ham_choice, nqubits):

    mpstab_backend = construct_backend("mpstab")
    np_backend = construct_backend("numpy")

    circ = construct_test_circuit(nqubits=nqubits, rng_seed=rng_seed)

    if ham_choice == "random":
        # Expectation value with mpstab
        mpstab_ham = construct_symbolic_hamiltonian(
            nqubits=nqubits,
            qibo_backend=mpstab_backend,
            rng_seed=rng_seed,
        )

    elif ham_choice == "TFIM":
        mpstab_ham = TFIM(nqubits=nqubits, h=0.0, backend=mpstab_backend, dense=False)

    mpstab_exp = mpstab_ham.expectation(circ)

    if ham_choice == "random":
        # Expectation value with mpstab
        np_ham = construct_symbolic_hamiltonian(
            nqubits=nqubits,
            qibo_backend=np_backend,
            rng_seed=rng_seed,
        )
    elif ham_choice == "TFIM":
        np_ham = TFIM(nqubits=nqubits, h=1.0, backend=np_backend, dense=False)

    np_exp = np_ham.expectation(circ)

    assert np.allclose(np_exp, mpstab_exp, atol=DEFAULT_ATOL)


#
