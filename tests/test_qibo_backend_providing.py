import numpy as np
import pytest
from qibo import construct_backend
from utils import DEFAULT_ATOL, construct_symbolic_hamiltonian, construct_test_circuit


@pytest.mark.parametrize("rng_seed", range(5))
@pytest.mark.parametrize("nqubits", [8, 12])
def test_providing_backend_to_qibo(rng_seed, nqubits):
    """The "mpstab" qibo backend gives the same expectation as numpy."""
    circ = construct_test_circuit(nqubits=nqubits, rng_seed=rng_seed)
    expectations = [
        construct_symbolic_hamiltonian(
            nqubits=nqubits,
            qibo_backend=construct_backend(backend),
            rng_seed=rng_seed,
        ).expectation(circ)
        for backend in ("mpstab", "numpy")
    ]
    assert np.allclose(*expectations, atol=DEFAULT_ATOL)
