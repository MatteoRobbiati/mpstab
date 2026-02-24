from qibo import construct_backend, set_backend
from qibo.hamiltonians import XXZ
from utils import DEFAULT_RNG_SEED, construct_test_circuit


def test_providing_backend_to_qibo():
    set_backend("mpstab")
    circ = construct_test_circuit(nqubits=8, rng_seed=DEFAULT_RNG_SEED)

    mpstab_ham = XXZ(nqubits=8, delta=0.5, dense=False)
    mpstab_exp = mpstab_ham.expectation(circ)

    np_backend = construct_backend(backend="numpy")
    np_ham = XXZ(nqubits=8, delta=0.5, backend=np_backend)
    np_exp = np_ham.expectation(circ)

    assert np_exp == mpstab_exp
