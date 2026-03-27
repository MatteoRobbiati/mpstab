import pytest

from mpstab.engines import StimEngine, QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_backend_support(backend):
    
    nqubits=9
    nlayers=3
    observable = "ZX" * int(nqubits/2)

    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)
    hs = HSMPO(ansatz)

    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(backend=backend))

    expectation = hs.expectation(observable)

    assert expectation is not None