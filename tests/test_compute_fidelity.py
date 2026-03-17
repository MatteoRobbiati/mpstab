import pytest
from numpy import round, allclose
from qibo import set_backend

from mpstab.engines import StimEngine, QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient
from utils import DEFAULT_ATOL
set_backend("numpy")

@pytest.mark.parametrize("nlayers", [2, 9, 15])
def test_fidelity_decreases_with_layers(nlayers):
    """Checks that fidelity decreases as layers increase and bond dimension decreases."""
    nqubits = 20
    max_bond_dim = 2
    observable = "Z" * nqubits
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    hs = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=max_bond_dim,
    )
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine())
    _, fidelity = hs.expectation(observable, return_fidelity=True)

    # Store fidelity for the first run
    if not hasattr(test_fidelity_decreases_with_layers, "last_fidelity"):
        test_fidelity_decreases_with_layers.last_fidelity = 1.0

    assert fidelity <= test_fidelity_decreases_with_layers.last_fidelity
    test_fidelity_decreases_with_layers.last_fidelity = fidelity

@pytest.mark.parametrize("max_bond_dim", [10, 4, 2])
def test_fidelity_decreases_with_bond_dim(max_bond_dim):
    """Checks that fidelity decreases as layers increase and bond dimension decreases."""
    nqubits = 20
    nlayers = 5
    observable = "Z" * nqubits
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    hs = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=max_bond_dim,
    )
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine())
    _, fidelity = hs.expectation(observable, return_fidelity=True)

    # Store fidelity for the first run
    if not hasattr(test_fidelity_decreases_with_bond_dim, "last_fidelity"):
        test_fidelity_decreases_with_bond_dim.last_fidelity = 1.0

    assert fidelity <= test_fidelity_decreases_with_bond_dim.last_fidelity
    test_fidelity_decreases_with_bond_dim.last_fidelity = fidelity

@pytest.mark.parametrize("nqubits",[4,7,10,15])
def test_fidelity_faithfull(nqubits):
    nlayers = 5
    max_bond_dim = 4
    observable = "Z" * nqubits
    ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers)

    hs = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=max_bond_dim,
    )
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine())    
    _, fidelity = hs.expectation(observable, return_fidelity=True)

    fidelity_check = hs.truncation_fidelity()
    fidelity_pure = hs.truncation_fidelity_pure_tn
    assert allclose(fidelity,fidelity_check,atol=DEFAULT_ATOL)
    assert round(fidelity, decimals=7) >= round(fidelity_pure, decimals=7)
