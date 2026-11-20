"""Fixtures shared by the whole test suite."""

import numpy as np
import pytest
from qibo import set_backend
from utils import folded_mpo, set_rng_seed

from mpstab.engines import (
    NativeStabilizersEngine,
    NativeTensorNetworkEngine,
    QuimbEngine,
    StimEngine,
)


def pytest_configure(config):
    """Run every test on qibo's numpy backend, whatever other backends are installed."""
    set_backend("numpy")


@pytest.fixture(autouse=True)
def _seed():
    """Every test starts from the same RNG state, whatever ran before it."""
    set_rng_seed()


@pytest.fixture
def stim_engine():
    return StimEngine()


@pytest.fixture
def quimb_engine():
    return QuimbEngine()


@pytest.fixture(params=[StimEngine, NativeStabilizersEngine], ids=["stim", "native"])
def stab_engine(request):
    """Each stabilizers engine in turn."""
    return request.param()


@pytest.fixture(
    params=[QuimbEngine, NativeTensorNetworkEngine], ids=["quimb", "native"]
)
def tn_engine(request):
    """Each tensor-network engine in turn."""
    return request.param()


@pytest.fixture
def scrambled_mpo():
    """
    A 4-qubit Hermitian operator scrambled enough to have a nontrivial bond
    dimension and several sizeable Pauli terms: conjugation by real-angle Pauli
    rotations keeps it Hermitian throughout.
    """
    rng = np.random.default_rng(3)
    generators = ["YYII", "IXZY", "ZIXY", "YZYX"]
    angles = [rng.uniform(0.3, 1.5) for _ in generators]
    return folded_mpo("ZZZI", generators, angles)


@pytest.fixture
def scrambled_sites(scrambled_mpo):
    """:func:`scrambled_mpo` as the site arrays an engine exports."""
    return QuimbEngine().operator_site_arrays(scrambled_mpo)
