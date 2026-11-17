import numpy as np
import pytest
from utils import folded_mpo, set_rng_seed


@pytest.fixture(autouse=True)
def _seed():
    """Every test starts from the same RNG state, whatever ran before it."""
    set_rng_seed()


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
