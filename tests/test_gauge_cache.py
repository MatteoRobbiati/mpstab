"""The numpy fast path of QuimbEngine caches the MPS gauge on the state; these
are cases where a stale cache used to give silently wrong expectations."""

import numpy as np
import pytest
import quimb.tensor as qtn
from utils import pauli_matrix

from mpstab.engines import QuimbEngine

L = 8
PAULI = "IIZZIIII"
G = np.array([[1, 0.8], [0, 0.3]])  # non-unitary, so it changes <PAULI>


def _exact(mps):
    psi = mps.to_dense().ravel()
    op = pauli_matrix(PAULI)
    return (psi.conj() @ op @ psi).real / (psi.conj() @ psi).real


@pytest.fixture
def engine():
    return QuimbEngine()


@pytest.fixture
def mps(engine):
    state = qtn.MPS_rand_state(L, bond_dim=8, seed=1, dtype=complex)
    engine.pauli_expvals(state, [PAULI])  # caches the gauge
    return state


def _write_gate_in_place(mps, site):
    data = mps[site].data
    data[...] = np.einsum("pq,...q->...p", G, data)


def test_in_place_write_raises(engine, mps):
    before = engine.pauli_expvals(mps, [PAULI])[0]
    with pytest.raises(ValueError, match="read-only"):
        _write_gate_in_place(mps, 3)
    np.testing.assert_allclose(before, _exact(mps), atol=1e-12)


def test_in_place_write_on_copy_source_raises(engine, mps):
    copied = engine.copy_state(mps)
    before = engine.pauli_expvals(copied, [PAULI])[0]
    with pytest.raises(ValueError, match="read-only"):
        _write_gate_in_place(mps, 3)
    after = engine.pauli_expvals(copied, [PAULI])[0]
    np.testing.assert_allclose(after, before, atol=1e-14)
    np.testing.assert_allclose(after, _exact(copied), atol=1e-12)


def test_extra_tensor_is_not_ignored(engine, mps):
    stale = engine.pauli_expvals(mps, [PAULI])[0]
    mps.gate_(G, 3)  # contract=False: a 9th tensor, site arrays untouched
    exact = _exact(mps)
    assert not np.isclose(stale, exact)
    try:
        value = engine.pauli_expvals(mps, [PAULI])[0]
    except ValueError:
        return
    np.testing.assert_allclose(value, exact, atol=1e-12)
