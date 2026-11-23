"""The numpy fast path of QuimbEngine caches the MPS gauge on the state; these
are cases where a stale cache used to give silently wrong expectations."""

import copy

import numpy as np
import pytest
import quimb.tensor as qtn
from qibo import Circuit

from mpstab.pauli import pauli_matrix

L = 8
PAULI = "IIZZIIII"
G = np.array([[1, 0.8], [0, 0.3]])  # non-unitary, so it changes <PAULI>


def _exact(mps):
    psi = mps.to_dense().ravel()
    op = pauli_matrix(PAULI)
    return (psi.conj() @ op @ psi).real / (psi.conj() @ psi).real


@pytest.fixture
def mps(quimb_engine):
    state = qtn.MPS_rand_state(L, bond_dim=8, seed=1, dtype=complex)
    quimb_engine.pauli_expvals(state, [PAULI])  # caches the gauge
    return state


def _write_gate_in_place(mps, site):
    data = mps[site].data
    data[...] = np.einsum("pq,...q->...p", G, data)


def test_in_place_write_raises(quimb_engine, mps):
    before = quimb_engine.pauli_expvals(mps, [PAULI])[0]
    with pytest.raises(ValueError, match="read-only"):
        _write_gate_in_place(mps, 3)
    np.testing.assert_allclose(before, _exact(mps), atol=1e-12)


def test_in_place_write_on_copy_source_raises(quimb_engine, mps):
    copied = quimb_engine.copy_state(mps)
    before = quimb_engine.pauli_expvals(copied, [PAULI])[0]
    with pytest.raises(ValueError, match="read-only"):
        _write_gate_in_place(mps, 3)
    after = quimb_engine.pauli_expvals(copied, [PAULI])[0]
    np.testing.assert_allclose(after, before, atol=1e-14)
    np.testing.assert_allclose(after, _exact(copied), atol=1e-12)


def test_extra_tensor_is_not_ignored(quimb_engine, mps):
    stale = quimb_engine.pauli_expvals(mps, [PAULI])[0]
    mps.gate_(G, 3)  # contract=False: a 9th tensor, site arrays untouched
    exact = _exact(mps)
    assert not np.isclose(stale, exact)
    try:
        value = quimb_engine.pauli_expvals(mps, [PAULI])[0]
    except ValueError:
        return
    np.testing.assert_allclose(value, exact, atol=1e-12)


def _evolved_state(engine, n=4):
    """An MPS built and rotated through the engine, its gauge cached."""
    state = engine.build_circuit_mps(n, Circuit(n))
    for generator, angle in [("XXII", 0.7), ("IYZI", 1.1), ("IIXY", 0.4)]:
        engine.pauli_rot(state, generator, angle, max_bond_dimension=None)
    return state


@pytest.mark.parametrize("site", range(4))
def test_squared_norm_of_deep_copy_written_in_place(quimb_engine, site):
    state = _evolved_state(quimb_engine)
    copied = copy.deepcopy(state)
    copied[site].data[...] *= 2.0
    np.testing.assert_allclose(
        quimb_engine.squared_norm(copied),
        copied.norm(squared=True).real,
        atol=1e-12,
    )


@pytest.mark.parametrize("site", range(4))
def test_pauli_expvals_of_deep_copy_written_in_place(quimb_engine, site):
    state = _evolved_state(quimb_engine)
    copied = copy.deepcopy(state)
    copied[site].data[...] = np.einsum("pq,...q->...p", G, copied[site].data)
    paulis = ["ZZII", "IXYI", "IIZX", "YIIZ"]
    psi = copied.to_dense().ravel()
    exact = [
        (psi.conj() @ pauli_matrix(p) @ psi).real / (psi.conj() @ psi).real
        for p in paulis
    ]
    np.testing.assert_allclose(
        quimb_engine.pauli_expvals(copied, paulis), exact, atol=1e-12
    )
