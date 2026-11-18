"""The BLAS-heavy factorizations behind the windowed MPS kernels: Cholesky QR
and the Gram-matrix truncation must agree with Householder QR and the SVD."""

import numpy as np
import pytest

from mpstab.engines.tensor_networks import _mps_kernels as kernels

RNG = np.random.default_rng(7)


def _complex(rows, cols, rank=None):
    m = RNG.normal(size=(rows, cols)) + 1j * RNG.normal(size=(rows, cols))
    if rank is not None:
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        m = (u[:, :rank] * s[:rank]) @ vh[:rank]
    return m


def _decaying(rows, cols, decay=0.2):
    """A matrix with geometrically decaying singular values, like MPS bonds."""
    u, _ = np.linalg.qr(_complex(rows, rows))
    v, _ = np.linalg.qr(_complex(cols, cols))
    k = min(rows, cols)
    return (u[:, :k] * np.exp(-decay * np.arange(k))) @ v[:, :k].conj().T


@pytest.mark.parametrize("shape", [(40, 40), (64, 20), (12, 1)])
def test_cholesky_qr_matches_householder(shape):
    m = _complex(*shape)
    q, r = kernels._cholesky_qr(m)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(shape[1]), atol=1e-12)
    np.testing.assert_allclose(q @ r, m, atol=1e-12)
    np.testing.assert_allclose(np.triu(r), r)


def test_cholesky_qr_recovers_orthogonality_of_ill_conditioned_input():
    m = _decaying(60, 30, decay=0.5)  # condition number ~ e^15 ~ 3e6
    q, r = kernels._cholesky_qr(m)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(30), atol=1e-12)
    np.testing.assert_allclose(q @ r, m, atol=1e-12)


@pytest.mark.parametrize("m", [_complex(10, 20), _complex(20, 10, rank=4)])
def test_cholesky_qr_declines_wide_or_rank_deficient(m):
    assert kernels._cholesky_qr(m) is None
    q, r = kernels._qr(m)  # falls back to Householder
    np.testing.assert_allclose(q @ r, m, atol=1e-12)


@pytest.mark.parametrize("shape", [(30, 50), (50, 30), (40, 40)])
@pytest.mark.parametrize("max_bond", [None, 12])
def test_splits_match_svd_truncation(shape, max_bond):
    m = _decaying(*shape)
    cutoff = 1e-10
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    k = kernels.n_kept(s, cutoff, max_bond)
    best = (u[:, :k] * s[:k]) @ vh[:k]

    iso, rest = kernels._split_left_isometry(m, cutoff, max_bond)
    assert iso.shape[1] == k
    np.testing.assert_allclose(iso.conj().T @ iso, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(iso @ rest, best, atol=1e-10)

    rest, iso = kernels._split_right_isometry(m, cutoff, max_bond)
    assert iso.shape[0] == k
    np.testing.assert_allclose(iso @ iso.conj().T, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(rest @ iso, best, atol=1e-10)


def test_split_of_zero_matrix_keeps_one_value():
    iso, rest = kernels._split_left_isometry(np.zeros((6, 9), complex), 1e-10, None)
    assert iso.shape == (6, 1)
    np.testing.assert_allclose(iso @ rest, 0.0)
