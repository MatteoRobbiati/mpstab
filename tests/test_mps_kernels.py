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


def _quimb_conjugation(engine, operator, rotations, max_bond):
    """The generic quimb route: two MPO-MPO products per rotation."""
    for generator, angle in rotations:
        rotation = engine._pauli_exp(generator, angle)
        rotation_dag = engine._pauli_exp(generator, -angle)
        operator = rotation_dag.apply(operator, compress=True, max_bond=max_bond)
        operator = operator.apply(rotation, compress=True, max_bond=max_bond)
    return operator


@pytest.mark.parametrize("max_bond", [None, 4])
def test_windowed_operator_conjugation_matches_quimb(max_bond):
    from scipy.linalg import expm
    from utils import pauli_matrix

    from mpstab.engines import QuimbEngine

    rng = np.random.default_rng(3)
    n, base = 6, "IZXIYI"
    rotations = [
        ("".join(rng.choice(list("IXYZ"), n)), rng.uniform(-1.5, 1.5))
        for _ in range(12)
    ] + [("IIXIII", 0.7), ("IIIIII", 0.3), ("ZIIIIY", -1.1)]
    engine = QuimbEngine()

    fast = engine.conjugate_operator_many(engine.pauli_mpo(base), rotations, max_bond)
    generic = _quimb_conjugation(engine, engine.pauli_mpo(base), rotations, max_bond)
    np.testing.assert_allclose(fast.to_dense(), generic.to_dense(), atol=1e-11)

    if max_bond is None:
        exact = pauli_matrix(base)
        for generator, angle in rotations:
            r = expm(-0.5j * angle * pauli_matrix(generator))
            exact = r.conj().T @ exact @ r
        np.testing.assert_allclose(fast.to_dense(), exact, atol=1e-11)


def test_single_conjugation_matches_many():
    from mpstab.engines import QuimbEngine

    engine = QuimbEngine()
    rotations = [("XYZI", 0.4), ("IZZX", -0.9), ("YIIY", 1.3)]
    one_by_one = engine.pauli_mpo("ZIXI")
    for generator, angle in rotations:
        one_by_one = engine.conjugate_operator(one_by_one, generator, angle, None)
    batched = engine.conjugate_operator_many(engine.pauli_mpo("ZIXI"), rotations, None)
    np.testing.assert_allclose(one_by_one.to_dense(), batched.to_dense(), atol=1e-12)


@pytest.mark.parametrize("shape", [(30, 50), (50, 30)])
def test_zero_cutoff_keeps_the_exact_rank(shape):
    """Gram eigenvalues at roundoff level are zeros, not directions to keep."""
    m = _complex(*shape, rank=7)
    for split in (kernels._split_left_isometry, kernels._split_right_isometry):
        a, b = split(m, 0.0, None)
        assert min(a.shape + b.shape) == 7
        np.testing.assert_allclose(a @ b, m, atol=1e-10)


def _degenerate_hermitian(order):
    """A Hermitian matrix with exactly repeated integer eigenvalues, like the
    Gram matrices of Clifford+T operators."""
    q, _ = np.linalg.qr(_complex(order, order))
    values = np.repeat(np.arange(1.0, order // 2 + 1), 2)[:order]
    return (q * values) @ q.conj().T


@pytest.mark.parametrize("order", [4, 2 * kernels._HEEVR_MIN_ORDER])
def test_eigh_is_orthonormal_on_a_degenerate_spectrum(order):
    hermitian = _degenerate_hermitian(order)
    values, vectors = kernels._eigh(hermitian)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(order), atol=1e-12)
    np.testing.assert_allclose(hermitian @ vectors, vectors * values, atol=1e-10)


def test_eigh_falls_back_when_heevr_loses_orthogonality(monkeypatch):
    """MRRR may return non-orthogonal eigenvectors on clustered eigenvalues;
    those must be recomputed, not handed to the splits as an isometry."""
    order = 2 * kernels._HEEVR_MIN_ORDER
    hermitian = _degenerate_hermitian(order)
    heevr = kernels._HEEVR

    def skewed(*args, **kwargs):
        values, vectors, m, isuppz, info = heevr(*args, **kwargs)
        vectors = vectors.copy()
        vectors[:, 1] += 1e-2 * vectors[:, 0]
        return values, vectors, m, isuppz, info

    monkeypatch.setattr(kernels, "_HEEVR", skewed)
    _, vectors = kernels._eigh(hermitian)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(order), atol=1e-12)
