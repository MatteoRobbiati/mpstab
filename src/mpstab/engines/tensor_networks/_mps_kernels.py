"""Backend-agnostic MPS kernels for the operations HSMPO spends its time in.

Every routine works on a list ``A`` of site tensors shaped ``(left, phys, right)``
kept in *mixed canonical form* around an orthogonality center ``c``: sites left
of ``c`` are left-isometries, sites right of it right-isometries. That gauge is
what makes both kernels local:

- :func:`pauli_rotation` applies ``exp(-i angle/2 P)`` touching only the sites
  between the first and last non-identity letter of ``P`` (plus the sites the
  center has to travel over to get there), instead of re-sweeping the chain;
- :func:`pauli_expectation` contracts ``<psi|P|psi>`` over the same window, since
  the canonical sites outside it contract to identities.

Truncation follows quimb's default ``"rsum2"`` rule, so the states match what
``MatrixProductState.gate_with_mpo`` produces.

The kernels only shuffle and contract arrays; the factorizations and the array
module come from a :class:`Linalg`. :data:`NUMPY` is the CPU one, built on raw
LAPACK calls; a GPU engine passes its own (CuPy arrays, cuTensorNet
factorizations), and every kernel runs unchanged on its arrays.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from mpstab.pauli import PAULI_MATRICES


def bounds(pauli: str) -> tuple[int, int] | None:
    """First and last non-identity sites, or ``None`` for the identity."""
    stripped = pauli.lstrip("I")
    if not stripped:
        return None
    return len(pauli) - len(stripped), len(pauli.rstrip("I")) - 1


def apply_pauli(tensor, label: str, xp=np):
    """``label`` applied to the physical index of a ``(l, p, r)`` tensor."""
    if label == "I":
        return tensor
    if label == "X":
        return tensor[:, ::-1, :]
    out = xp.empty(tensor.shape, dtype=np.result_type(tensor.dtype, np.complex64))
    if label == "Z":
        out[:, 0, :] = tensor[:, 0, :]
        out[:, 1, :] = -tensor[:, 1, :]
    elif label == "Y":
        out[:, 0, :] = -1j * tensor[:, 1, :]
        out[:, 1, :] = 1j * tensor[:, 0, :]
    else:
        raise ValueError(f"Invalid Pauli label {label!r}.")
    return out


#: Largest ``|Q^H Q - I|`` entry a Cholesky QR may leave before it is rejected.
_ORTHOGONALITY_TOL = 1e-12

# Raw complex LAPACK routines: scipy.linalg's wrappers validate and copy their
# inputs, which costs more than the factorization itself on the small matrices
# of a low bond dimension.
_POTRF, _TRTRS, _HEEVD, _HEEVR, _HEEVR_LWORK = scipy.linalg.lapack.get_lapack_funcs(
    ("potrf", "trtrs", "heevd", "heevr", "heevr_lwork"), dtype=np.complex128
)

#: Optimal ``heevr`` workspace sizes per matrix order: its default is the
#: minimum, which runs unblocked and is markedly slower on large matrices.
_HEEVR_WORKSPACE: dict = {}

#: Matrix order above which ``heevr`` (MRRR) beats ``heevd`` (divide and
#: conquer); below it ``heevd`` is as fast or faster.
_HEEVR_MIN_ORDER = 128

#: Largest ``|V^H V - I|`` entry accepted from ``heevr`` eigenvectors.
_EIGENVECTOR_TOL = 1e-10


def _svd(matrix: np.ndarray):
    """Thin SVD, falling back to the slower but sturdier ``gesvd`` driver."""
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(
            matrix, full_matrices=False, lapack_driver="gesvd", check_finite=False
        )


def _cholesky_qr(matrix: np.ndarray):
    """
    Thin QR of a tall matrix by Cholesky QR, repeated once if the first pass
    left ``Q`` short of orthonormal (CholQR2), or ``None`` when the matrix is
    too ill-conditioned for it (Cholesky fails or ``Q`` stays non-orthonormal).

    Its cost is a few matrix products, which BLAS runs far faster than the
    Householder QR of LAPACK on the small, tall matrices MPS sweeps produce.
    """
    rows, cols = matrix.shape
    if rows < cols:
        return None
    identity = np.eye(cols)
    q, r = np.asarray(matrix, dtype=np.complex128), None
    gram = q.conj().T @ q
    for _ in range(2):
        step, info = _POTRF(gram, lower=0, clean=1)
        if info != 0:
            return None
        # Q = M R^-1, from R^T Q^T = M^T.
        solution, info = _TRTRS(step, q.T, lower=0, trans=1)
        if info != 0:
            return None
        q = solution.T
        r = step if r is None else step @ r
        gram = q.conj().T @ q
        if np.abs(gram - identity).max() < _ORTHOGONALITY_TOL:
            return q, r
    return None


def _qr(matrix: np.ndarray):
    """Thin QR: Cholesky QR when the matrix allows it, Householder otherwise."""
    result = _cholesky_qr(matrix)
    return np.linalg.qr(matrix) if result is None else result


def n_kept(s: np.ndarray, cutoff: float, max_bond: int | None) -> int:
    """
    How many singular values survive truncation.

    Mirrors quimb's ``"rsum2"`` mode: drop the smallest values for as long as
    their summed squares stay within ``cutoff`` times the total, then cap at
    ``max_bond``. At least one value is always kept.
    """
    keep = s.size
    if cutoff > 0.0 and keep > 1:
        tail = np.cumsum((s * s)[::-1])[::-1]
        keep = int(np.count_nonzero(tail > cutoff * tail[0]))
    if max_bond is not None and max_bond > 0:
        keep = min(keep, max_bond)
    return max(keep, 1)


def _heevd(hermitian: np.ndarray):
    """Eigendecomposition by LAPACK's divide-and-conquer driver (``heevd``)."""
    values, vectors, info = _HEEVD(hermitian, compute_v=1, lower=0)
    if info != 0:
        raise np.linalg.LinAlgError(f"heevd failed with info={info}")
    return values, vectors


def _eigh(hermitian: np.ndarray, heevr=None):
    """
    Eigendecomposition of a Hermitian matrix, ascending, with orthonormal
    eigenvectors.

    Small matrices go to the divide-and-conquer driver (``heevd``), larger
    ones to the MRRR driver (``heevr``), the faster of the two above
    :data:`_HEEVR_MIN_ORDER`. MRRR can return eigenvectors that are far from
    orthogonal on (nearly) degenerate eigenvalues, which the Gram matrices of
    Clifford+T operators have exactly, and how far depends on the LAPACK
    build. Its eigenvectors are therefore checked, and recomputed by
    ``heevd`` when they are not orthonormal.

    Args:
        hermitian: the matrix.
        heevr: the MRRR routine, with LAPACK's ``heevr`` signature; ``None``
            for scipy's. Given only to check the fallback on a routine that
            misbehaves.
    """
    hermitian = np.asarray(hermitian, dtype=np.complex128)
    order = hermitian.shape[0]
    if order <= _HEEVR_MIN_ORDER:
        return _heevd(hermitian)
    workspace = _HEEVR_WORKSPACE.get(order)
    if workspace is None:
        lwork, lrwork, liwork, _ = _HEEVR_LWORK(order)
        workspace = _HEEVR_WORKSPACE[order] = dict(
            lwork=int(lwork.real), lrwork=int(lrwork), liwork=int(liwork)
        )
    heevr = _HEEVR if heevr is None else heevr
    values, vectors, _, _, info = heevr(hermitian, compute_v=1, lower=0, **workspace)
    if info != 0 or (
        np.abs(vectors.conj().T @ vectors - np.eye(order)).max() > _EIGENVECTOR_TOL
    ):
        return _heevd(hermitian)
    return values, vectors


def _numerical_rank(singular_values: np.ndarray) -> int:
    """
    How many singular values (descending) stand above roundoff: the squared
    ones above ``order * eps`` of the largest, the threshold of
    :func:`_gram_rank`, with ``eps`` of the values' own precision. At least
    one.
    """
    squares = singular_values * singular_values
    if not squares.size:
        return 1
    # The precision of the values themselves: float32 for complex64 sites.
    eps = np.finfo(np.result_type(singular_values.dtype, np.float32)).eps
    noise = squares.size * eps * squares[0]
    return max(int(np.count_nonzero(squares > noise)), 1)


def _gram_rank(values: np.ndarray, cutoff: float, max_bond: int | None) -> int:
    """
    :func:`n_kept` for the descending eigenvalues of a Gram matrix, the
    squared singular values. Eigenvalues within roundoff of zero, below
    ``order * eps`` of the largest, are dropped even at ``cutoff=0``: the Gram
    matrix cannot tell them from an exact zero, and keeping them would only
    inflate the bond with noise directions.
    """
    values = np.clip(values, 0.0, None)
    noise = values.size * np.finfo(float).eps * (values[0] if values.size else 0.0)
    rank = max(int(np.count_nonzero(values > noise)), 1)
    return min(n_kept(np.sqrt(values), cutoff, max_bond), rank)


def _split_left_isometry(matrix: np.ndarray, cutoff: float, max_bond: int | None):
    """
    Truncated ``matrix ~ iso @ rest`` with ``iso`` orthonormal columns: the
    rank-``k`` SVD truncation ``U_k (S_k V_k^H)``, up to a gauge on the bond.

    The singular values come from the eigendecomposition of the Gram matrix
    on the smaller side: about a fifth cheaper than LAPACK's SVD on the
    square bond-doubled matrices of a rotation window, more on oblong ones.
    Squaring the matrix only blurs singular values below ``sqrt(eps)`` of the
    largest, the ones the ``"rsum2"`` cutoff drops anyway. When the Gram matrix sits on
    the side of ``rest``, ``iso`` is re-orthonormalized by Cholesky QR, and
    anything too ill-conditioned for that goes through a plain SVD.
    """
    rows, cols = matrix.shape
    try:
        if rows <= cols:
            # M M^H = U S^2 U^H, so U_k is the isometry outright.
            values, vectors = _eigh(matrix @ matrix.conj().T)
            k = _gram_rank(values[::-1], cutoff, max_bond)
            iso = vectors[:, ::-1][:, :k]
            return iso, iso.conj().T @ matrix
        # M^H M = V S^2 V^H; M V_k = U_k S_k still needs orthonormalizing.
        values, vectors = _eigh(matrix.conj().T @ matrix)
        k = _gram_rank(values[::-1], cutoff, max_bond)
        kept = vectors[:, ::-1][:, :k]
        factors = _cholesky_qr(matrix @ kept)
        if factors is not None:
            iso, r = factors
            return iso, r @ kept.conj().T
    except np.linalg.LinAlgError:
        pass
    u, s, vh = _svd(matrix)
    k = n_kept(s, cutoff, max_bond)
    return u[:, :k], s[:k, None] * vh[:k]


def _split_right_isometry(matrix: np.ndarray, cutoff: float, max_bond: int | None):
    """Truncated ``matrix ~ rest @ iso`` with ``iso`` orthonormal rows."""
    iso, rest = _split_left_isometry(matrix.conj().T, cutoff, max_bond)
    return rest.conj().T, iso.conj().T


#: quimb's default singular-value cutoff (``"rsum2"`` mode) when compressing.
DEFAULT_SVD_CUTOFF = 1e-10


class Linalg:
    """
    The array module and factorizations the kernels run on.

    Subclasses provide :meth:`qr` and :meth:`svd`; the truncated splits are
    built on them with the ``"rsum2"`` rule of :func:`n_kept`, so every backend
    truncates exactly alike.
    """

    #: The array module (numpy, cupy, ...): creation, concatenation and
    #: contraction functions are taken from it.
    xp = np

    def asarray(self, array):
        """``array`` on this backend."""
        return self.xp.asarray(array)

    def to_numpy(self, array) -> np.ndarray:
        """A host (numpy) copy of a backend array."""
        return np.asarray(array)

    def qr(self, matrix):
        """Thin QR ``matrix = q @ r``, ``q`` with orthonormal columns."""
        raise NotImplementedError

    def svd(self, matrix):
        """Thin SVD ``(u, s, vh)``, singular values descending."""
        raise NotImplementedError

    def split_left_isometry(self, matrix, cutoff: float, max_bond: int | None):
        """
        Truncated ``matrix ~ iso @ rest`` with ``iso`` orthonormal columns: the
        rank-``k`` SVD truncation ``U_k (S_k V_k^H)``, ``k`` from :func:`n_kept`.
        Singular values within roundoff of zero are dropped even at
        ``cutoff=0``, as :func:`_gram_rank` does on the NumPy path, so both
        paths keep the same bonds.
        """
        u, s, vh = self.svd(matrix)
        values = self.to_numpy(s)
        k = min(n_kept(values, cutoff, max_bond), _numerical_rank(values))
        return u[:, :k], s[:k, None] * vh[:k]

    def split_right_isometry(self, matrix, cutoff: float, max_bond: int | None):
        """Truncated ``matrix ~ rest @ iso`` with ``iso`` orthonormal rows."""
        iso, rest = self.split_left_isometry(matrix.conj().T, cutoff, max_bond)
        return rest.conj().T, iso.conj().T


class NumpyLinalg(Linalg):
    """
    CPU factorizations: Cholesky QR and Gram-matrix truncation on BLAS, with
    Householder QR and the SVD as fallbacks (:func:`_qr`,
    :func:`_split_left_isometry`).
    """

    xp = np

    def qr(self, matrix):
        return _qr(matrix)

    def svd(self, matrix):
        return _svd(matrix)

    def split_left_isometry(self, matrix, cutoff, max_bond):
        return _split_left_isometry(matrix, cutoff, max_bond)

    def split_right_isometry(self, matrix, cutoff, max_bond):
        return _split_right_isometry(matrix, cutoff, max_bond)


#: The default backend, every kernel's ``linalg`` unless given another.
NUMPY = NumpyLinalg()


def move_center(A, center: int, target: int, linalg: Linalg = NUMPY) -> int:
    """Shift the orthogonality center to ``target`` with QR steps; returns it."""
    xp = linalg.xp
    while center < target:
        left, phys, right = A[center].shape
        q, r = linalg.qr(A[center].reshape(left * phys, right))
        A[center] = q.reshape(left, phys, -1)
        A[center + 1] = xp.tensordot(r, A[center + 1], axes=(1, 0))
        center += 1
    while center > target:
        left, phys, right = A[center].shape
        # M = R^T Q^T from the QR of M^T, so Q^T has orthonormal rows.
        q, r = linalg.qr(A[center].reshape(left, phys * right).T)
        A[center] = q.T.reshape(-1, phys, right)
        A[center - 1] = xp.tensordot(A[center - 1], r.T, axes=(2, 0))
        center -= 1
    return center


def pauli_rotation(
    A,
    center: int,
    generator: str,
    angle: float,
    max_bond: int | None,
    cutoff: float,
    linalg: Linalg = NUMPY,
) -> int:
    """
    Apply ``exp(-i angle/2 generator)`` to the MPS in place.

    The rotation is ``cos(angle/2) I - i sin(angle/2) P``, a bond-2 MPO that is
    the identity outside the generator's support ``[a, b]``. On ``[a, b]`` it is
    applied directly as a block-diagonal stack of the two branches, after which
    the window alone is re-canonicalized with QR and truncated. A one-site
    generator is a 2x2 unitary and needs no truncation at all.

    Returns:
        The new orthogonality center.
    """
    return _windowed_rotation(
        A,
        center,
        bounds(generator),
        lambda j, tensor: apply_pauli(tensor, generator[j], linalg.xp),
        angle,
        max_bond,
        cutoff,
        linalg,
    )


#: ``P (x) I`` and ``I (x) P^T`` on the vectorized index ``2 * out + in`` of an
#: operator site: left and right multiplication by ``P``.
_LEFT_PAULI = {label: np.kron(m, np.eye(2)) for label, m in PAULI_MATRICES.items()}
_RIGHT_PAULI = {label: np.kron(np.eye(2), m.T) for label, m in PAULI_MATRICES.items()}


def operator_conjugation(
    A,
    center: int,
    generator: str,
    angle: float,
    max_bond: int | None,
    cutoff: float,
    linalg: Linalg = NUMPY,
) -> int:
    """
    Replace the MPO ``O`` by ``R^dag O R``, ``R = exp(-i angle/2 generator)``,
    in place.

    ``A`` holds the MPO as an MPS whose site tensors are ``(left, 4, right)``,
    the physical index being ``2 * out + in``, in mixed canonical form for the
    Frobenius inner product. Right multiplication by ``R`` is then the
    rotation generated by ``I (x) P^T`` on that index, and left multiplication
    by ``R^dag`` the one generated by ``P (x) I`` at angle ``-angle``: two
    windowed bond-2 rotations, each truncated like :func:`pauli_rotation`.

    Returns:
        The new orthogonality center.
    """
    window = bounds(generator)
    if window is None:  # a global phase, which conjugation cancels
        return center
    xp = linalg.xp
    # In the operator's own precision, so complex64 sites stay complex64.
    dtype = np.result_type(A[window[0]].dtype, np.complex64)
    for table, signed_angle in ((_RIGHT_PAULI, angle), (_LEFT_PAULI, -angle)):
        matrices = {
            label: linalg.asarray(table[label].astype(dtype))
            for label in set(generator) - {"I"}
        }
        center = _windowed_rotation(
            A,
            center,
            window,
            lambda j, tensor, matrices=matrices: (
                tensor
                if generator[j] == "I"
                else xp.einsum("pq,lqr->lpr", matrices[generator[j]], tensor)
            ),
            signed_angle,
            max_bond,
            cutoff,
            linalg,
        )
    return center


def _windowed_rotation(
    A, center, window, flip, angle, max_bond, cutoff, linalg: Linalg = NUMPY
) -> int:
    """
    Apply ``cos(angle/2) - i sin(angle/2) F`` in place, ``F`` a product of
    local involutions on the sites of ``window`` (``(a, b)``, or ``None`` for
    no site), ``flip(j, tensor)`` applying the one on site ``j``.
    """
    xp = linalg.xp
    # Python floats: a numpy float64 scalar would promote complex64 sites.
    cos, sin = float(np.cos(angle / 2)), float(np.sin(angle / 2))

    if window is None:  # a global phase
        A[center] = A[center] * (cos - 1j * sin)
        return center

    a, b = window
    if a == b:
        # A physical-index unitary keeps every isometry an isometry.
        A[a] = cos * A[a] - 1j * sin * flip(a, A[a])
        return center

    # Enter the window from whichever end the center is closer to.
    from_left = abs(center - a) <= abs(center - b)
    center = move_center(A, center, a if from_left else b, linalg)

    for j in range(a, b + 1):
        tensor = A[j]
        flipped = flip(j, tensor)
        if j == a:
            A[j] = xp.concatenate([tensor, flipped], axis=2)
        elif j == b:
            A[j] = xp.concatenate([cos * tensor, (-1j * sin) * flipped], axis=0)
        else:
            left, phys, right = tensor.shape
            dtype = np.result_type(tensor.dtype, flipped.dtype)
            block = xp.zeros((2 * left, phys, 2 * right), dtype=dtype)
            block[:left, :, :right] = tensor
            block[left:, :, right:] = flipped
            A[j] = block

    if from_left:
        # QR the window into left-isometries, then truncate sweeping back.
        move_center(A, a, b, linalg)
        for j in range(b, a, -1):
            left, phys, right = A[j].shape
            rest, iso = linalg.split_right_isometry(
                A[j].reshape(left, phys * right), cutoff, max_bond
            )
            A[j] = iso.reshape(-1, phys, right)
            A[j - 1] = xp.tensordot(A[j - 1], rest, axes=(2, 0))
        return a

    move_center(A, b, a, linalg)
    for j in range(a, b):
        left, phys, right = A[j].shape
        iso, rest = linalg.split_left_isometry(
            A[j].reshape(left * phys, right), cutoff, max_bond
        )
        A[j] = iso.reshape(left, phys, -1)
        A[j + 1] = xp.tensordot(rest, A[j + 1], axes=(1, 0))
    return b


def pauli_expectation(A, center: int, pauli: str, linalg: Linalg = NUMPY) -> complex:
    """
    Unnormalized ``<psi|P|psi>``, contracting only from the support's ends to
    the orthogonality center: the isometries outside contract to identities.
    """
    xp = linalg.xp
    window = bounds(pauli)
    if window is None:
        return complex(xp.vdot(A[center], A[center]))

    lo, hi = min(window[0], center), max(window[1], center)
    env = None
    for j in range(lo, hi + 1):
        ket = apply_pauli(A[j], pauli[j], xp)
        if env is not None:
            ket = xp.tensordot(env, ket, axes=(1, 0))
        env = xp.tensordot(A[j].conj(), ket, axes=([0, 1], [0, 1]))
    return complex(xp.trace(env))


def squared_norm(A, center: int, linalg: Linalg = NUMPY) -> float:
    """``<psi|psi>``, which in mixed canonical form lives on the center alone."""
    return float(linalg.xp.vdot(A[center], A[center]).real)


def pauli_transition(bra, ket, pauli: str, linalg: Linalg = NUMPY) -> complex:
    """
    Unnormalized ``<bra|P|ket>`` between two MPS, by a full left-to-right
    sweep: the two states have different gauges, so nothing outside the
    support contracts away.
    """
    xp = linalg.xp
    if len(bra) != len(ket):
        raise ValueError(f"The bra has {len(bra)} sites, the ket {len(ket)}.")
    env = xp.ones((1, 1), dtype=ket[0].dtype)
    for j in range(len(ket)):
        ket_j = apply_pauli(ket[j], pauli[j], xp)
        env = xp.tensordot(env, ket_j, axes=(1, 0))  # (bra_l, phys, ket_r)
        env = xp.tensordot(bra[j].conj(), env, axes=([0, 1], [0, 1]))
    return complex(env[0, 0])


def operator_expectation(A, W, linalg: Linalg = NUMPY) -> complex:
    """
    Unnormalized ``<psi|O|psi>`` for an MPO given as ``(left, 4, right)`` site
    tensors, the middle index ``2 * out + in``, by a full left-to-right sweep.
    """
    xp = linalg.xp
    if len(A) != len(W):
        raise ValueError(
            f"The operator acts on {len(W)} sites, the state has {len(A)}."
        )
    env = xp.ones((1, 1, 1), dtype=A[0].dtype)  # (bra, operator, ket) bonds
    for a_j, w_j in zip(A, W):
        w_j = w_j.reshape(w_j.shape[0], 2, 2, w_j.shape[-1])  # (o, out, in, q)
        t = xp.tensordot(env, a_j, axes=(2, 0))  # (a, o, in, d)
        t = xp.tensordot(t, w_j, axes=([1, 2], [0, 2]))  # (a, d, out, q)
        t = xp.tensordot(a_j.conj(), t, axes=([0, 1], [0, 2]))  # (b, d, q)
        env = t.transpose(0, 2, 1)
    return complex(env[0, 0, 0])
