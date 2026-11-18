"""Plain-NumPy MPS kernels for the two operations HSMPO spends its time in.

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
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from mpstab.pauli import PAULI_MATRICES as _PAULI_2x2


def bounds(pauli: str) -> tuple[int, int] | None:
    """First and last non-identity sites, or ``None`` for the identity."""
    stripped = pauli.lstrip("I")
    if not stripped:
        return None
    return len(pauli) - len(stripped), len(pauli.rstrip("I")) - 1


def apply_pauli(tensor: np.ndarray, label: str) -> np.ndarray:
    """``label`` applied to the physical index of a ``(l, p, r)`` tensor."""
    if label == "I":
        return tensor
    if label == "X":
        return tensor[:, ::-1, :]
    out = np.empty(tensor.shape, dtype=np.result_type(tensor.dtype, np.complex64))
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
_POTRF, _TRTRS, _HEEVR, _HEEVR_LWORK = scipy.linalg.lapack.get_lapack_funcs(
    ("potrf", "trtrs", "heevr", "heevr_lwork"), dtype=np.complex128
)

#: Optimal ``heevr`` workspace sizes per matrix order: its default is the
#: minimum, which runs unblocked and is markedly slower on large matrices.
_HEEVR_WORKSPACE: dict = {}


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


def _eigh(hermitian: np.ndarray):
    """
    Eigendecomposition of a Hermitian matrix, ascending, by LAPACK's MRRR
    driver (``heevr``), the fastest of the ones scipy exposes at these sizes.
    """
    order = hermitian.shape[0]
    workspace = _HEEVR_WORKSPACE.get(order)
    if workspace is None:
        lwork, lrwork, liwork, _ = _HEEVR_LWORK(order)
        workspace = _HEEVR_WORKSPACE[order] = dict(
            lwork=int(lwork.real), lrwork=int(lrwork), liwork=int(liwork)
        )
    values, vectors, _, _, info = _HEEVR(
        np.asarray(hermitian, dtype=np.complex128), compute_v=1, lower=0, **workspace
    )
    if info != 0:
        raise np.linalg.LinAlgError(f"heevr failed with info={info}")
    return values, vectors


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


def move_center(A: list[np.ndarray], center: int, target: int) -> int:
    """Shift the orthogonality center to ``target`` with QR steps; returns it."""
    while center < target:
        left, phys, right = A[center].shape
        q, r = _qr(A[center].reshape(left * phys, right))
        A[center] = q.reshape(left, phys, -1)
        A[center + 1] = np.tensordot(r, A[center + 1], axes=(1, 0))
        center += 1
    while center > target:
        left, phys, right = A[center].shape
        # M = R^T Q^T from the QR of M^T, so Q^T has orthonormal rows.
        q, r = _qr(A[center].reshape(left, phys * right).T)
        A[center] = q.T.reshape(-1, phys, right)
        A[center - 1] = np.tensordot(A[center - 1], r.T, axes=(2, 0))
        center -= 1
    return center


def pauli_rotation(
    A: list[np.ndarray],
    center: int,
    generator: str,
    angle: float,
    max_bond: int | None,
    cutoff: float,
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
        lambda j, tensor: apply_pauli(tensor, generator[j]),
        angle,
        max_bond,
        cutoff,
    )


#: ``P (x) I`` and ``I (x) P^T`` on the vectorized index ``2 * out + in`` of an
#: operator site: left and right multiplication by ``P``.
_LEFT_PAULI = {label: np.kron(m, np.eye(2)) for label, m in _PAULI_2x2.items()}
_RIGHT_PAULI = {label: np.kron(np.eye(2), m.T) for label, m in _PAULI_2x2.items()}


def operator_conjugation(
    A: list[np.ndarray],
    center: int,
    generator: str,
    angle: float,
    max_bond: int | None,
    cutoff: float,
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
    for table, signed_angle in ((_RIGHT_PAULI, angle), (_LEFT_PAULI, -angle)):
        center = _windowed_rotation(
            A,
            center,
            window,
            lambda j, tensor, table=table: (
                tensor
                if generator[j] == "I"
                else np.einsum("pq,lqr->lpr", table[generator[j]], tensor)
            ),
            signed_angle,
            max_bond,
            cutoff,
        )
    return center


def _windowed_rotation(A, center, window, flip, angle, max_bond, cutoff) -> int:
    """
    Apply ``cos(angle/2) - i sin(angle/2) F`` in place, ``F`` a product of
    local involutions on the sites of ``window`` (``(a, b)``, or ``None`` for
    no site), ``flip(j, tensor)`` applying the one on site ``j``.
    """
    cos, sin = np.cos(angle / 2), np.sin(angle / 2)

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
    center = move_center(A, center, a if from_left else b)

    for j in range(a, b + 1):
        tensor = A[j]
        flipped = flip(j, tensor)
        if j == a:
            A[j] = np.concatenate([tensor, flipped], axis=2)
        elif j == b:
            A[j] = np.concatenate([cos * tensor, (-1j * sin) * flipped], axis=0)
        else:
            left, phys, right = tensor.shape
            block = np.zeros(
                (2 * left, phys, 2 * right), dtype=np.result_type(tensor, flipped)
            )
            block[:left, :, :right] = tensor
            block[left:, :, right:] = flipped
            A[j] = block

    if from_left:
        # QR the window into left-isometries, then truncate sweeping back.
        move_center(A, a, b)
        for j in range(b, a, -1):
            left, phys, right = A[j].shape
            rest, iso = _split_right_isometry(
                A[j].reshape(left, phys * right), cutoff, max_bond
            )
            A[j] = iso.reshape(-1, phys, right)
            A[j - 1] = np.tensordot(A[j - 1], rest, axes=(2, 0))
        return a

    move_center(A, b, a)
    for j in range(a, b):
        left, phys, right = A[j].shape
        iso, rest = _split_left_isometry(
            A[j].reshape(left * phys, right), cutoff, max_bond
        )
        A[j] = iso.reshape(left, phys, -1)
        A[j + 1] = np.tensordot(rest, A[j + 1], axes=(1, 0))
    return b


def pauli_expectation(A: list[np.ndarray], center: int, pauli: str) -> complex:
    """
    Unnormalized ``<psi|P|psi>``, contracting only from the support's ends to
    the orthogonality center: the isometries outside contract to identities.
    """
    window = bounds(pauli)
    if window is None:
        return np.vdot(A[center], A[center])

    lo, hi = min(window[0], center), max(window[1], center)
    env = None
    for j in range(lo, hi + 1):
        ket = apply_pauli(A[j], pauli[j])
        if env is not None:
            ket = np.tensordot(env, ket, axes=(1, 0))
        env = np.tensordot(A[j].conj(), ket, axes=([0, 1], [0, 1]))
    return np.trace(env)


def squared_norm(A: list[np.ndarray], center: int) -> float:
    """``<psi|psi>``, which in mixed canonical form lives on the center alone."""
    return float(np.vdot(A[center], A[center]).real)


def pauli_transition(
    bra: list[np.ndarray], ket: list[np.ndarray], pauli: str
) -> complex:
    """
    Unnormalized ``<bra|P|ket>`` between two MPS, by a full left-to-right
    sweep: the two states have different gauges, so nothing outside the
    support contracts away.
    """
    env = np.ones((1, 1))
    for j in range(len(ket)):
        ket_j = apply_pauli(ket[j], pauli[j])
        env = np.tensordot(env, ket_j, axes=(1, 0))  # (bra_l, phys, ket_r)
        env = np.tensordot(bra[j].conj(), env, axes=([0, 1], [0, 1]))
    return complex(env[0, 0])
