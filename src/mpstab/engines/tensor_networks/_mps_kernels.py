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


def _svd(matrix: np.ndarray):
    """Thin SVD, falling back to the slower but sturdier ``gesvd`` driver."""
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(
            matrix, full_matrices=False, lapack_driver="gesvd", check_finite=False
        )


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


def move_center(A: list[np.ndarray], center: int, target: int) -> int:
    """Shift the orthogonality center to ``target`` with QR steps; returns it."""
    while center < target:
        left, phys, right = A[center].shape
        q, r = np.linalg.qr(A[center].reshape(left * phys, right))
        A[center] = q.reshape(left, phys, -1)
        A[center + 1] = np.tensordot(r, A[center + 1], axes=(1, 0))
        center += 1
    while center > target:
        left, phys, right = A[center].shape
        # M = R^T Q^T from the QR of M^T, so Q^T has orthonormal rows.
        q, r = np.linalg.qr(A[center].reshape(left, phys * right).T)
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
    the window alone is re-canonicalized with QR and truncated with SVD. A
    one-site generator is a 2x2 unitary and needs no truncation at all.

    Returns:
        The new orthogonality center.
    """
    window = bounds(generator)
    cos, sin = np.cos(angle / 2), np.sin(angle / 2)

    if window is None:  # a global phase
        A[center] = A[center] * (cos - 1j * sin)
        return center

    a, b = window
    if a == b:
        # A physical-index unitary keeps every isometry an isometry.
        gate = cos * _PAULI_2x2["I"] - 1j * sin * _PAULI_2x2[generator[a]]
        A[a] = np.einsum("pq,lqr->lpr", gate, A[a])
        return center

    # Enter the window from whichever end the center is closer to.
    from_left = abs(center - a) <= abs(center - b)
    center = move_center(A, center, a if from_left else b)

    for j in range(a, b + 1):
        tensor = A[j]
        flipped = apply_pauli(tensor, generator[j])
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
            u, s, vh = _svd(A[j].reshape(left, phys * right))
            k = n_kept(s, cutoff, max_bond)
            A[j] = vh[:k].reshape(k, phys, right)
            A[j - 1] = np.tensordot(A[j - 1], u[:, :k] * s[:k], axes=(2, 0))
        return a

    move_center(A, b, a)
    for j in range(a, b):
        left, phys, right = A[j].shape
        u, s, vh = _svd(A[j].reshape(left * phys, right))
        k = n_kept(s, cutoff, max_bond)
        A[j] = u[:, :k].reshape(left, phys, k)
        A[j + 1] = np.tensordot(s[:k, None] * vh[:k], A[j + 1], axes=(1, 0))
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
