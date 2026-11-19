"""Pauli-string conventions shared by every part of mpstab.

A Pauli string is a plain ``str`` over ``"IXYZ"``, qubit-0-leftmost, one
character per qubit. It never carries a sign: signs travel next to the string
as a separate ``+1``/``-1`` factor, so callers always handle them explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np
import stim

#: Pauli labels, in the order used as the physical index of Pauli-basis tensors.
PAULI_LABELS = "IXYZ"

#: Single-qubit Pauli matrices, keyed by label.
PAULI_MATRICES = {
    "I": np.array([[1, 0], [0, 1]], dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

#: The same matrices stacked in :data:`PAULI_LABELS` order, shape ``(4, 2, 2)``.
PAULI_ARRAY = np.stack([PAULI_MATRICES[label] for label in PAULI_LABELS])


def pauli_matrix(pauli: str) -> np.ndarray:
    """
    Dense matrix of a Pauli string, qubit 0 being the most significant.

    Args:
        pauli: the Pauli string; the empty string gives the ``1 x 1`` identity.

    Returns:
        The ``2**n x 2**n`` complex matrix ``P_0 (x) P_1 (x) ... (x) P_{n-1}``.
    """
    return reduce(
        np.kron, (PAULI_MATRICES[label] for label in pauli), np.eye(1, dtype=complex)
    )


def validate_pauli_string(pauli: str, nqubits: int | None = None) -> None:
    """
    Check that ``pauli`` is a well-formed Pauli string of ``nqubits`` characters.

    Args:
        pauli: the string to check.
        nqubits: expected length; ``None`` skips the length check.

    Raises:
        ValueError: on characters outside ``IXYZ`` or on a length mismatch.
    """
    invalid = set(pauli) - set(PAULI_LABELS)
    if invalid:
        raise ValueError(
            f"Invalid characters {sorted(invalid)} in Pauli string {pauli!r}. "
            "Use only I, X, Y and Z, with no signs or coefficients: "
            "'XYZIX', not '-2*XYZIX'."
        )
    if nqubits is not None and len(pauli) != nqubits:
        raise ValueError(
            f"Pauli string {pauli!r} has length {len(pauli)}, expected {nqubits} "
            "(one character per qubit)."
        )


def weight(pauli: str) -> int:
    """Number of non-identity sites in ``pauli``."""
    return sum(label != "I" for label in pauli)


def from_stim(pauli: stim.PauliString) -> tuple[str, int]:
    """Split a ``stim.PauliString`` into an unsigned label and its ``+/-1`` sign."""
    text = str(pauli)
    sign = -1 if text.startswith("-") else 1
    return text.lstrip("+-").replace("_", "I"), sign


def conjugate(
    pauli: str, tableau: stim.Tableau, sign: float = 1.0
) -> tuple[str, float]:
    """
    Conjugate a Pauli string by a Clifford tableau: ``U P U^dag``.

    Args:
        pauli: the Pauli string to conjugate.
        tableau: the Clifford ``U``. Pass ``tableau.inverse()`` to get
            ``U^dag P U`` instead, which is what folding an operator applied
            *after* the state into the observable needs.
        sign: prefactor carried on the input Pauli.

    Returns:
        ``(pauli, sign)`` for the conjugated operator.
    """
    padded = pauli.ljust(len(tableau), "I")
    image = tableau(stim.PauliString(padded))
    label, image_sign = from_stim(image)
    return label, sign * image_sign


def anticommutation(left: Sequence[str], right: Sequence[str], nqubits: int):
    """
    Which pairs of Pauli strings anticommute, via the symplectic product.

    Args:
        left, right: full-width Pauli strings.
        nqubits: their width, needed when either sequence is empty.

    Returns:
        Boolean ``(len(left), len(right))`` matrix, ``True`` where the strings
        anticommute.

    Example::

        from mpstab.pauli import anticommutation

        print(anticommutation(["XI", "ZZ"], ["ZI", "XX"], 2))
        # [[ True False]
        #  [False False]]
    """
    return _anticommuting(*_symplectic(left, nqubits), *_symplectic(right, nqubits))


#: Rows of rotations :func:`light_cone` handles at once, bounding its memory.
_LIGHT_CONE_BLOCK = 1024


def light_cone(
    generators: Sequence[str], observables: Sequence[str], nqubits: int
) -> np.ndarray:
    """
    The rotations in a chain that can affect the expectation of any of
    ``observables``.

    For ``<psi| R_1^dag ... R_n^dag  O  R_n ... R_1 |psi>`` with Pauli
    rotations ``R_j = exp(-i theta_j/2 G_j)``, a rotation whose generator
    commutes with ``O`` and with every later rotation still present commutes
    through to ``R_j^dag R_j = 1``. Walking the chain backwards and dropping
    such rotations is exact, whatever the angles, and the result holds
    term by term for an observable that is a sum of the given strings.

    Args:
        generators: generator of each rotation, in the order they are applied.
        observables: Pauli strings of the observable's terms.
        nqubits: width of every string.

    Returns:
        Increasing indices of the rotations to keep.

    Example::

        from mpstab.pauli import light_cone

        # IX commutes with everything after it and drops out; ZI commutes
        # with the observable but not with the kept XI, so it stays.
        print(light_cone(["ZI", "IX", "XI"], ["ZI"], 2))
        # [0 2]
    """
    x, z = _symplectic(generators, nqubits)
    observable_x, observable_z = _symplectic(observables, nqubits)
    kept = np.zeros(len(x), dtype=bool)
    for stop in range(len(x), 0, -_LIGHT_CONE_BLOCK):
        start = max(0, stop - _LIGHT_CONE_BLOCK)
        block_x, block_z = x[start:stop], z[start:stop]
        # Whether each rotation in the block reaches past it on its own...
        reaches = _anticommuting(block_x, block_z, observable_x, observable_z).any(1)
        later = np.flatnonzero(kept[stop:]) + stop
        if later.size:
            reaches |= _anticommuting(block_x, block_z, x[later], z[later]).any(1)
        # ...or through a kept rotation later in the same block.
        inner = _anticommuting(block_x, block_z, block_x, block_z)
        for i in reversed(range(stop - start)):
            kept[start + i] = (
                reaches[i] or (inner[i, i + 1 :] & kept[start + i + 1 : stop]).any()
            )
    return np.flatnonzero(kept)


def _symplectic(paulis: Sequence[str], nqubits: int):
    """
    The ``(x, z)`` bit matrices of some Pauli strings, as floats so their
    products run through BLAS; counts up to ``2**53`` are exact.
    """
    labels = np.array([list(p) for p in paulis], dtype="<U1").reshape(-1, nqubits)
    x = np.isin(labels, ("X", "Y")).astype(np.float64)
    z = np.isin(labels, ("Z", "Y")).astype(np.float64)
    return x, z


def _anticommuting(left_x, left_z, right_x, right_z) -> np.ndarray:
    """Anticommutation matrix from symplectic bit matrices."""
    # Some BLAS builds (Apple Accelerate) raise spurious floating-point flags
    # on these 0/1 products; the integer-valued results are exact regardless.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        products = left_x @ right_z.T + left_z @ right_x.T
    return (products % 2).astype(bool)
