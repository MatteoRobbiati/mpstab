"""Pauli-string conventions shared by every part of mpstab.

A Pauli string is a plain ``str`` over ``"IXYZ"``, qubit-0-leftmost, one
character per qubit. It never carries a sign: signs travel next to the string
as a separate ``+1``/``-1`` factor, so callers always handle them explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import numpy as np

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


def _product_tables():
    """
    Single-site Pauli products ``a . b = i**phase[a, b] * label[a, b]``, on
    the codes of :data:`PAULI_LABELS` (``I=0, X=1, Y=2, Z=3``).
    """
    labels = np.zeros((4, 4), dtype=np.int8)
    phases = np.zeros((4, 4), dtype=np.int8)
    for a, left in enumerate(PAULI_ARRAY):
        for b, right in enumerate(PAULI_ARRAY):
            product = left @ right
            for c, candidate in enumerate(PAULI_ARRAY):
                overlap = np.trace(candidate.conj().T @ product) / 2
                if abs(overlap) > 0.5:
                    labels[a, b] = c
                    phases[a, b] = int(round(np.angle(overlap) / (np.pi / 2))) % 4
    return labels, phases


#: Pauli codes and phases (powers of ``i``) of every single-site product.
_PRODUCT_LABELS, _PRODUCT_PHASES = _product_tables()

#: :data:`PAULI_LABELS` as an array, indexed by code.
_LABEL_CHARS = np.array(list(PAULI_LABELS))


#: Byte value -> Pauli code, for :func:`pauli_codes`; ``-1`` marks a byte
#: that is not a Pauli label.
_CODE_OF_BYTE = np.full(256, -1, dtype=np.int8)
for _code, _label in enumerate(PAULI_LABELS):
    _CODE_OF_BYTE[ord(_label)] = _code


def pauli_codes(paulis: Sequence[str] | str) -> np.ndarray:
    """
    Pauli strings as integer codes (``I=0, X=1, Y=2, Z=3``): shape ``(n,)``
    for one string, ``(m, n)`` for a sequence of ``m`` strings (``(0, 0)``
    for none).

    Raises:
        ValueError: on characters outside ``IXYZ``, or strings of different
            lengths.
    """
    if isinstance(paulis, str):
        codes = _CODE_OF_BYTE[np.frombuffer(paulis.encode(), dtype=np.uint8)]
        if (codes < 0).any():
            validate_pauli_string(paulis)  # raises, naming the bad characters
        return codes
    rows = [pauli_codes(p) for p in paulis]
    if not rows:
        return np.zeros((0, 0), dtype=np.int8)
    if len({row.size for row in rows}) > 1:
        raise ValueError(
            f"Pauli strings of different lengths: {sorted({r.size for r in rows})}."
        )
    return np.stack(rows)


def pauli_labels(codes: np.ndarray) -> list[str]:
    """Inverse of :func:`pauli_codes` for a ``(m, n)`` array: ``m`` strings."""
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"Expected a (strings, sites) array, got {codes.shape}.")
    return ["".join(row) for row in _LABEL_CHARS[codes]]


def multiply_codes(left: np.ndarray, right: np.ndarray):
    """
    Site-wise product of Pauli strings given as codes, broadcasting over
    leading axes: ``left . right = i**phase * result``.

    Returns:
        ``(result, phase)``: the product's codes, and its phase as a power of
        ``i`` modulo 4, one per string.
    """
    if np.shape(left)[-1:] != np.shape(right)[-1:]:
        raise ValueError(
            f"Pauli strings of different lengths: {np.shape(left)[-1]} and "
            f"{np.shape(right)[-1]}."
        )
    left, right = np.broadcast_arrays(left, right)
    result = _PRODUCT_LABELS[left, right]
    phase = _PRODUCT_PHASES[left, right].sum(axis=-1, dtype=np.int64) % 4
    return result, phase


def pauli_product(*paulis: str) -> tuple[str, complex]:
    """
    ``P_1 . P_2 . ... . P_k`` of equal-length Pauli strings, left to right.

    Returns:
        ``(pauli, phase)``, the product being ``phase * pauli`` with ``phase``
        one of ``1, 1j, -1, -1j``.

    Example::

        from mpstab.pauli import pauli_product

        print(pauli_product("XI", "YZ"))
        # ('ZZ', 1j)
    """
    if not paulis:
        raise ValueError("pauli_product needs at least one Pauli string.")
    codes, phase = pauli_codes(paulis[0]), 0
    for pauli in paulis[1:]:
        codes, step = multiply_codes(codes, pauli_codes(pauli))
        phase += int(step)
    return pauli_labels(codes[None])[0], 1j ** (phase % 4)


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


#: Rows of rotations :func:`pauli_light_cone` handles at once, bounding its memory.
_LIGHT_CONE_BLOCK = 1024


def pauli_light_cone(
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

        from mpstab.pauli import pauli_light_cone

        # IX commutes with everything after it and drops out; ZI commutes
        # with the observable but not with the kept XI, so it stays.
        print(pauli_light_cone(["ZI", "IX", "XI"], ["ZI"], 2))
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
