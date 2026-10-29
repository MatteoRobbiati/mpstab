"""Pauli-string conventions shared by every part of mpstab.

A Pauli string is a plain ``str`` over ``"IXYZ"``, qubit-0-leftmost, one
character per qubit. It never carries a sign: signs travel next to the string
as a separate ``+1``/``-1`` factor, so callers always handle them explicitly.
"""

from __future__ import annotations

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
