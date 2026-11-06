"""Observables, normalised into the one form mpstab works with.

Everything downstream -- MPO construction, Clifford backpropagation,
measurement planning -- consumes a constant offset plus a
``{pauli_string: coefficient}`` map. :func:`pauli_terms` is the single place
that produces it, from any of the observable formats the public API accepts.
"""

from __future__ import annotations

from typing import Union

from qibo import symbols
from qibo.backends import Backend
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.pauli import validate_pauli_string

Observable = Union[str, list, dict, SymbolicHamiltonian]


def pauli_terms(observable: Observable, nqubits: int) -> tuple[float, dict]:
    """
    Normalise an observable into ``(constant, {pauli_string: coefficient})``.

    Accepted formats:

    - a Pauli string, e.g. ``"XZIZ"``, taken with coefficient 1;
    - a list of Pauli strings, each taken with coefficient 1;
    - a ``{pauli_string: coefficient}`` mapping, returned as given;
    - a qibo ``SymbolicHamiltonian``, whose terms are padded with identities to
      the full width and summed when two terms share a string.

    The constant offset is returned separately rather than as an all-identity
    term, so callers add it once instead of measuring it.
    """
    if isinstance(observable, str):
        validate_pauli_string(observable, nqubits)
        return 0.0, {observable: 1.0}

    if isinstance(observable, dict):
        for pauli in observable:
            validate_pauli_string(pauli, nqubits)
        return 0.0, dict(observable)

    if isinstance(observable, (list, tuple)):
        for pauli in observable:
            validate_pauli_string(pauli, nqubits)
        return 0.0, {pauli: 1.0 for pauli in observable}

    if isinstance(observable, SymbolicHamiltonian):
        terms: dict = {}
        coefficients, names, targets = observable.simple_terms
        for coefficient, term_names, term_targets in zip(coefficients, names, targets):
            labels = ["I"] * nqubits
            for name, qubit in zip(term_names, term_targets):
                labels[qubit] = name
            pauli = "".join(labels)
            terms[pauli] = terms.get(pauli, 0.0) + coefficient.real
        return observable.constant.real, terms

    raise ValueError(
        f"Unsupported observable type {type(observable).__name__}. Pass a Pauli "
        "string, a list of Pauli strings, a {pauli: coefficient} dict or a qibo "
        "SymbolicHamiltonian."
    )


def pauli_string_to_hamiltonian(
    pauli: str, backend: Backend | None = None
) -> SymbolicHamiltonian:
    """The qibo ``SymbolicHamiltonian`` for a single Pauli string, e.g. ``"XZIY"``."""
    validate_pauli_string(pauli)
    form = 1
    for qubit, label in enumerate(pauli):
        form *= getattr(symbols, label)(qubit)
    return SymbolicHamiltonian(form=form, backend=backend)
