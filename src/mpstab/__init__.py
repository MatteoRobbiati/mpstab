"""Hybrid stabilizer-MPO simulation of quantum circuits."""

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import pauli_string_to_hamiltonian, pauli_terms
from mpstab.qibo_tools import MetaBackend

__all__ = [
    "HSMPO",
    "HQSMPO",
    "MetaBackend",
    "pauli_string_to_hamiltonian",
    "pauli_terms",
]


def __getattr__(name):
    # HQSMPO was called HSynthSMPO; keep the old name working for a while.
    if name == "HSynthSMPO":
        import warnings

        warnings.warn(
            "HSynthSMPO was renamed HQSMPO; the old name will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return HQSMPO
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
