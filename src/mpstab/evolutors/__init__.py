"""
The surrogates: they decide what the stabilizer and tensor-network engines each
handle, and drive the simulation through a circuit.
"""

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.evolutors.hsmpo import HSMPO

__all__ = ["HSMPO", "HQSMPO"]


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
