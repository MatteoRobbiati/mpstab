"""Pauli and coherent noise on the HSMPO surrogate.

Describe the noise as :class:`PauliChannel` and :class:`CoherentRotation`
events (or convert a qibo ``NoiseModel`` with :func:`from_qibo_noise_model`),
compile it once against a surrogate with :func:`compile_noise`, then estimate
expectation values with :func:`noisy_expectation`.
"""

from mpstab.noise.events import (
    CoherentRotation,
    NoiseEvent,
    PauliChannel,
    from_qibo_noise_model,
)

__all__ = [
    "CoherentRotation",
    "NoiseEvent",
    "PauliChannel",
    "from_qibo_noise_model",
]
