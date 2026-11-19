"""Pauli and coherent noise on the HSMPO surrogate.

Build a :class:`NoiseModel` with :meth:`NoiseModel.add`, as in qibo, then call
:meth:`~mpstab.HSMPO.noisy_expectation`::

    noise = NoiseModel()
    noise.add(DepolarizingError(1e-3))
    noise.add(PauliError({"ZZ": 5e-4}), gates.CNOT)
    result = hsmpo.noisy_expectation(observable, noise)

The machinery underneath (noise events placed by gate index, their compilation
into the Clifford frame, and the fault-pattern estimator) lives in
:mod:`mpstab.noise.events`, :mod:`mpstab.noise.compile` and
:mod:`mpstab.noise.mc_noise_estimator`.
"""

from mpstab.noise.mc_noise_estimator import NoisyExpectation
from mpstab.noise.model import (
    CoherentError,
    DepolarizingError,
    NoiseModel,
    PauliError,
    ReadoutError,
)

__all__ = [
    "CoherentError",
    "DepolarizingError",
    "NoiseModel",
    "NoisyExpectation",
    "PauliError",
    "ReadoutError",
]
