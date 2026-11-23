"""Pauli, coherent and general (Kraus) noise on the HSMPO surrogate.

Build a :class:`NoiseModel` with :meth:`NoiseModel.add`, as in qibo, then call
:meth:`~mpstab.HSMPO.noisy_expectation`::

    noise = NoiseModel()
    noise.add(DepolarizingError(1e-3))
    noise.add(PauliError({"ZZ": 5e-4}), gates.CNOT)
    result = hsmpo.noisy_expectation(observable, noise)

The machinery underneath (noise events placed by gate index, their compilation
into the Clifford frame, and the fault-pattern estimator) lives in
:mod:`mpstab.models.noise.events`, :mod:`mpstab.models.noise.compile` and
:mod:`mpstab.models.noise.mc_noise_estimator`.
"""

from mpstab.models.noise.mc_noise_estimator import NoisyExpectation
from mpstab.models.noise.model import (
    AmplitudeDampingError,
    CoherentError,
    DepolarizingError,
    KrausError,
    NoiseModel,
    PauliError,
    ReadoutError,
    StatePreparationError,
    ThermalRelaxationError,
)

__all__ = [
    "AmplitudeDampingError",
    "CoherentError",
    "DepolarizingError",
    "KrausError",
    "NoiseModel",
    "NoisyExpectation",
    "PauliError",
    "ReadoutError",
    "StatePreparationError",
    "ThermalRelaxationError",
]
