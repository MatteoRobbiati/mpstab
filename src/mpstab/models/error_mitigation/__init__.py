"""Data-driven error mitigation, with HSMPO simulating the training circuits.

Clifford data regression learns a map from noisy expectation values to
noiseless ones on low-magic variants of the target circuit, whose noiseless
values HSMPO computes cheaply::

    from mpstab.models.error_mitigation import cdr

    noise_map = cdr(ansatz, "ZZII", noise, n_training_circuits=50, seed=0)
    mitigated = noise_map(noisy_value_of_the_target)

:func:`cdr` fits one noisy value per circuit; :func:`vncdr` fits the values
at several noise levels. ``noise`` is a :class:`~mpstab.models.noise.NoiseModel`
to simulate the noisy values with, or a callable returning them, such as one
running each circuit on hardware.
"""

from mpstab.models.error_mitigation.cdr import (
    MitigationNoiseMap,
    cdr,
    noisy_value,
    vncdr,
)
from mpstab.models.error_mitigation.training import fold_circuit, training_circuits

__all__ = [
    "MitigationNoiseMap",
    "cdr",
    "fold_circuit",
    "noisy_value",
    "training_circuits",
    "vncdr",
]
