"""Error mitigation of expectation values, with HSMPO as the classical simulator.

Clifford data regression learns a map from noisy expectation values to
noiseless ones on low-magic variants of the target circuit, whose noiseless
values HSMPO computes cheaply::

    from mpstab import cdr

    noise_map = cdr(ansatz, "ZZII", noise, n_training_circuits=50, seed=0)
    mitigated = noise_map(noisy_value_of_the_target)

:func:`cdr` fits one noisy value per circuit; :func:`vncdr` fits the values
at several noise levels. ``noise`` is a :class:`~mpstab.models.noise.NoiseModel`
to simulate the noisy values with, or a callable returning them, such as one
running each circuit on hardware.

The data-driven methods live in
:mod:`~mpstab.models.error_mitigation.data_driven_methods`; noise amplification
by folding, which other methods such as zero-noise extrapolation share, in
:mod:`~mpstab.models.error_mitigation.folding`.
"""

from mpstab.models.error_mitigation.data_driven_methods import (
    MitigatedResult,
    MitigationNoiseMap,
    MitigationPlan,
    VnCDRMitigation,
    cdr,
    mitigated_expectation_at_cut,
    noisy_value,
    training_circuits,
    vncdr,
)
from mpstab.models.error_mitigation.folding import fold_circuit

__all__ = [
    "MitigatedResult",
    "MitigationPlan",
    "VnCDRMitigation",
    "mitigated_expectation_at_cut",
    "MitigationNoiseMap",
    "cdr",
    "fold_circuit",
    "noisy_value",
    "training_circuits",
    "vncdr",
]
