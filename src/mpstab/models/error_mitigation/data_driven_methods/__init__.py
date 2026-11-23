"""Data-driven error mitigation: learn a noise map on training circuits whose
noiseless values HSMPO computes cheaply, then apply it to the target.

- :mod:`~mpstab.models.error_mitigation.data_driven_methods.cdr`: Clifford data
  regression, one noisy value per circuit;
- :mod:`~mpstab.models.error_mitigation.data_driven_methods.vncdr`:
  variable-noise CDR, the values at several noise levels;
- :mod:`~mpstab.models.error_mitigation.data_driven_methods.plan`: vnCDR on the
  head of an HQSMPO measurement plan, with a calibrated measurement layer;
- :mod:`~mpstab.models.error_mitigation.data_driven_methods.training` and
  :mod:`~mpstab.models.error_mitigation.data_driven_methods.regression`: the
  training circuits and the machinery CDR and vnCDR share.
"""

from mpstab.models.error_mitigation.data_driven_methods.cdr import cdr
from mpstab.models.error_mitigation.data_driven_methods.plan import (
    MitigatedResult,
    MitigationPlan,
    VnCDRMitigation,
    mitigated_expectation_at_cut,
)
from mpstab.models.error_mitigation.data_driven_methods.regression import (
    MitigationNoiseMap,
    noisy_value,
)
from mpstab.models.error_mitigation.data_driven_methods.training import (
    training_circuits,
)
from mpstab.models.error_mitigation.data_driven_methods.vncdr import vncdr

__all__ = [
    "MitigatedResult",
    "MitigationPlan",
    "VnCDRMitigation",
    "mitigated_expectation_at_cut",
    "MitigationNoiseMap",
    "cdr",
    "noisy_value",
    "training_circuits",
    "vncdr",
]
