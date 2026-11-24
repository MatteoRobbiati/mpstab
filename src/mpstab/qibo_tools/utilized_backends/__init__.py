"""Qibo simulators mpstab utilizes as backends for measurement plans.

Each has ``execute_circuits(circuits, nshots)``, as
:func:`~mpstab.qpu.execution.backend.run_plan` expects, and stands in for a
device:

- :mod:`~mpstab.qibo_tools.utilized_backends.statevector_exact_simulator`:
  :class:`StatevectorExactSimulator`, noiseless statevector simulation, the
  default backend of :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.expectation_at_cut`;
- :mod:`~mpstab.qibo_tools.utilized_backends.density_matrix_noise_simulator`:
  :class:`DensityMatrixNoiseSimulator`, density-matrix simulation under a
  :class:`~mpstab.models.noise.NoiseModel`, to validate error mitigation on
  small circuits.
"""

from mpstab.qibo_tools.utilized_backends.density_matrix_noise_simulator import (
    DensityMatrixNoiseSimulator,
)
from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    StatevectorExactSimulator,
)

__all__ = ["DensityMatrixNoiseSimulator", "StatevectorExactSimulator"]
