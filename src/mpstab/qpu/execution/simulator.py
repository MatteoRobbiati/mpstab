"""
Deprecated alias of the former simulator module: ``QiboSimulator`` is now
:class:`~mpstab.qibo_tools.utilized_backends.statevector_exact_simulator.StatevectorExactSimulator`
and ``execute_plan`` is in :mod:`mpstab.qpu.execution.execute`.
"""

import warnings

from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    StatevectorExactSimulator,
)
from mpstab.qpu.execution.execute import execute_plan  # noqa: F401

warnings.warn(
    "mpstab.qpu.execution.simulator was removed: QiboSimulator is now "
    "mpstab.qibo_tools.StatevectorExactSimulator and execute_plan is in "
    "mpstab.qpu.execution.execute; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

QiboSimulator = StatevectorExactSimulator
