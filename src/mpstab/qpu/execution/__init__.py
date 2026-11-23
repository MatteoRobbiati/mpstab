"""
Step 3 of the QPU pipeline: running a plan's circuits, here or elsewhere.

- :mod:`~mpstab.qpu.execution.execute`: the backend contract and
  :func:`~mpstab.qpu.execution.execute.execute_plan`. Backends running the
  circuits with qibo are in :mod:`mpstab.qibo_tools`.
- :mod:`~mpstab.qpu.execution.qasm`: OpenQASM 2 export of a plan's circuits.
- :mod:`~mpstab.qpu.execution.storage`: saving them to a folder, so they can run
  on a device.
"""

import warnings

from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    StatevectorExactSimulator,
)
from mpstab.qpu.execution.execute import execute_plan
from mpstab.qpu.execution.qasm import QELIB1_GATES, circuit_to_qasm, plan_to_qasm
from mpstab.qpu.execution.storage import save_circuits

__all__ = [
    "QELIB1_GATES",
    "circuit_to_qasm",
    "execute_plan",
    "plan_to_qasm",
    "save_circuits",
]


def __getattr__(name):
    # QiboSimulator was renamed StatevectorExactSimulator and moved to qibo_tools.
    if name == "QiboSimulator":
        warnings.warn(
            "mpstab.qpu.execution.QiboSimulator moved to "
            "mpstab.qibo_tools.StatevectorExactSimulator; the old name will be "
            "removed.",
            DeprecationWarning,
            stacklevel=2,
        )
        return StatevectorExactSimulator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
