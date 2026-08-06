"""
The backend contract for running a :class:`~mpstab.quantum_hardware.plan.MeasurementPlan`.

A backend is any object with an ``execute_circuits(circuits, nshots)`` method
returning one frequency dictionary per circuit -- exactly what qibo's
``Circuit.__call__(nshots=...).frequencies()`` gives. That single method is the
whole contract: no base class to subclass, so a qibolab device or a mock is a
drop-in replacement for :class:`QiboSimulator` below.
"""

from __future__ import annotations

import warnings


class QiboSimulator:
    """Runs circuits with qibo's own simulator. The default backend."""

    def execute_circuits(self, circuits, nshots: int) -> list:
        nqubits = circuits[0].nqubits
        if nqubits > 30:
            warnings.warn(
                f"{nqubits} qubits: exact simulation is impractical above ~30."
            )
        return [circuit(nshots=nshots).frequencies() for circuit in circuits]
