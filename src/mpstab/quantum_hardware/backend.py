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
from collections import defaultdict


class QiboSimulator:
    """Runs circuits with qibo's own simulator. The default backend."""

    def execute_circuits(self, circuits, nshots: int) -> list:
        nqubits = circuits[0].nqubits
        if nqubits > 30:
            warnings.warn(
                f"{nqubits} qubits: exact simulation is impractical above ~30."
            )
        return [circuit(nshots=nshots).frequencies() for circuit in circuits]


def execute_plan(backend, plan) -> list:
    """
    Run every circuit in a :class:`~mpstab.quantum_hardware.plan.MeasurementPlan`
    with its own shot count, batching circuits that share a shot count into a
    single ``backend.execute_circuits`` call instead of one call per circuit.

    A ``"shadows"``/``"tnice"`` plan's circuits mostly share
    ``shots_per_setting`` shots each, so this collapses what would otherwise be
    one backend call per shot into a handful of calls -- the contract already
    supports a batch (``execute_circuits`` takes a list), the naive call site
    just wasn't using it.

    Args:
        backend: anything with ``execute_circuits(circuits, nshots)``.
        plan: a plan whose ``circuits`` and ``shots`` are the same length.

    Returns:
        One frequency dict per circuit, in ``plan.circuits`` order -- the
        ordering :mod:`~mpstab.quantum_hardware.estimate` assumes.
    """
    groups: dict = defaultdict(list)
    for index, shots in enumerate(plan.shots):
        groups[shots].append(index)

    frequencies: list = [None] * len(plan.circuits)
    for shots, indices in groups.items():
        circuits = [plan.circuits[i] for i in indices]
        for index, result in zip(indices, backend.execute_circuits(circuits, shots)):
            frequencies[index] = result
    return frequencies
