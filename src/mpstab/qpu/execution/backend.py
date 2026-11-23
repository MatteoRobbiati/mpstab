"""
Running a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan` on a backend.

A backend is any object with an ``execute_circuits(circuits, nshots)`` method
returning one frequency dictionary per circuit, as qibo's
``Circuit.__call__(nshots=...).frequencies()`` does; there is no base class to
subclass. The default,
:class:`~mpstab.qibo_tools.utilized_backends.statevector_exact_simulator.StatevectorExactSimulator`,
runs them with qibo.

Bitstrings are qubit-0-leftmost throughout mpstab, as in qibo. A backend whose
provider reports them the other way round, qubit 0 rightmost as in qiskit, must
reverse them before returning.
"""

from __future__ import annotations

from collections import defaultdict


def run_plan(plan, backend) -> list:
    """
    Run every circuit in a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan`
    with its own shot count, batching circuits that share a shot count into a
    single ``backend.execute_circuits`` call instead of one call per circuit.

    A ``"shadows"`` plan's circuits mostly share
    ``shots_per_setting`` shots each, so this collapses what would otherwise be
    one backend call per shot into a handful of calls -- the contract already
    supports a batch (``execute_circuits`` takes a list), the naive call site
    just wasn't using it.

    Args:
        plan: a plan whose ``circuits`` and ``shots`` are the same length.
        backend: anything with ``execute_circuits(circuits, nshots)``.

    Returns:
        One frequency dict per circuit, in ``plan.circuits`` order and
        qubit-0-leftmost -- what :mod:`~mpstab.qpu.measurement`
        assumes.

    Raises:
        ValueError: if the backend returns a different number of results than
            the circuits it was given, which would otherwise leave gaps that
            only surface later, inside the estimator.
    """
    groups: dict = defaultdict(list)
    for index, shots in enumerate(plan.shots):
        groups[shots].append(index)

    frequencies: list = [None] * len(plan.circuits)
    for shots, indices in groups.items():
        circuits = [plan.circuits[i] for i in indices]
        results = backend.execute_circuits(circuits, shots)
        if len(results) != len(circuits):
            raise ValueError(
                f"{type(backend).__name__}.execute_circuits returned "
                f"{len(results)} results for {len(circuits)} circuits; it must "
                "return one frequency dict per circuit, in order."
            )
        for index, result in zip(indices, results):
            frequencies[index] = result

    return frequencies
