"""
Running a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan` on a backend.

A backend is any object with an ``execute_circuits(circuits, nshots)`` method
returning one frequency dictionary per circuit, as qibo's
``Circuit.__call__(nshots=...).frequencies()`` does; there is no base class to
subclass. :class:`QiboSimulator` below is the default.

Bitstrings are qubit-0-leftmost throughout mpstab, as in qibo. A backend whose
provider reports them the other way round, qubit 0 rightmost as in qiskit, must
reverse them before returning.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict

import numpy as np


class QiboSimulator:
    """
    Runs circuits with qibo's own simulator. The default backend.

    Identical circuits are simulated once: a ``"shadows"`` plan repeats the
    same state preparation under only ``3^n`` distinct measurement bases, so
    its thousands of one-shot circuits collapse to a few simulations. Each
    circuit still gets its own ``nshots`` independent samples, so the results
    are distributed exactly as if every circuit had been run separately.
    """

    def execute_circuits(self, circuits, nshots: int) -> list:
        nqubits = circuits[0].nqubits
        if nqubits > 30:
            warnings.warn(
                f"{nqubits} qubits: exact simulation is impractical above ~30."
            )
        groups: dict = defaultdict(list)
        keys: dict = {}  # plans often repeat the same circuit object
        for index, circuit in enumerate(circuits):
            key = keys.get(id(circuit))
            if key is None:
                key = keys[id(circuit)] = _circuit_key(circuit)
            groups[key].append(index)

        frequencies: list = [None] * len(circuits)
        for indices in groups.values():
            if len(indices) == 1:
                circuit = circuits[indices[0]]
                frequencies[indices[0]] = circuit(nshots=nshots).frequencies()
                continue
            result = circuits[indices[0]](nshots=nshots * len(indices))
            # Shuffle so the split never depends on how qibo orders its samples.
            samples = np.random.permutation(np.asarray(result.samples(binary=True)))
            bitstrings = ["".join(map(str, row)) for row in samples.astype(int)]
            for k, index in enumerate(indices):
                frequencies[index] = Counter(bitstrings[k * nshots : (k + 1) * nshots])
        return frequencies


def _circuit_key(circuit) -> tuple:
    """Everything that determines a circuit's output distribution."""
    return (
        (circuit.nqubits, circuit.density_matrix)
        + tuple(
            (
                type(gate).__name__,
                gate.qubits,
                tuple(np.ravel(gate.parameters)) if gate.parameters else (),
            )
            for gate in circuit.queue
        )
        + (tuple(m.qubits for m in circuit.measurements),)
    )


def execute_plan(backend, plan) -> list:
    """
    Run every circuit in a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan`
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
