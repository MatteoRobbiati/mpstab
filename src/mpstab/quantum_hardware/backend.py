"""
The backend contract for running a :class:`~mpstab.quantum_hardware.plan.MeasurementPlan`.

A backend is any object with an ``execute_circuits(circuits, nshots)`` method
returning one frequency dictionary per circuit, as qibo's
``Circuit.__call__(nshots=...).frequencies()`` does; there is no base class to
subclass. :class:`QiboSimulator` below is the default.

Bitstrings are qubit-0-leftmost throughout mpstab, as in qibo. A backend whose
provider reports them the other way round, qubit 0 rightmost as in qiskit, must
reverse them before returning.

:func:`circuit_to_qasm` and :func:`plan_to_qasm` write a plan's circuits as
OpenQASM 2, the form a device backend works on;
:func:`~mpstab.quantum_hardware.storage.save_circuits` writes them to disk for
an offline run.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from qibo import gates


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
        One frequency dict per circuit, in ``plan.circuits`` order and
        qubit-0-leftmost -- what :mod:`~mpstab.quantum_hardware.estimate`
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


#: The gates of the original OpenQASM 2 ``qelib1.inc`` (Cross et al., 2017),
#: the set every OpenQASM 2 parser knows. Later additions such as ``sx`` are
#: missing from some parsers' default include (qiskit's needs
#: ``LEGACY_CUSTOM_INSTRUCTIONS``), so :func:`circuit_to_qasm` emits only these.
QELIB1_GATES = frozenset({
    "u3", "u2", "u1", "cx", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "cz", "cy", "ch", "ccx", "crz", "cu1", "cu3",
})


def _u3_angles(matrix) -> tuple:
    """
    ``(theta, phi, lam)`` with ``matrix = e^{i alpha} u3(theta, phi, lam)``, in
    OpenQASM's ``u3`` convention. The global phase ``alpha`` is dropped: it
    changes no measurement outcome.
    """
    u = np.asarray(matrix, dtype=complex)
    u = u / np.sqrt(np.linalg.det(u))  # SU(2): [[a, -b*], [b, a*]]
    theta = 2 * np.arctan2(abs(u[1, 0]), abs(u[0, 0]))
    plus = 2 * np.angle(u[1, 1]) if abs(u[1, 1]) > 1e-12 else 0.0  # phi + lam
    minus = 2 * np.angle(u[1, 0]) if abs(u[1, 0]) > 1e-12 else 0.0  # phi - lam
    return float(theta), float((plus + minus) / 2), float((plus - minus) / 2)


def circuit_to_qasm(circuit) -> str:
    """
    OpenQASM 2 text of a qibo circuit, e.g. one of a plan's circuits.

    Only the gates of the original ``qelib1.inc`` (:data:`QELIB1_GATES`) are
    written by name, so any OpenQASM 2 parser reads the text without custom
    definitions. Every other one-qubit gate becomes a ``u3`` gate equal to it
    up to a global phase: the ``Unitary`` basis changes a plan adds before
    measurement, which qibo's own ``Circuit.to_qasm()`` rejects, and gates
    such as ``SX``/``SXDG`` that rustiq's resynthesis emits, which qiskit's
    parser rejects by default. Qubit ``i`` is measured into ``c[i]``, so a
    provider reporting counts as ``c[n-1]...c[0]`` (as qiskit does) returns
    them qubit 0 rightmost: reverse them before estimating.

    This only translates the circuit: mapping it to a device's native gates
    and qubits is the provider's transpiler's job.

    Raises:
        ValueError: on a gate acting on more than one qubit that is not in
            :data:`QELIB1_GATES`.
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{circuit.nqubits}];",
        f"creg c[{circuit.nqubits}];",
    ]
    for gate in circuit.queue:
        if isinstance(gate, gates.M):
            lines += [f"measure q[{q}] -> c[{q}];" for q in gate.target_qubits]
            continue
        label = _qelib1_label(gate)
        if label is not None:
            params = ",".join(repr(float(p)) for p in gate.parameters)
            args = ",".join(f"q[{q}]" for q in gate.qubits)
            lines.append(f"{label}{f'({params})' if params else ''} {args};")
        elif len(gate.qubits) == 1:
            matrix = gate.parameters[0] if isinstance(gate, gates.Unitary) else gate.matrix()
            theta, phi, lam = _u3_angles(matrix)
            lines.append(f"u3({theta!r},{phi!r},{lam!r}) q[{gate.qubits[0]}];")
        else:
            raise ValueError(
                f"{type(gate).__name__} on qubits {gate.qubits} has no form in the "
                "original qelib1.inc, and only one-qubit gates can be rewritten as u3."
            )
    return "\n".join(lines) + "\n"


def _qelib1_label(gate):
    """``gate``'s OpenQASM name if it is in :data:`QELIB1_GATES`, else ``None``."""
    if isinstance(gate, gates.Unitary):
        return None
    try:
        label = gate.qasm_label
    except NotImplementedError:
        return None
    return label if label in QELIB1_GATES else None


def plan_to_qasm(plan) -> list:
    """
    :func:`circuit_to_qasm` of every circuit of a plan, in ``plan.circuits``
    order: circuit ``i`` runs ``plan.shots[i]`` shots, and its counts must come
    back in position ``i`` for :func:`~mpstab.quantum_hardware.estimate.estimate`.
    """
    return [circuit_to_qasm(circuit) for circuit in plan.circuits]
