"""
:class:`StatevectorExactSimulator`, a backend running measurement-plan circuits
with qibo's exact statevector simulator, as
:func:`~mpstab.qpu.execution.backend.run_plan` expects:
``execute_circuits(circuits, nshots)`` returning one frequency dict per circuit.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict

import numpy as np
from qibo import Circuit, gates


class StatevectorExactSimulator:
    """
    Runs circuits with qibo's exact statevector simulator. The default backend.

    Measurement plans run one state preparation under many measurement bases,
    so circuits are grouped by their preparation, everything before the
    basis rotations of their final ``gates.M``: each preparation is simulated
    once, and each circuit then only rotates that state into its own basis
    and samples it. Every circuit still gets its own ``nshots`` independent
    samples, so the results are distributed exactly as if every circuit had
    been run separately. Circuits of any other shape (density matrices, noise
    channels, several or mid-circuit measurements) are simulated as they are,
    identical ones once.
    """

    def execute_circuits(self, circuits, nshots: int) -> list:
        nqubits = circuits[0].nqubits
        if nqubits > 30:
            warnings.warn(
                f"{nqubits} qubits: exact simulation is impractical above ~30."
            )
        frequencies: list = [None] * len(circuits)
        preparations: dict = defaultdict(list)
        others = []
        splits: dict = {}  # plans often repeat the same circuit object
        for index, circuit in enumerate(circuits):
            if id(circuit) not in splits:
                splits[id(circuit)] = _split_basis_measurement(circuit)
            split = splits[id(circuit)]
            if split is None:
                others.append(index)
            else:
                preparations[_gates_key(nqubits, split[0])].append((index, split))

        for members in preparations.values():
            state = _prepared_state(nqubits, members[0][1][0])
            probabilities: dict = {}  # per distinct basis and measured qubits
            for index, (_, basis, measured) in members:
                key = (_gates_key(nqubits, basis), measured)
                if key not in probabilities:
                    probabilities[key] = _measured_probabilities(state, basis, measured)
                frequencies[index] = _sample(probabilities[key], len(measured), nshots)

        for index, result in zip(
            others, self._execute_as_is([circuits[i] for i in others], nshots)
        ):
            frequencies[index] = result
        return frequencies

    @staticmethod
    def _execute_as_is(circuits, nshots: int) -> list:
        """Simulate each distinct circuit once, splitting its samples."""
        groups: dict = defaultdict(list)
        keys: dict = {}
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


def _gate_key(gate) -> tuple:
    return (
        type(gate).__name__,
        gate.qubits,
        tuple(np.ravel(gate.parameters)) if gate.parameters else (),
    )


def _gates_key(nqubits: int, gates_) -> tuple:
    return (nqubits,) + tuple(_gate_key(gate) for gate in gates_)


def _circuit_key(circuit) -> tuple:
    """Everything that determines a circuit's output distribution."""
    return (
        (circuit.nqubits, circuit.density_matrix)
        + tuple(_gate_key(gate) for gate in circuit.queue)
        + (tuple(m.qubits for m in circuit.measurements),)
    )


def _split_basis_measurement(circuit):
    """
    ``(preparation, basis rotations, measured qubits)`` of a pure-state circuit
    ending in a single ``gates.M`` whose basis rotations close its queue, or
    ``None`` for any other circuit.
    """
    if circuit.density_matrix or len(circuit.measurements) != 1:
        return None
    measurement = circuit.measurements[0]
    queue = list(circuit.queue)
    if not queue or queue[-1] is not measurement or measurement.collapse:
        return None
    basis = list(measurement.basis or [])
    n_basis = len(basis)
    if any(q is not b for q, b in zip(queue[len(queue) - 1 - n_basis : -1], basis)):
        return None
    preparation = queue[: len(queue) - 1 - n_basis]
    if any(isinstance(gate, (gates.Channel, gates.M)) for gate in preparation):
        return None
    return tuple(preparation), tuple(basis), tuple(measurement.qubits)


def _prepared_state(nqubits: int, preparation) -> np.ndarray:
    """The state ``preparation`` leaves ``|0...0>`` in, as a ``[2] * n`` array."""
    circuit = Circuit(nqubits)
    for gate in preparation:
        circuit.add(gate)
    state = np.asarray(circuit().state(), dtype=complex)
    return state.reshape([2] * nqubits)


def _measured_probabilities(state, basis, measured) -> np.ndarray:
    """Outcome probabilities of ``measured`` (in that order) after ``basis``."""
    for gate in basis:
        targets = list(gate.qubits)
        k = len(targets)
        matrix = np.asarray(gate.matrix(), dtype=complex).reshape([2] * (2 * k))
        state = np.tensordot(matrix, state, axes=(list(range(k, 2 * k)), targets))
        state = np.moveaxis(state, list(range(k)), targets)
    probabilities = np.abs(state) ** 2
    others = tuple(q for q in range(state.ndim) if q not in measured)
    marginal = probabilities.sum(axis=others) if others else probabilities
    # The marginal keeps the measured axes in ascending order.
    order = [sorted(measured).index(q) for q in measured]
    marginal = np.transpose(marginal, order).ravel()
    return marginal / marginal.sum()


def _sample(probabilities, n_measured: int, nshots: int) -> Counter:
    """``nshots`` samples as qibo's ``frequencies()``: bitstring -> count."""
    counts = np.random.multinomial(nshots, probabilities)
    return Counter(
        {
            format(outcome, f"0{n_measured}b"): int(counts[outcome])
            for outcome in np.flatnonzero(counts)
        }
    )
