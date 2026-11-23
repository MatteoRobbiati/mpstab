"""A noisy stand-in for a device, to validate error mitigation on a plan."""

from __future__ import annotations

from collections import Counter

import numpy as np
from qibo import Circuit

from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    _circuit_key,
    _gates_key,
    _split_basis_measurement,
)


class DensityMatrixNoiseSimulator:
    """
    Runs plan circuits under a :class:`~mpstab.models.noise.NoiseModel`, on a
    qibo density matrix.

    Each circuit's preparation and basis rotations are run as one noisy
    circuit (:meth:`~mpstab.models.noise.NoiseModel.apply`), so the basis
    rotations of the measurement layer are as noisy as any other gate. Readout
    errors then flip the sampled bits. Density matrices limit it to small
    circuits, about ten qubits: it is a reference for testing, not a
    simulator for production runs.

    Args:
        noise: the noise model.
        seed: seed of the sampled shots and readout flips.
    """

    def __init__(self, noise, seed: int | None = None):
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._probabilities: dict = {}
        self._states: dict = {}

    def execute_circuits(self, circuits, nshots: int) -> list:
        return [self._sample(circuit, nshots) for circuit in circuits]

    def _sample(self, circuit, nshots: int) -> Counter:
        split = _split_basis_measurement(circuit)
        if split is None:
            raise ValueError(
                "DensityMatrixNoiseSimulator runs circuits ending in one "
                "measurement layer, as measurement plans build them."
            )
        key = _circuit_key(circuit)
        if key not in self._probabilities:
            self._probabilities[key] = self._outcome_probabilities(circuit, *split)
        probabilities = self._probabilities[key]
        measured = split[2]

        outcomes = self.rng.choice(len(probabilities), size=nshots, p=probabilities)
        bits = (outcomes[:, None] >> np.arange(len(measured))[::-1]) & 1
        bits = self._readout(bits, measured)
        return Counter("".join(map(str, row)) for row in bits)

    def _outcome_probabilities(self, circuit, preparation, basis, measured):
        """
        The noisy state is built in two steps, the preparation and then the
        basis rotations, so that plan circuits sharing a preparation simulate
        it once. The noise events are those of the whole circuit, split at the
        preparation's end, so the result is the same as in one step.
        """
        from mpstab.models.noise.model import _qibo_gate

        nqubits = circuit.nqubits
        unitary = Circuit(nqubits)
        for gate in (*preparation, *basis):
            unitary.add(gate)
        by_gate: dict = {}
        for event in self.noise.events(unitary):
            by_gate.setdefault(event.after_gate, []).append(event)
        split = len(preparation)

        prepared_events = tuple(
            repr(event)
            for index in range(-1, split)
            for event in by_gate.get(index, ())
        )
        state_key = (_gates_key(nqubits, preparation), prepared_events)
        if state_key not in self._states:
            noisy = Circuit(nqubits, density_matrix=True)
            for event in by_gate.get(-1, ()):
                noisy.add(_qibo_gate(event))
            for index, gate in enumerate(preparation):
                noisy.add(gate)
                for event in by_gate.get(index, ()):
                    noisy.add(_qibo_gate(event))
            self._states[state_key] = np.asarray(noisy().state())

        rho = self._states[state_key]
        rotations = Circuit(nqubits, density_matrix=True)
        for index, gate in enumerate(basis, start=split):
            rotations.add(gate)
            for event in by_gate.get(index, ()):
                rotations.add(_qibo_gate(event))
        if rotations.queue:
            rho = np.asarray(rotations(initial_state=rho.copy()).state())
        return _marginal(rho, nqubits, measured)

    def _readout(self, bits: np.ndarray, measured) -> np.ndarray:
        """Flip each measured bit with its readout error probability."""
        scale, offset = self.noise.readout_maps(max(measured) + 1)
        p0 = (1 - scale - offset) / 2  # 0 read as 1
        p1 = (1 - scale + offset) / 2  # 1 read as 0
        flip_probability = np.where(bits == 0, p0[list(measured)], p1[list(measured)])
        flips = self.rng.random(bits.shape) < flip_probability
        return bits ^ flips


def _marginal(rho, nqubits: int, measured) -> np.ndarray:
    """Outcome probabilities of ``measured``, in that order, from a density matrix."""
    probabilities = np.clip(np.real(np.diag(np.asarray(rho))), 0, None)
    probabilities = probabilities.reshape([2] * nqubits)
    others = tuple(q for q in range(nqubits) if q not in measured)
    probabilities = probabilities.sum(axis=others) if others else probabilities
    kept = [q for q in range(nqubits) if q in measured]
    probabilities = np.transpose(probabilities, [kept.index(q) for q in measured])
    probabilities = probabilities.reshape(-1)
    return probabilities / probabilities.sum()
