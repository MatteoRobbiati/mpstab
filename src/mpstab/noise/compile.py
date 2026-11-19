"""Pull noise events into the Clifford frame of an HSMPO surrogate.

A Pauli fault ``E`` after Clifford prefix ``C`` is the dressed fault
``E' = C^dag E C`` at the same point of the dressed-rotation chain. Moving it
through every later dressed rotation and the Clifford remainder to the
observable leaves the tableau untouched: it only flips the sign of the angle
of each later rotation whose generator anticommutes with ``E'``, and the sign
of the dressed observable if that anticommutes too. A coherent rotation is
dressed exactly like a magic gate and joins the chain.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np
import stim
from qibo import gates

from mpstab.engines import StimEngine
from mpstab.engines.stabilizers.stim import gate_instruction
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.noise.events import CoherentRotation, NoiseEvent, PauliChannel
from mpstab.pauli import anticommutation, conjugate, light_cone, validate_pauli_string

if TYPE_CHECKING:
    from mpstab.evolutors.hsmpo import HSMPO


@dataclass
class CompiledNoise:
    """
    A noisy circuit in the Clifford frame of an HSMPO surrogate.

    Build it with :func:`compile_noise`. Each location is one
    :class:`~mpstab.noise.PauliChannel`, and each fault is one of its
    non-identity terms with nonzero probability; faults are stored in
    location order.

    Attributes:
        hsmpo: the surrogate supplying the engines and the initial MPS. It is
            only read from, never modified.
        generators: dressed generators of the rotation chain: the magic gates
            and the coherent rotations, in circuit order.
        angles: signed angle of each rotation, shape ``(R,)``.
        inverse_tableau: ``U^dag`` for the full Clifford circuit ``U``, which
            dresses observables.
        fault_paulis: dressed Pauli of each fault, unsigned, length ``F``.
        fault_probs: probability of each fault, shape ``(F,)``.
        fault_location: location index of each fault, shape ``(F,)``,
            nondecreasing.
        location_probs: probability that each location faults at all, shape
            ``(L,)``.
        flips: boolean ``(F, R)`` matrix, ``True`` where a fault flips the sign
            of a rotation's angle.
    """

    hsmpo: HSMPO
    generators: list[str]
    angles: np.ndarray
    inverse_tableau: stim.Tableau
    fault_paulis: list[str]
    fault_probs: np.ndarray
    fault_location: np.ndarray
    location_probs: np.ndarray
    flips: np.ndarray

    @property
    def nqubits(self) -> int:
        """Width of the circuit."""
        return self.hsmpo.nqubits

    def dress_observable(self, pauli: str):
        """
        Dress a Pauli observable with the full Clifford circuit.

        Args:
            pauli: full-width Pauli string.

        Returns:
            The dressed string and its sign, as
            :meth:`~mpstab.engines.StimEngine.backpropagate` returns them.
        """
        return conjugate(pauli, self.inverse_tableau)

    def observable_flips(self, dressed_paulis: Sequence[str]) -> np.ndarray:
        """
        Which faults flip the sign of each dressed observable term.

        Args:
            dressed_paulis: dressed observable terms, from
                :meth:`dress_observable`.

        Returns:
            Boolean ``(F, len(dressed_paulis))`` matrix.
        """
        return anticommutation(self.fault_paulis, list(dressed_paulis), self.nqubits)

    def restricted_to(self, dressed_paulis: Sequence[str]) -> CompiledNoise:
        """
        The same noisy circuit, reduced to what can affect an observable.

        Rotations outside the observable's light cone
        (:func:`~mpstab.pauli.light_cone`) are dropped. A fault that then flips
        neither a remaining rotation nor an observable term leaves every
        expectation value unchanged, so it is folded into its location's
        no-fault probability, and locations left with no fault are dropped.
        Both steps are exact for this observable.

        Args:
            dressed_paulis: dressed observable terms, from
                :meth:`dress_observable`.

        Returns:
            A new :class:`CompiledNoise`; this one is left untouched.
        """
        dressed_paulis = list(dressed_paulis)
        kept = light_cone(self.generators, dressed_paulis, self.nqubits)
        flips = self.flips[:, kept]
        relevant = flips.any(axis=1) | self.observable_flips(dressed_paulis).any(axis=1)
        _, location = np.unique(self.fault_location[relevant], return_inverse=True)
        location = location.reshape(-1)
        probs = self.fault_probs[relevant]
        return replace(
            self,
            generators=[self.generators[j] for j in kept],
            angles=self.angles[kept],
            fault_paulis=[p for p, r in zip(self.fault_paulis, relevant) if r],
            fault_probs=probs,
            fault_location=location,
            location_probs=np.bincount(
                location, weights=probs, minlength=location.max(initial=-1) + 1
            ),
            flips=flips[relevant],
        )


def compile_noise(hsmpo: HSMPO, events: Sequence[NoiseEvent]) -> CompiledNoise:
    """
    Dress the noise events of a circuit with its Clifford frame.

    One pass of a ``stim.TableauSimulator`` over the circuit dresses every magic
    gate, coherent rotation and Pauli fault at its own position, and the flip
    matrix then follows from the symplectic form in one matrix product.

    Args:
        hsmpo: surrogate of the noiseless circuit. Its ``ansatz.circuit.queue``
            is what ``after_gate`` indexes. It may be built with
            :meth:`~mpstab.HSMPO.rotations_only`.
        events: the noise, as :class:`~mpstab.noise.PauliChannel` and
            :class:`~mpstab.noise.CoherentRotation` events.

    Returns:
        The :class:`CompiledNoise`.

    Raises:
        NotImplementedError: on an event type this function does not know, or a
            stabilizers engine other than :class:`~mpstab.engines.StimEngine`.
        ValueError: on an ``after_gate`` outside the circuit, or an ansatz whose
            own partitioning does not follow its circuit queue
            (``FloquetAnsatz``).

    Example::

        from qibo import Circuit, gates
        from mpstab import HSMPO
        from mpstab.noise.compile import compile_noise
        from mpstab.noise.events import PauliChannel

        circuit = Circuit(2)
        circuit.add(gates.H(0))
        circuit.add(gates.RZ(0, theta=0.3))
        circuit.add(gates.CNOT(0, 1))
        # A bit flip between the H and the RZ anticommutes with the RZ.
        events = [PauliChannel(after_gate=0, qubits=(0,), probs={"X": 0.01})]
        compiled = compile_noise(HSMPO(circuit), events)
        print(compiled.generators, compiled.flips)  # ['XI'] [[ True]]
    """
    if not isinstance(hsmpo.stab_engine, StimEngine):
        raise NotImplementedError(
            "compile_noise requires StimEngine; call "
            "set_engines(stab_engine=StimEngine()) first."
        )
    nqubits = hsmpo.nqubits
    queue = hsmpo.ansatz.circuit.queue

    by_gate = defaultdict(list)
    for event in events:
        if not isinstance(event, (PauliChannel, CoherentRotation)):
            raise NotImplementedError(
                f"{type(event).__name__} is not supported: compile_noise handles "
                "PauliChannel and CoherentRotation. General Kraus channels need "
                "a trajectory method, which is not implemented."
            )
        if not -1 <= event.after_gate < len(queue):
            raise ValueError(
                f"{event} follows gate {event.after_gate}, but the circuit has "
                f"gates 0 to {len(queue) - 1} (-1 for the initial state)."
            )
        if isinstance(event, CoherentRotation):
            validate_pauli_string(event.generator, nqubits)
        by_gate[event.after_gate].append(event)

    simulator = stim.TableauSimulator()
    simulator.do(stim.Circuit(f"I {nqubits - 1}"))

    generators, angles = [], []
    fault_paulis, fault_probs, fault_location, fault_position = [], [], [], []
    location_probs = []
    magic_gates = []

    def add_rotation(generator: str, angle: float, inverse: stim.Tableau):
        dressed, sign = conjugate(generator, inverse)
        generators.append(dressed)
        angles.append(sign * angle)

    def add_events(index: int):
        if index not in by_gate:
            return
        inverse = simulator.current_inverse_tableau()
        for event in by_gate[index]:
            if isinstance(event, CoherentRotation):
                add_rotation(event.generator, event.angle, inverse)
                continue
            terms = [
                (p, prob) for p, prob in event.full_width_terms(nqubits) if prob > 0
            ]
            if not terms:
                continue
            for pauli, prob in terms:
                fault_paulis.append(conjugate(pauli, inverse)[0])
                fault_probs.append(prob)
                fault_location.append(len(location_probs))
                fault_position.append(len(generators))
            location_probs.append(sum(prob for _, prob in terms))

    add_events(-1)
    for index, gate in enumerate(queue):
        if not gate.clifford and not isinstance(gate, gates.M):
            magic_gates.append(gate)
            add_rotation(
                gate_generator(gate, nqubits),
                gate_angle(gate),
                simulator.current_inverse_tableau(),
            )
        else:
            instruction = gate_instruction(gate)
            if instruction is not None:
                simulator.do(stim.Circuit(instruction))
        add_events(index)

    expected = [gate for _, gate in hsmpo.magic_gates]
    if len(expected) != len(magic_gates) or any(
        a is not b for a, b in zip(expected, magic_gates)
    ):
        raise ValueError(
            f"{type(hsmpo.ansatz).__name__} partitions its circuit differently "
            "from its gate queue, so noise positions cannot be placed on it."
        )

    flips = anticommutation(fault_paulis, generators, nqubits)
    # A fault only flips the rotations that come after it.
    flips &= np.array(fault_position)[:, None] <= np.arange(len(generators))[None, :]

    return CompiledNoise(
        hsmpo=hsmpo,
        generators=generators,
        angles=np.array(angles, dtype=float),
        inverse_tableau=simulator.current_inverse_tableau(),
        fault_paulis=fault_paulis,
        fault_probs=np.array(fault_probs, dtype=float),
        fault_location=np.array(fault_location, dtype=int),
        location_probs=np.array(location_probs, dtype=float),
        flips=flips,
    )
