"""Pull noise events into the Clifford frame of an HSMPO surrogate.

A Pauli fault ``E`` after Clifford prefix ``C`` is the dressed fault
``E' = C^dag E C`` at the same point of the dressed-rotation chain. Moving it
through every later dressed rotation and the Clifford remainder to the
observable leaves the tableau untouched: it only flips the sign of the angle
of each later rotation whose generator anticommutes with ``E'``, and the sign
of the dressed observable if that anticommutes too. A coherent rotation is
dressed exactly like a magic gate and joins the chain.

A general channel ``sum_{A, B} chi[A, B] A rho B`` splits into its diagonal,
a Pauli channel compiled as above, and pairs ``(A, B)`` of its off-diagonal
entries, the coherences. A coherence acts on the ket with ``A'`` and on the
bra with ``B'``, so it flips the ket's and the bra's rotation signs
differently and contributes a transition amplitude between the two.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import stim
from qibo import gates

from mpstab.engines import StimEngine
from mpstab.engines.stabilizers.stim import gate_instruction
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.models.noise.events import (
    CoherentRotation,
    KrausChannel,
    NoiseEvent,
    PauliChannel,
)
from mpstab.pauli import (
    _anticommuting,
    _symplectic,
    anticommutation,
    conjugate,
    light_cone,
    validate_pauli_string,
)


@dataclass
class CompiledNoise:
    """
    A noisy circuit in the Clifford frame of an HSMPO surrogate.

    Build it with :func:`compile_noise`. Each location is one
    :class:`~mpstab.models.noise.PauliChannel` or
    :class:`~mpstab.models.noise.events.KrausChannel`. Each fault is a non-identity
    Pauli of a location with nonzero probability: a term of the Pauli channel,
    or a diagonal entry of the Kraus channel's process matrix. Each coherence
    is a pair of distinct Paulis with a nonzero off-diagonal entry, stored
    once for the pair. Both are stored in location order.

    Attributes
    ----------
    hsmpo : HSMPO
        The surrogate supplying the engines and the initial MPS. It is only
        read from, never modified.
    generators : list of str
        Dressed generators of the rotation chain: the magic gates and the
        coherent rotations, in circuit order.
    angles : numpy.ndarray
        Signed angle of each rotation, shape ``(R,)``.
    inverse_tableau : stim.Tableau
        ``U^dag`` for the full Clifford circuit ``U``, which dresses observables.
    fault_paulis : list of str
        Dressed Pauli of each fault, unsigned, length ``F``.
    fault_probs : numpy.ndarray
        Probability of each fault, shape ``(F,)``.
    fault_location : numpy.ndarray
        Location index of each fault, shape ``(F,)``, nondecreasing.
    location_probs : numpy.ndarray
        Probability that each location faults at all, shape ``(L,)``.
    flips : numpy.ndarray
        Boolean ``(F, R)`` matrix, ``True`` where a fault flips the sign of a
        rotation's angle.
    coherence_ket, coherence_bra : list of str
        Dressed Paulis ``A'`` and ``B'`` of each coherence, unsigned, length
        ``C``; the identity string where one of them is the identity.
    coherence_weights : numpy.ndarray
        Complex ``chi[A, B]`` of each coherence, times the dressing signs of
        ``A'`` and ``B'``, shape ``(C,)``. The pair ``(B, A)`` has the conjugate
        weight.
    coherence_location : numpy.ndarray
        Location index of each coherence, shape ``(C,)``, nondecreasing.
    ket_flips, bra_flips : numpy.ndarray
        Boolean ``(C, R)`` matrices: the rotations ``A'`` and ``B'`` flip.
    location_position : numpy.ndarray
        Rotations applied before each location, shape ``(L,)``,
        nondecreasing.
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
    coherence_ket: list[str] = field(default_factory=list)
    coherence_bra: list[str] = field(default_factory=list)
    coherence_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))
    coherence_location: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ket_flips: np.ndarray | None = None
    bra_flips: np.ndarray | None = None
    location_position: np.ndarray | None = None

    def __post_init__(self):
        empty = np.zeros((0, len(self.generators)), dtype=bool)
        if self.ket_flips is None:
            self.ket_flips = empty
        if self.bra_flips is None:
            self.bra_flips = empty

    @property
    def nqubits(self) -> int:
        return self.hsmpo.nqubits

    @property
    def coherence_mass(self) -> np.ndarray:
        """
        Total ``|chi|`` of the coherences of each location, both orientations
        counted, shape ``(L,)``.
        """
        return np.bincount(
            self.coherence_location,
            weights=2 * np.abs(self.coherence_weights),
            minlength=len(self.location_probs),
        )

    def dress_observable(self, pauli: str):
        """
        Dress a Pauli observable with the full Clifford circuit.

        Parameters
        ----------
        pauli : str
            Full-width Pauli string.

        Returns
        -------
        (str, int)
            The dressed string and its sign, as
            :meth:`~mpstab.engines.StimEngine.backpropagate` returns them.
        """
        return conjugate(pauli, self.inverse_tableau)

    def observable_flips(self, dressed_paulis: Sequence[str]) -> np.ndarray:
        """
        Which faults flip the sign of each dressed observable term.

        Parameters
        ----------
        dressed_paulis : sequence of str
            Dressed observable terms, from :meth:`dress_observable`.

        Returns
        -------
        numpy.ndarray
            Boolean ``(F, len(dressed_paulis))`` matrix.
        """
        return anticommutation(self.fault_paulis, list(dressed_paulis), self.nqubits)

    def restricted_to(self, dressed_paulis: Sequence[str]) -> CompiledNoise:
        """
        The same noisy circuit, reduced to what can affect an observable.

        Rotations outside the light cone (:func:`~mpstab.pauli.light_cone`)
        of the observable are dropped. A fault that then flips no remaining
        rotation and commutes with the observable terms leaves every
        expectation value unchanged, so it is folded into its location's
        no-fault probability, and locations left with nothing are dropped.

        A location with coherences is dropped whole when all its Paulis pass
        the same test: the channel then commutes through to the observable,
        where trace preservation makes it the identity. Otherwise its Paulis
        spread the observable, so they join the light cone, which may keep
        more rotations and locations; this is repeated until nothing changes.
        Every step is exact for this observable.

        Parameters
        ----------
        dressed_paulis : sequence of str
            Dressed observable terms, from :meth:`dress_observable`.

        Returns
        -------
        CompiledNoise
            A new instance; this one is left untouched.
        """
        dressed_paulis = list(dressed_paulis)
        if len(self.coherence_weights):
            kept, relevant, coherent = self._causal_cone(dressed_paulis)
        else:
            kept = light_cone(self.generators, dressed_paulis, self.nqubits)
            relevant = self.flips[:, kept].any(axis=1) | self.observable_flips(
                dressed_paulis
            ).any(axis=1)
            coherent = np.zeros(0, dtype=bool)

        old_locations = np.union1d(
            self.fault_location[relevant], self.coherence_location[coherent]
        )
        location = np.searchsorted(old_locations, self.fault_location[relevant])
        probs = self.fault_probs[relevant]
        return replace(
            self,
            generators=[self.generators[j] for j in kept],
            angles=self.angles[kept],
            fault_paulis=[p for p, r in zip(self.fault_paulis, relevant) if r],
            fault_probs=probs,
            fault_location=location,
            location_probs=np.bincount(
                location, weights=probs, minlength=len(old_locations)
            ),
            flips=self.flips[relevant][:, kept],
            coherence_ket=[p for p, c in zip(self.coherence_ket, coherent) if c],
            coherence_bra=[p for p, c in zip(self.coherence_bra, coherent) if c],
            coherence_weights=self.coherence_weights[coherent],
            coherence_location=np.searchsorted(
                old_locations, self.coherence_location[coherent]
            ),
            ket_flips=self.ket_flips[coherent][:, kept],
            bra_flips=self.bra_flips[coherent][:, kept],
            location_position=(
                None
                if self.location_position is None
                else np.searchsorted(kept, self.location_position[old_locations])
            ),
        )

    def _causal_cone(self, dressed_paulis: list[str]):
        """
        Walk the chain backwards from the observable, growing the set ``S`` of
        Paulis its Heisenberg-evolved form is made of:

        - a rotation is kept if it anticommutes with ``S``, and joins it;
        - a fault is relevant if it anticommutes with ``S``;
        - a location with coherences is active if any of its Paulis does, and
          then its coherences' Paulis join ``S``, as ``B O A`` spreads ``O``.

        Returns ``(kept rotations, relevant faults, coherences of active
        locations)`` as an index array and two boolean masks.
        """
        n = self.nqubits
        cone = _PauliSet(n)
        cone.add(*_symplectic(dressed_paulis, n))
        generator_x, generator_z = _symplectic(self.generators, n)
        fault_x, fault_z = _symplectic(self.fault_paulis, n)
        ket_x, ket_z = _symplectic(self.coherence_ket, n)
        bra_x, bra_z = _symplectic(self.coherence_bra, n)

        n_locations = len(self.location_probs)
        fault_starts = np.searchsorted(self.fault_location, np.arange(n_locations + 1))
        coherence_starts = np.searchsorted(
            self.coherence_location, np.arange(n_locations + 1)
        )
        kept = np.zeros(len(self.generators), dtype=bool)
        relevant = np.zeros(len(self.fault_paulis), dtype=bool)
        coherent = np.zeros(len(self.coherence_weights), dtype=bool)

        location = n_locations - 1
        for position in range(len(self.generators), -1, -1):
            while location >= 0 and self.location_position[location] == position:
                faults = slice(fault_starts[location], fault_starts[location + 1])
                coherences = slice(
                    coherence_starts[location], coherence_starts[location + 1]
                )
                relevant[faults] = cone.anticommutes(fault_x[faults], fault_z[faults])
                if coherences.stop > coherences.start and (
                    relevant[faults].any()
                    or cone.anticommutes(ket_x[coherences], ket_z[coherences]).any()
                    or cone.anticommutes(bra_x[coherences], bra_z[coherences]).any()
                ):
                    coherent[coherences] = True
                    cone.add(ket_x[coherences], ket_z[coherences])
                    cone.add(bra_x[coherences], bra_z[coherences])
                location -= 1
            j = position - 1
            if (
                j >= 0
                and cone.anticommutes(generator_x[j : j + 1], generator_z[j : j + 1])[0]
            ):
                kept[j] = True
                cone.add(generator_x[j : j + 1], generator_z[j : j + 1])
        return np.flatnonzero(kept), relevant, coherent


class _PauliSet:
    """A growing set of Pauli strings in symplectic form, for anticommutation tests."""

    def __init__(self, nqubits: int):
        self.x = np.zeros((16, nqubits))
        self.z = np.zeros((16, nqubits))
        self.size = 0

    def add(self, x: np.ndarray, z: np.ndarray):
        keep = (x.any(axis=1) | z.any(axis=1)) if len(x) else np.zeros(0, bool)
        x, z = x[keep], z[keep]
        while self.size + len(x) > len(self.x):
            self.x = np.vstack([self.x, np.zeros_like(self.x)])
            self.z = np.vstack([self.z, np.zeros_like(self.z)])
        self.x[self.size : self.size + len(x)] = x
        self.z[self.size : self.size + len(z)] = z
        self.size += len(x)

    def anticommutes(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Whether each of the given strings anticommutes with any in the set."""
        if self.size == 0 or len(x) == 0:
            return np.zeros(len(x), dtype=bool)
        return _anticommuting(x, z, self.x[: self.size], self.z[: self.size]).any(1)


def compile_noise(hsmpo: HSMPO, events: Sequence[NoiseEvent]) -> CompiledNoise:
    """
    Dress the noise events of a circuit with its Clifford frame.

    One pass of a ``stim.TableauSimulator`` over the circuit dresses every magic
    gate, coherent rotation and Pauli fault at its own position, and the flip
    matrix then follows from the symplectic form in one matrix product.

    Parameters
    ----------
    hsmpo : HSMPO
        Surrogate of the noiseless circuit. Its ``ansatz.circuit.queue`` is what
        ``after_gate`` indexes. It may be built with
        :meth:`~mpstab.HSMPO.rotations_only`.
    events : sequence of NoiseEvent
        The noise, as :class:`~mpstab.models.noise.events.PauliChannel`,
        :class:`~mpstab.models.noise.events.CoherentRotation` and
        :class:`~mpstab.models.noise.events.KrausChannel` events.

    Returns
    -------
    CompiledNoise

    Raises
    ------
    NotImplementedError
        On an event type this function does not know, or a stabilizers engine
        other than :class:`~mpstab.engines.StimEngine`.
    ValueError
        On an ``after_gate`` outside the circuit, or an ansatz whose own
        partitioning does not follow its circuit queue (``FloquetAnsatz``).

    Examples
    --------
    ::

        from qibo import Circuit, gates
        from mpstab import HSMPO
        from mpstab.models.noise.compile import compile_noise
        from mpstab.models.noise.events import PauliChannel

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
        if not isinstance(event, (PauliChannel, CoherentRotation, KrausChannel)):
            raise NotImplementedError(
                f"{type(event).__name__} is not supported: compile_noise handles "
                "PauliChannel, CoherentRotation and KrausChannel."
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
    location_probs, location_position = [], []
    coherences = []  # (ket, bra, weight, location, position)
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
            if isinstance(event, KrausChannel):
                terms, pairs = _kraus_terms(event, nqubits)
            else:
                terms, pairs = event.full_width_terms(nqubits), []
            terms = [(p, prob) for p, prob in terms if prob > 0]
            if not terms and not pairs:
                continue
            location = len(location_probs)
            for pauli, prob in terms:
                fault_paulis.append(conjugate(pauli, inverse)[0])
                fault_probs.append(prob)
                fault_location.append(location)
                fault_position.append(len(generators))
            for a, b, weight in pairs:
                ket, ket_sign = conjugate(a, inverse)
                bra, bra_sign = conjugate(b, inverse)
                weight *= ket_sign * bra_sign
                coherences.append((ket, bra, weight, location, len(generators)))
            location_probs.append(sum(prob for _, prob in terms))
            location_position.append(len(generators))

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

    def later_flips(paulis, positions):
        flips = anticommutation(paulis, generators, nqubits)
        # A fault only flips the rotations that come after it.
        return flips & (
            np.array(positions, dtype=int)[:, None] <= np.arange(len(generators))
        )

    columns = [list(c) for c in zip(*coherences)] or [[] for _ in range(5)]
    kets, bras, weights, locations, positions = columns

    return CompiledNoise(
        hsmpo=hsmpo,
        generators=generators,
        angles=np.array(angles, dtype=float),
        inverse_tableau=simulator.current_inverse_tableau(),
        fault_paulis=fault_paulis,
        fault_probs=np.array(fault_probs, dtype=float),
        fault_location=np.array(fault_location, dtype=int),
        location_probs=np.array(location_probs, dtype=float),
        flips=later_flips(fault_paulis, fault_position),
        coherence_ket=kets,
        coherence_bra=bras,
        coherence_weights=np.array(weights, dtype=complex),
        coherence_location=np.array(locations, dtype=int),
        ket_flips=later_flips(kets, positions),
        bra_flips=later_flips(bras, positions),
        location_position=np.array(location_position, dtype=int),
    )


def _kraus_terms(event: KrausChannel, nqubits: int):
    """
    A Kraus channel as full-width ``[(pauli, probability)]`` for the diagonal
    of its process matrix and ``[(A, B, chi[A, B])]`` for each pair of
    distinct Paulis with a nonzero off-diagonal entry, ``A`` before ``B``.
    """
    if max(event.qubits, default=-1) >= nqubits:
        raise ValueError(
            f"KrausChannel acts on qubits {event.qubits}, outside a "
            f"{nqubits}-qubit circuit."
        )

    def full(label: str) -> str:
        labels = ["I"] * nqubits
        for qubit, pauli in zip(event.qubits, label):
            labels[qubit] = pauli
        return "".join(labels)

    terms, pairs = [], []
    for (a, b), value in event.chi.items():
        if a == b:
            if set(a) != {"I"}:
                terms.append((full(a), float(value.real)))
        elif a < b:
            pairs.append((full(a), full(b), complex(value)))
    return terms, pairs
