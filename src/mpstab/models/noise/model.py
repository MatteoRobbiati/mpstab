"""A gate-attached noise model, built with :meth:`NoiseModel.add` as in qibo."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Iterable, Sequence, Union

import numpy as np
from qibo import Circuit, gates

from mpstab.hamiltonians import pauli_terms
from mpstab.models.noise.events import (
    CoherentRotation,
    KrausChannel,
    NoiseEvent,
    PauliChannel,
)
from mpstab.pauli import pauli_matrix, validate_pauli_string

# -- Errors acting after gates ----------------------------------------------------


@dataclass(frozen=True)
class PauliError:
    """
    Stochastic Pauli error: ``probs[P]`` is the probability that ``P`` acts.

    Single-qubit labels (``{"X": 0.01, "Z": 0.02}``) act independently on each
    qubit of the gate. Labels as wide as the gate (``{"ZZ": 0.01}`` after a
    two-qubit gate) act jointly on its qubits, in the gate's qubit order, which
    is how correlated noise is written.

    Attributes:
        probs: probability of each non-identity Pauli label.

    Example::

        from mpstab.models.noise import PauliError

        dephasing = PauliError({"Z": 0.01})
        correlated = PauliError({"ZZ": 0.005, "XX": 0.001})
    """

    probs: dict[str, float]

    def __post_init__(self):
        widths = {len(label) for label in self.probs}
        if len(widths) > 1:
            raise ValueError(f"PauliError labels have different widths: {self.probs}.")
        for label, prob in self.probs.items():
            validate_pauli_string(label)
            if prob < 0:
                raise ValueError(
                    f"PauliError term {label!r} has probability {prob} < 0."
                )

    @property
    def width(self) -> int:
        """Qubits each label acts on: 1 per qubit of the gate, or the whole gate."""
        return len(next(iter(self.probs), "X"))

    def scaled(self, factor: float) -> PauliError:
        return PauliError({label: factor * p for label, p in self.probs.items()})

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        return PauliChannel(index, support, dict(self.probs))


@dataclass(frozen=True)
class DepolarizingError:
    """
    Depolarizing channel ``(1 - lam) rho + lam I / 2**k`` on the ``k`` qubits of
    the gate, with qibo's convention: probability ``lam / 4**k`` on each of the
    ``4**k - 1`` non-identity Paulis.

    For independent single-qubit depolarizing on every qubit of a multi-qubit
    gate, use ``PauliError({"X": p / 3, "Y": p / 3, "Z": p / 3})``. On idle
    qubits and spectators (:meth:`NoiseModel.add_idle`,
    :meth:`NoiseModel.add_crosstalk`) it acts on each qubit alone.

    Attributes:
        lam: depolarizing strength, in ``[0, 4/3]``.

    Example::

        from mpstab.models.noise import DepolarizingError

        error = DepolarizingError(0.01)
    """

    lam: float

    def __post_init__(self):
        if not 0 <= self.lam <= 4 / 3:
            raise ValueError(f"DepolarizingError lam={self.lam} outside [0, 4/3].")

    #: Always acts on the whole gate.
    width = None

    def scaled(self, factor: float) -> DepolarizingError:
        return DepolarizingError(factor * self.lam)

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        k = len(support)
        labels = ("".join(p) for p in product("IXYZ", repeat=k))
        probs = {label: self.lam / 4**k for label in labels if set(label) != {"I"}}
        return PauliChannel(index, support, probs)


@dataclass(frozen=True)
class CoherentError:
    """
    Coherent error ``exp(-i angle/2 P)``, such as an over-rotation or a residual
    ZZ coupling.

    A single-qubit ``generator`` acts on each qubit of the gate; one as wide as
    the gate acts jointly on its qubits, in the gate's qubit order. Each one
    joins the rotation chain like a magic gate, so many of them cost bond
    dimension: :meth:`twirled` is the stochastic Pauli error the same rotation
    becomes under randomized compiling, which costs none.

    Attributes:
        generator: Pauli label of the rotation, one qubit or as wide as the gate.
        angle: rotation angle.

    Example::

        from mpstab.models.noise import CoherentError

        residual_zz = CoherentError("ZZ", 0.02)
        over_rotation = CoherentError("X", 0.01)
        print(residual_zz.twirled())
    """

    generator: str
    angle: float

    def __post_init__(self):
        validate_pauli_string(self.generator)

    @property
    def width(self) -> int:
        """Qubits the generator acts on: 1 per qubit of the gate, or the whole gate."""
        return len(self.generator)

    def scaled(self, factor: float) -> CoherentError:
        return CoherentError(self.generator, factor * self.angle)

    def twirled(self) -> PauliError:
        """
        The Pauli twirl of the rotation, ``rho -> cos^2 rho + sin^2 P rho P``
        with ``sin^2 = sin(angle/2)**2``.
        """
        return PauliError({self.generator: float(np.sin(self.angle / 2) ** 2)})

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        labels = ["I"] * nqubits
        for q, p in zip(support, self.generator):
            labels[q] = p
        return CoherentRotation(index, "".join(labels), self.angle)


@dataclass(frozen=True, eq=False)
class KrausError:
    """
    Any channel, given by its Kraus operators: ``rho -> sum_k K_k rho K_k^dag``.

    Operators on one qubit act on each qubit of the gate; operators as wide as
    the gate act jointly on its qubits, the first of them the most significant
    factor, as in qibo. The channel is split into its Pauli twirl, sampled as
    Pauli faults, and the off-diagonal part of its process matrix, evaluated
    as transition amplitudes (:class:`~mpstab.models.noise.events.KrausChannel`).

    Attributes:
        operators: the Kraus operators, each ``2**k x 2**k``.

    Example::

        # A phase-damping channel.
        import numpy as np
        from mpstab.models.noise import KrausError

        lam = 0.05
        phase_damping = KrausError([
            np.diag([1, np.sqrt(1 - lam)]),
            np.diag([0, np.sqrt(lam)]),
        ])
    """

    operators: Sequence[np.ndarray]

    def __post_init__(self):
        operators = tuple(np.asarray(k, dtype=complex) for k in self.operators)
        dim = operators[0].shape[0] if operators else 0
        width = int(np.log2(dim)) if dim else 0
        if dim < 2 or 2**width != dim:
            raise ValueError("KrausError operators must be 2**k x 2**k matrices.")
        object.__setattr__(self, "operators", operators)
        KrausChannel(0, tuple(range(width)), operators)  # validates

    @property
    def width(self) -> int:
        return int(np.log2(self.operators[0].shape[0]))

    def scaled(self, factor: float) -> KrausError:
        raise NotImplementedError(
            "A KrausError has no strength parameter to scale; use "
            "AmplitudeDampingError or ThermalRelaxationError, or rebuild it."
        )

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        return KrausChannel(index, support, self.operators)


@dataclass(frozen=True)
class AmplitudeDampingError:
    """
    Amplitude damping ``|1> -> |0>`` with probability ``gamma`` on each qubit
    of the gate: energy relaxation (T1), which is not unital.

    Attributes:
        gamma: damping probability, in ``[0, 1]``.

    Example::

        from mpstab.models.noise import AmplitudeDampingError

        t1 = AmplitudeDampingError(0.01)
    """

    gamma: float

    def __post_init__(self):
        if not 0 <= self.gamma <= 1:
            raise ValueError(
                f"AmplitudeDampingError gamma={self.gamma} outside [0, 1]."
            )

    width = 1

    def scaled(self, factor: float) -> AmplitudeDampingError:
        return AmplitudeDampingError(factor * self.gamma)

    @property
    def kraus(self) -> tuple[np.ndarray, ...]:
        g = self.gamma
        return (
            np.array([[1, 0], [0, np.sqrt(1 - g)]], dtype=complex),
            np.array([[0, np.sqrt(g)], [0, 0]], dtype=complex),
        )

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        return KrausChannel(index, support, self.kraus)


@dataclass(frozen=True)
class ThermalRelaxationError:
    """
    Thermal relaxation for a duration ``time`` on each qubit of the gate: the
    excited population relaxes to ``excited_population`` with time constant
    ``t1``, and coherences decay as ``exp(-time / t2)``.

    It is generalized amplitude damping with ``gamma = 1 - exp(-time / t1)``
    followed by the pure dephasing that brings the coherence decay from
    ``exp(-time / (2 t1))`` down to ``exp(-time / t2)``, so ``t2 <= 2 t1``.
    :meth:`scaled` scales ``time``.

    Attributes:
        t1: energy relaxation time.
        t2: dephasing time, at most ``2 * t1``.
        time: duration of the error, in the units of ``t1`` and ``t2``.
        excited_population: equilibrium population of ``|1>``, in ``[0, 1]``.

    Example::

        from mpstab.models.noise import ThermalRelaxationError

        idle = ThermalRelaxationError(t1=100e-6, t2=80e-6, time=50e-9)
    """

    t1: float
    t2: float
    time: float
    excited_population: float = 0.0

    def __post_init__(self):
        if self.t1 <= 0 or self.t2 <= 0:
            raise ValueError("ThermalRelaxationError needs t1 > 0 and t2 > 0.")
        if self.t2 > 2 * self.t1 * (1 + 1e-12):
            raise ValueError(
                f"ThermalRelaxationError t2={self.t2} > 2 t1={2 * self.t1}: "
                "not a physical channel."
            )
        if self.time < 0:
            raise ValueError(f"ThermalRelaxationError time={self.time} < 0.")
        if not 0 <= self.excited_population <= 1:
            raise ValueError(
                f"ThermalRelaxationError excited_population="
                f"{self.excited_population} outside [0, 1]."
            )

    width = 1

    def scaled(self, factor: float) -> ThermalRelaxationError:
        return replace(self, time=factor * self.time)

    @property
    def kraus(self) -> tuple[np.ndarray, ...]:
        gamma = 1 - np.exp(-self.time / self.t1)
        pe = self.excited_population
        damping = (
            np.sqrt(1 - pe) * np.array([[1, 0], [0, np.sqrt(1 - gamma)]]),
            np.sqrt(1 - pe) * np.array([[0, np.sqrt(gamma)], [0, 0]]),
            np.sqrt(pe) * np.array([[np.sqrt(1 - gamma), 0], [0, 1]]),
            np.sqrt(pe) * np.array([[0, 0], [np.sqrt(gamma), 0]]),
        )
        # The remaining coherence decay, as a phase flip with 1 - 2 p_z equal to it.
        remaining = np.exp(-self.time / self.t2 + self.time / (2 * self.t1))
        p_z = max(0.0, (1 - remaining) / 2)
        dephasing = (np.sqrt(1 - p_z) * np.eye(2), np.sqrt(p_z) * np.diag([1, -1]))
        return tuple(
            (d @ k).astype(complex)
            for d in dephasing
            for k in damping
            if np.any(d) and np.any(k)
        )

    def _event(self, index: int, support: tuple[int, ...], nqubits: int):
        return KrausChannel(index, support, self.kraus)


# -- Errors on measurement and preparation ----------------------------------------


@dataclass(frozen=True)
class ReadoutError:
    """
    Readout bit flips: a measured ``0`` is reported as ``1`` with probability
    ``p0``, and a ``1`` as ``0`` with probability ``p1`` (``p0`` if not given).

    A Pauli expectation value is estimated from the bits read out in the
    term's eigenbasis, so every non-identity factor ``P`` of a term on a noisy
    qubit becomes ``(1 - p0 - p1) P + (p1 - p0) I``. A symmetric error only
    attenuates the term; an asymmetric one also mixes in its lower-weight
    subterms. Add it with no gate, optionally restricted to some qubits;
    errors on the same qubit compose.

    Attributes:
        p0: probability of reading a ``0`` as ``1``, in ``[0, 1]``.
        p1: probability of reading a ``1`` as ``0``, in ``[0, 1]``; ``p0`` if
            not given.

    Example::

        from mpstab.models.noise import NoiseModel, ReadoutError

        noise = NoiseModel()
        noise.add(ReadoutError(0.02))                    # every qubit
        noise.add(ReadoutError(0.01, 0.05), qubits=[0])  # 1 -> 0 is likelier
    """

    p0: float
    p1: float | None = None

    def __post_init__(self):
        if self.p1 is None:
            object.__setattr__(self, "p1", self.p0)
        for p in (self.p0, self.p1):
            if not 0 <= p <= 1:
                raise ValueError(f"ReadoutError probability {p} outside [0, 1].")

    def scaled(self, factor: float) -> ReadoutError:
        return ReadoutError(factor * self.p0, factor * self.p1)


@dataclass(frozen=True)
class StatePreparationError:
    """
    Preparation error: each qubit starts flipped by ``X`` with probability
    ``p``, so ``|0>`` is prepared as ``|1>``. Add it with no gate, optionally
    restricted to some qubits; it acts on the initial state, before any gate.

    Attributes:
        p: flip probability of each qubit, in ``[0, 1]``.

    Example::

        from mpstab.models.noise import NoiseModel, StatePreparationError

        noise = NoiseModel().add(StatePreparationError(0.01))
    """

    p: float

    def __post_init__(self):
        if not 0 <= self.p <= 1:
            raise ValueError(f"StatePreparationError p={self.p} outside [0, 1].")

    def scaled(self, factor: float) -> StatePreparationError:
        return StatePreparationError(factor * self.p)


GateError = Union[
    PauliError,
    DepolarizingError,
    CoherentError,
    KrausError,
    AmplitudeDampingError,
    ThermalRelaxationError,
]
Error = Union[GateError, ReadoutError, StatePreparationError]
_GATE_ERRORS = (
    PauliError,
    DepolarizingError,
    CoherentError,
    KrausError,
    AmplitudeDampingError,
    ThermalRelaxationError,
)
_UNANCHORED_ERRORS = (ReadoutError, StatePreparationError)


# -- Rules ----------------------------------------------------------------------


def _supports(error, qubits: tuple[int, ...]):
    """The qubit tuples ``error`` acts on after a gate on ``qubits``."""
    if error.width is None:
        return [qubits]
    if error.width == 1:
        return [(q,) for q in qubits]
    if error.width == len(qubits):
        return [qubits]
    raise ValueError(
        f"{type(error).__name__} of width {error.width} cannot act on a "
        f"{len(qubits)}-qubit gate: use single-qubit labels or labels as wide as "
        "the gate."
    )


def _single_or_pair(error, kind: str) -> int:
    """Whether ``error`` acts on single qubits (1) or on coupled pairs (2)."""
    if error.width in (None, 1):
        return 1
    if error.width == 2:
        return 2
    raise ValueError(
        f"{kind} noise acts on single qubits or coupled pairs, but "
        f"{type(error).__name__} has width {error.width}."
    )


def _gate_items(circuit: Circuit):
    """``(index, gate)`` for every gate of the queue but measurements."""
    return [
        (index, gate)
        for index, gate in enumerate(circuit.queue)
        if not isinstance(gate, gates.M)
    ]


def _moments(circuit: Circuit):
    """
    In-order layering of the queue: a gate lands one moment after the last
    gate on any of its qubits, and never before the gate queued ahead of it.
    Moments therefore never decrease along the queue, so everything in moment
    ``m`` can be placed after every gate of an earlier moment and before every
    gate of a later one.

    Returns ``(layers, history)``: ``layers[m]`` is the set of qubits busy in
    moment ``m``, and ``history[q]`` lists ``(moment, index)`` of the gates on
    qubit ``q``, in order.
    """
    depth = [0] * circuit.nqubits
    layers: list[set[int]] = []
    history: dict[int, list[tuple[int, int]]] = {q: [] for q in range(circuit.nqubits)}
    moment = 0
    for index, gate in _gate_items(circuit):
        moment = max([moment] + [depth[q] for q in gate.qubits])
        for q in gate.qubits:
            depth[q] = moment + 1
            history[q].append((moment, index))
        while len(layers) <= moment:
            layers.append(set())
        layers[moment].update(gate.qubits)
    return layers, history


def _last_before(history, q: int, moment: int) -> int:
    """Queue index of the last gate on ``q`` before ``moment``, or ``-1``."""
    previous = -1
    for m, index in history[q]:
        if m >= moment:
            break
        previous = index
    return previous


@dataclass(frozen=True)
class _Rule:
    """
    One rule of a :class:`NoiseModel`; ``kind`` is ``"gate"``, ``"crosstalk"``,
    ``"idle"``, ``"readout"`` or ``"preparation"``.
    """

    kind: str
    error: Error
    gate: tuple[type, ...] | None = None
    qubits: frozenset[int] | None = None
    coupling: tuple[tuple[int, int], ...] | None = None

    def scaled(self, factor: float) -> _Rule:
        return replace(self, error=self.error.scaled(factor))

    def events(self, circuit: Circuit) -> list[NoiseEvent]:
        n = circuit.nqubits
        if self.kind in ("gate", "crosstalk"):
            emit = self._gate_events if self.kind == "gate" else self._crosstalk_events
            return [
                event
                for index, gate in _gate_items(circuit)
                if self.gate is None or isinstance(gate, self.gate)
                for event in emit(index, tuple(gate.qubits), n)
            ]
        if self.kind == "idle":
            return self._idle_events(circuit)
        if self.kind == "preparation":
            return [
                PauliChannel(-1, (q,), {"X": self.error.p}) for q in self._qubits(n)
            ]
        return []

    def _qubits(self, nqubits: int) -> list[int]:
        qubits = range(nqubits) if self.qubits is None else sorted(self.qubits)
        for q in qubits:
            if not 0 <= q < nqubits:
                raise ValueError(
                    f"{type(self.error).__name__} on qubit {q}, outside a "
                    f"{nqubits}-qubit circuit."
                )
        return list(qubits)

    def _gate_events(self, index, qubits, nqubits):
        if self.qubits is not None:
            if self.error.width != 1:  # spans the whole gate
                if not set(qubits) <= self.qubits:
                    return []
            else:
                qubits = tuple(q for q in qubits if q in self.qubits)
        return [
            self.error._event(index, support, nqubits)
            for support in (_supports(self.error, qubits) if qubits else ())
        ]

    def _crosstalk_events(self, index, qubits, nqubits):
        active = set(qubits)
        edges = sorted(
            (u, v) if u in active else (v, u)
            for u, v in self.coupling
            if (u in active) != (v in active)
        )
        if _single_or_pair(self.error, "Crosstalk") == 1:
            supports = [(s,) for s in sorted({s for _, s in edges})]
        else:
            supports = edges
        return [self.error._event(index, support, nqubits) for support in supports]

    def _idle_events(self, circuit: Circuit):
        n = circuit.nqubits
        layers, history = _moments(circuit)
        allowed = set(self._qubits(n))
        pairs = _single_or_pair(self.error, "Idle") == 2
        if pairs and self.coupling is None:
            raise ValueError(
                f"Two-qubit idle noise ({type(self.error).__name__}) needs coupling=."
            )
        events = []
        for moment, busy in enumerate(layers):
            idle = allowed - busy
            if pairs:
                supports = [(u, v) for u, v in self.coupling if u in idle and v in idle]
            else:
                supports = [(q,) for q in sorted(idle)]
            for support in supports:
                # After the last gate on the support before this moment; the
                # in-order layering puts it before the next one on each qubit.
                anchor = max(_last_before(history, q, moment) for q in support)
                events.append(self.error._event(anchor, support, n))
        return events


def _coupling(coupling) -> tuple[tuple[int, int], ...]:
    edges = []
    for edge in coupling:
        u, v = (int(q) for q in edge)
        if u == v:
            raise ValueError(f"Coupling edge {edge} joins a qubit to itself.")
        edges.append((min(u, v), max(u, v)))
    return tuple(sorted(set(edges)))


def _gate_types(gate) -> tuple[type, ...] | None:
    if gate is None:
        return None
    return (gate,) if isinstance(gate, type) else tuple(gate)


def _qubit_set(qubits) -> frozenset[int] | None:
    if qubits is None:
        return None
    if isinstance(qubits, (int, np.integer)):
        qubits = (int(qubits),)
    return frozenset(int(q) for q in qubits)


@dataclass
class NoiseModel:
    """
    Noise attached to a circuit, built rule by rule as in qibo.

    - :meth:`add` puts an error right after each matching gate, or, for
      :class:`ReadoutError` and :class:`StatePreparationError`, on the measured
      bits and on the initial state.
    - :meth:`add_crosstalk` puts it on the spectators of each matching gate:
      the qubits coupled to the gate but not acted on by it.
    - :meth:`add_idle` puts it on the qubits left idle in each moment of the
      circuit.

    Pass the model to :meth:`~mpstab.HSMPO.noisy_expectation`; :meth:`apply`
    gives the same noisy circuit as a qibo density-matrix circuit, for exact
    reference values.

    Attributes:
        rules: the rules added so far, in order.

    Example::

        from qibo import gates
        from mpstab import HSMPO
        from mpstab.models.ansatze import KickedIsing
        from mpstab.models.noise import (
            AmplitudeDampingError,
            CoherentError,
            DepolarizingError,
            NoiseModel,
            PauliError,
        )

        chain = [(0, 1), (1, 2), (2, 3)]
        noise = NoiseModel()
        noise.add(DepolarizingError(1e-3))                     # after every gate
        noise.add(PauliError({"ZZ": 5e-4}), gates.CNOT)        # correlated, on CNOTs
        noise.add(CoherentError("ZZ", 0.02), gates.CNOT)       # residual coupling
        noise.add(PauliError({"Z": 1e-3}), gates.RX, qubits=[0, 1])
        noise.add_crosstalk(PauliError({"ZZ": 1e-3}), chain, gates.CNOT)
        noise.add_idle(AmplitudeDampingError(1e-3))            # T1 while idle

        hsmpo = HSMPO(KickedIsing(nqubits=4, n_steps=1, theta_h=0.7))
        ideal = hsmpo.expectation("ZIIZ")
        noisy = hsmpo.noisy_expectation("ZIIZ", noise, n_samples=500, seed=0)
        print(f"{ideal:+.4f} -> {noisy.mean:+.4f} +- {noisy.stderr:.4f}")
    """

    rules: list[_Rule] = field(default_factory=list)

    def add(
        self,
        error: Error,
        gate: type | Sequence[type] | None = None,
        qubits: int | Sequence[int] | None = None,
    ) -> NoiseModel:
        """
        Put ``error`` after every gate of type ``gate`` (all gates if ``None``).

        Args:
            error: the error to insert. Readout and preparation errors take no
                gate.
            gate: qibo gate class, or sequence of them; only gates of these
                types get the error. Measurements never do.
            qubits: restrict the error to these qubits. A single-qubit error
                acts only on the gate's qubits in this set; an error spanning
                the whole gate applies only to gates entirely inside it.

        Returns:
            ``self``, so calls can be chained.

        Raises:
            TypeError: if ``error`` is not an mpstab error.
            ValueError: if a :class:`ReadoutError` or a
                :class:`StatePreparationError` is given a ``gate``.
        """
        self._check(error, _GATE_ERRORS + _UNANCHORED_ERRORS)
        if isinstance(error, _UNANCHORED_ERRORS):
            if gate is not None:
                raise ValueError(
                    f"{type(error).__name__} acts on the measured bits or the "
                    "initial state, not after a gate: add it without a gate, "
                    "optionally with qubits=."
                )
            kind = "readout" if isinstance(error, ReadoutError) else "preparation"
        else:
            kind = "gate"
        self.rules.append(_Rule(kind, error, _gate_types(gate), _qubit_set(qubits)))
        return self

    def add_crosstalk(
        self,
        error: GateError,
        coupling: Iterable[tuple[int, int]],
        gate: type | Sequence[type] | None = None,
    ) -> NoiseModel:
        """
        Put ``error`` on the spectators of every gate of type ``gate``.

        A spectator is a qubit coupled to one of the gate's qubits without
        being acted on by the gate. A single-qubit error acts on each
        spectator; a two-qubit one acts on each ``(gate qubit, spectator)``
        edge, in that order, as ``ZZ`` crosstalk from a driven qubit does.

        Args:
            error: the error, on one qubit or on two.
            coupling: the coupling graph, as qubit pairs.
            gate: qibo gate class, or sequence of them; only gates of these
                types cause the crosstalk.

        Returns:
            ``self``, so calls can be chained.
        """
        self._check(error, _GATE_ERRORS)
        self.rules.append(
            _Rule("crosstalk", error, _gate_types(gate), coupling=_coupling(coupling))
        )
        return self

    def add_idle(
        self,
        error: GateError,
        qubits: int | Sequence[int] | None = None,
        coupling: Iterable[tuple[int, int]] | None = None,
    ) -> NoiseModel:
        """
        Put ``error`` on the qubits left idle in each moment of the circuit.

        Moments are an in-order layering of the gate queue: a gate lands one
        moment after the last gate on any of its qubits, and never before the
        gate queued ahead of it, so a circuit queued layer by layer gets its
        layers as moments. A single-qubit error acts on every idle qubit of
        every moment, before its first gate and after its last one included. A two-qubit error,
        such as an always-on ``ZZ`` coupling, acts on each ``coupling`` edge
        whose two qubits are both idle.

        Args:
            error: the error, on one qubit or on two.
            qubits: restrict the error to these qubits.
            coupling: the coupling graph, needed by two-qubit errors.

        Returns:
            ``self``, so calls can be chained.
        """
        self._check(error, _GATE_ERRORS)
        self.rules.append(
            _Rule(
                "idle",
                error,
                qubits=_qubit_set(qubits),
                coupling=None if coupling is None else _coupling(coupling),
            )
        )
        return self

    @staticmethod
    def _check(error, allowed):
        if not isinstance(error, allowed):
            names = ", ".join(cls.__name__ for cls in allowed)
            raise TypeError(
                f"{type(error).__name__} is not a mpstab error accepted here; "
                f"use one of {names}."
            )

    def scaled(self, factor: float) -> NoiseModel:
        """
        A copy with every error's strength multiplied by ``factor``: its
        probabilities, coherent angles, damping rates or relaxation times, for
        noise-strength sweeps such as zero-noise extrapolation.
        """
        return NoiseModel([rule.scaled(factor) for rule in self.rules])

    def events(self, circuit: Circuit) -> list[NoiseEvent]:
        """The model resolved on ``circuit``: one event per inserted error."""
        events = []
        for rule in self.rules:
            events += rule.events(circuit)
        return events

    def readout_maps(self, nqubits: int) -> tuple[np.ndarray, np.ndarray]:
        """
        The readout errors of each qubit as ``(scale, offset)``, each of shape
        ``(nqubits,)``: a measured Pauli factor ``P`` on qubit ``q`` reads as
        ``scale[q] P + offset[q] I``. Ones and zeros where there is no readout
        noise.
        """
        scale, offset = np.ones(nqubits), np.zeros(nqubits)
        for rule in self.rules:
            if rule.kind != "readout":
                continue
            c = 1 - rule.error.p0 - rule.error.p1
            a = rule.error.p1 - rule.error.p0
            for q in rule._qubits(nqubits):
                scale[q], offset[q] = c * scale[q], c * offset[q] + a
        return scale, offset

    def readout_attenuation(self, nqubits: int) -> np.ndarray:
        """
        The ``scale`` of :meth:`readout_maps`: the product of ``1 - p0 - p1``
        over the readout errors on each qubit, shape ``(nqubits,)``.
        """
        return self.readout_maps(nqubits)[0]

    def readout_observable(self, observable, nqubits: int) -> tuple[float, dict]:
        """
        ``observable`` as ``(constant, {pauli: coefficient})``, as
        :func:`~mpstab.hamiltonians.pauli_terms` returns it, with the readout
        errors folded in (:meth:`readout_maps`).

        A symmetric error only rescales the terms. An asymmetric one expands a
        term with ``w`` non-identity factors on asymmetric qubits into its
        ``2**w`` subterms, the empty one adding to the constant.
        """
        constant, terms = pauli_terms(observable, nqubits)
        scale, offset = self.readout_maps(nqubits)
        result: dict[str, float] = {}
        for pauli, coefficient in terms.items():
            support = [q for q, p in enumerate(pauli) if p != "I"]
            branching = [q for q in support if offset[q] != 0]
            fixed = coefficient * float(
                np.prod([scale[q] for q in support if offset[q] == 0])
            )
            for kept in product((True, False), repeat=len(branching)):
                value, labels = fixed, list(pauli)
                for q, keep in zip(branching, kept):
                    if keep:
                        value *= scale[q]
                    else:
                        value *= offset[q]
                        labels[q] = "I"
                label = "".join(labels)
                if set(label) == {"I"}:
                    constant += value
                else:
                    result[label] = result.get(label, 0.0) + value
        return constant, result

    def apply(self, circuit: Circuit) -> Circuit:
        """
        ``circuit`` with the noise inserted, as a qibo density-matrix circuit:
        Pauli errors as ``PauliNoiseChannel``, coherent ones as unitaries and
        the others as ``KrausChannel``.

        Readout errors act on the measured bits, not on the state, so they are
        not part of this circuit: apply them to the observable with
        :meth:`readout_observable`.
        """
        by_gate = {}
        for event in self.events(circuit):
            by_gate.setdefault(event.after_gate, []).append(event)
        noisy = Circuit(circuit.nqubits, density_matrix=True)
        for event in by_gate.get(-1, ()):
            noisy.add(_qibo_gate(event))
        for index, gate in enumerate(circuit.queue):
            noisy.add(gate)
            for event in by_gate.get(index, ()):
                noisy.add(_qibo_gate(event))
        return noisy


def _qibo_gate(event: NoiseEvent):
    if isinstance(event, PauliChannel):
        return gates.PauliNoiseChannel(event.qubits, list(event.probs.items()))
    if isinstance(event, KrausChannel):
        return gates.KrausChannel(event.qubits, list(event.kraus))
    support = [q for q, p in enumerate(event.generator) if p != "I"]
    matrix = pauli_matrix("".join(event.generator[q] for q in support))
    unitary = (
        np.cos(event.angle / 2) * np.eye(len(matrix))
        - 1j * np.sin(event.angle / 2) * matrix
    )
    return gates.Unitary(unitary, *support)
