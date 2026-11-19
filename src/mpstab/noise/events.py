"""Noise events: what happens, and after which gate of the circuit.

Every event is anchored to a position in ``ansatz.circuit.queue``: it acts right
after the gate at index ``after_gate``, or on the initial state when
``after_gate`` is ``-1``. Events sharing an anchor act in the order they are
listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qibo import Circuit, gates
from qibo.noise import NoiseModel

from mpstab.pauli import validate_pauli_string


@dataclass(frozen=True)
class NoiseEvent:
    """
    Base class of every noise event.

    :func:`~mpstab.noise.compile_noise` dispatches on the concrete type and
    rejects any it does not know. A new channel, such as amplitude damping
    simulated by quantum trajectories, is added as a subclass plus a branch
    there.

    Attributes:
        after_gate: index in ``ansatz.circuit.queue`` of the gate the event
            follows, or ``-1`` for the initial state.
    """

    after_gate: int


@dataclass(frozen=True)
class PauliChannel(NoiseEvent):
    """
    A stochastic Pauli channel, possibly correlated over several qubits.

    With probability ``probs[P]`` the Pauli string ``P`` acts on ``qubits``,
    and with the remaining probability nothing happens.

    Attributes:
        after_gate: index of the gate the channel follows, or ``-1`` for the
            initial state.
        qubits: the qubits the channel acts on, in the order the Pauli labels
            use.
        probs: probability of each non-identity Pauli string on ``qubits``, one
            label per qubit. They must sum to at most 1.

    Example:
        A correlated two-qubit channel after the first gate::

            from mpstab.noise.events import PauliChannel

            channel = PauliChannel(after_gate=0, qubits=(0, 1), probs={"ZZ": 0.01, "XI": 0.002})
            print(channel.no_fault_probability)
    """

    qubits: tuple[int, ...]
    probs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"PauliChannel qubits {self.qubits} repeat a qubit.")
        for label, prob in self.probs.items():
            validate_pauli_string(label, len(self.qubits))
            if set(label) == {"I"}:
                raise ValueError(
                    f"PauliChannel term {label!r} is the identity; its "
                    "probability is implied by the others."
                )
            if prob < 0:
                raise ValueError(
                    f"PauliChannel term {label!r} has probability {prob} < 0."
                )
        if self.no_fault_probability < -1e-12:
            raise ValueError(
                f"PauliChannel probabilities sum to {sum(self.probs.values())} > 1."
            )

    @property
    def no_fault_probability(self) -> float:
        """Probability that the channel does nothing."""
        return 1.0 - sum(self.probs.values())

    def full_width_terms(self, nqubits: int) -> list[tuple[str, float]]:
        """
        The channel's terms as ``(pauli, probability)``, padded to ``nqubits``.

        Args:
            nqubits: width of the circuit.

        Returns:
            One ``(pauli, probability)`` entry per term, in ``probs`` order.

        Raises:
            ValueError: if the channel acts on a qubit outside the circuit.
        """
        if max(self.qubits, default=-1) >= nqubits:
            raise ValueError(
                f"PauliChannel acts on qubits {self.qubits}, outside a "
                f"{nqubits}-qubit circuit."
            )
        terms = []
        for label, prob in self.probs.items():
            full = ["I"] * nqubits
            for qubit, pauli in zip(self.qubits, label):
                full[qubit] = pauli
            terms.append(("".join(full), prob))
        return terms


@dataclass(frozen=True)
class CoherentRotation(NoiseEvent):
    """
    A coherent error ``exp(-i angle/2 generator)``, such as ZZ crosstalk or an
    over-rotation.

    Attributes:
        after_gate: index of the gate the rotation follows, or ``-1`` for the
            initial state.
        generator: full-width Pauli string, one label per qubit of the circuit.
        angle: rotation angle.

    Example:
        ZZ crosstalk between qubits 1 and 2 of a 4-qubit circuit, after gate 3::

            from mpstab.noise.events import CoherentRotation

            crosstalk = CoherentRotation(after_gate=3, generator="IZZI", angle=0.02)
    """

    generator: str
    angle: float

    def __post_init__(self):
        validate_pauli_string(self.generator)


def from_qibo_noise_model(
    noise_model: NoiseModel, circuit: Circuit
) -> list[PauliChannel]:
    """
    The Pauli channels a qibo ``NoiseModel`` inserts into ``circuit``.

    The model is applied with qibo's own ``NoiseModel.apply``, so which gates
    get noise follows qibo's rules exactly. Only Pauli channels are supported:
    ``PauliError`` and ``DepolarizingError``. An ``n``-qubit depolarizing
    channel of strength ``lam`` becomes a Pauli channel with probability
    ``lam / 4**n`` on each of its ``4**n - 1`` non-identity Paulis.

    Args:
        noise_model: the qibo model to convert.
        circuit: the noiseless circuit, normally ``ansatz.circuit``. Event
            positions index its ``queue``.

    Returns:
        The :class:`PauliChannel` events, in circuit order.

    Raises:
        NotImplementedError: on any channel other than a Pauli or depolarizing
            one, such as amplitude damping, thermal relaxation or readout
            errors.
        ValueError: if a ``PauliError`` label does not match the width of the
            gate it is attached to, as happens when a multi-qubit
            ``PauliError`` is applied by qibo one qubit at a time.

    Example:
        Depolarizing noise after every CZ::

            import numpy as np
            from qibo import Circuit, gates
            from qibo.noise import DepolarizingError, NoiseModel
            from mpstab.noise.events import from_qibo_noise_model

            circuit = Circuit(2)
            circuit.add(gates.RY(0, theta=0.3))
            circuit.add(gates.CZ(0, 1))
            model = NoiseModel()
            model.add(DepolarizingError(0.01), gates.CZ)
            events = from_qibo_noise_model(model, circuit)
            print(events[0].after_gate, len(events[0].probs))  # 1 15
    """
    density = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        density.add(gate)
    noisy = noise_model.apply(density)

    events: list[PauliChannel] = []
    gate_index = -1
    for gate in noisy.queue:
        if not isinstance(gate, gates.Channel):
            gate_index += 1
            continue
        events.append(_pauli_channel(gate, gate_index))

    if gate_index + 1 != len(circuit.queue):
        raise ValueError(
            "The noise model changed the circuit's gates, not just added channels "
            "to it, so noise positions cannot be mapped back onto it."
        )
    return events


def _pauli_channel(channel, after_gate: int) -> PauliChannel:
    """Convert one qibo channel gate into a :class:`PauliChannel`."""
    qubits = tuple(channel.qubits)
    # DepolarizingChannel subclasses PauliNoiseChannel in qibo, so test it first.
    if isinstance(channel, gates.DepolarizingChannel):
        lam = channel.init_kwargs["lam"]
        n = len(qubits)
        labels = _all_paulis(n)[1:]  # drop the identity
        return PauliChannel(after_gate, qubits, {label: lam / 4**n for label in labels})

    if isinstance(channel, gates.PauliNoiseChannel):
        probs = dict(channel.init_kwargs)
        for label in probs:
            if len(label) != len(qubits):
                raise ValueError(
                    f"qibo applied the Pauli error {label!r} to qubits {qubits}. "
                    "qibo's PauliError spreads multi-qubit labels one qubit at a "
                    "time; use PauliChannel directly for correlated noise."
                )
        return PauliChannel(after_gate, qubits, probs)

    raise NotImplementedError(
        f"{type(channel).__name__} is not a Pauli channel. mpstab.noise supports "
        "qibo's PauliError and DepolarizingError; general Kraus channels such as "
        "amplitude damping, thermal relaxation, reset and readout errors are not "
        "supported."
    )


def _all_paulis(n: int) -> list[str]:
    """Every Pauli string on ``n`` qubits, identity first."""
    labels = [""]
    for _ in range(n):
        labels = [label + p for label in labels for p in "IXYZ"]
    return labels
