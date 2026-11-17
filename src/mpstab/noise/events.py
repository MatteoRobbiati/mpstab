"""Noise events: what happens, and after which gate of the circuit.

Every event is anchored to a position in ``ansatz.circuit.queue``: it acts right
after the gate at index ``after_gate``, or on the initial state when
``after_gate`` is ``-1``. Events sharing an anchor act in the order they are
listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mpstab.pauli import validate_pauli_string


@dataclass(frozen=True)
class NoiseEvent:
    """
    Base class of every noise event.

    :func:`~mpstab.noise.compile_noise` dispatches on the concrete type and
    rejects any it does not know. A new channel, such as amplitude damping
    simulated by quantum trajectories, is added as a subclass plus a branch
    there.

    Parameters
    ----------
    after_gate : int
        Index in ``ansatz.circuit.queue`` of the gate the event follows, or
        ``-1`` for the initial state.
    """

    after_gate: int


@dataclass(frozen=True)
class PauliChannel(NoiseEvent):
    """
    A stochastic Pauli channel, possibly correlated over several qubits.

    With probability ``probs[P]`` the Pauli string ``P`` acts on ``qubits``,
    and with the remaining probability nothing happens.

    Parameters
    ----------
    after_gate : int
        Index of the gate the channel follows, or ``-1`` for the initial state.
    qubits : tuple of int
        The qubits the channel acts on, in the order the Pauli labels use.
    probs : dict of str to float
        Probability of each non-identity Pauli string on ``qubits``, one label
        per qubit. They must sum to at most 1.

    Examples
    --------
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

        Parameters
        ----------
        nqubits : int
            Width of the circuit.

        Returns
        -------
        list of (str, float)
            One entry per term, in ``probs`` order.
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

    Parameters
    ----------
    after_gate : int
        Index of the gate the rotation follows, or ``-1`` for the initial state.
    generator : str
        Full-width Pauli string, one label per qubit of the circuit.
    angle : float
        Rotation angle.

    Examples
    --------
    ZZ crosstalk between qubits 1 and 2 of a 4-qubit circuit, after gate 3::

        from mpstab.noise.events import CoherentRotation

        crosstalk = CoherentRotation(after_gate=3, generator="IZZI", angle=0.02)
    """

    generator: str
    angle: float

    def __post_init__(self):
        validate_pauli_string(self.generator)
