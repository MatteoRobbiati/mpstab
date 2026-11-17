"""A gate-attached noise model, built with :meth:`NoiseModel.add` as in qibo."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Sequence, Union

import numpy as np
from qibo import Circuit, gates

from mpstab.noise.events import CoherentRotation, NoiseEvent, PauliChannel
from mpstab.pauli import PAULI_MATRICES, validate_pauli_string


@dataclass(frozen=True)
class PauliError:
    """
    Stochastic Pauli error: ``probs[P]`` is the probability that ``P`` acts.

    Single-qubit labels (``{"X": 0.01, "Z": 0.02}``) act independently on each
    qubit of the gate. Labels as wide as the gate (``{"ZZ": 0.01}`` after a
    two-qubit gate) act jointly on its qubits, in the gate's qubit order, which
    is how correlated noise is written.

    Examples
    --------
    ::

        from mpstab.noise import PauliError

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

    def _events(self, index: int, qubits: tuple[int, ...], nqubits: int):
        for support in _supports(self, qubits):
            yield PauliChannel(index, support, dict(self.probs))


@dataclass(frozen=True)
class DepolarizingError:
    """
    Depolarizing channel ``(1 - lam) rho + lam I / 2**k`` on the ``k`` qubits of
    the gate, with qibo's convention: probability ``lam / 4**k`` on each of the
    ``4**k - 1`` non-identity Paulis.

    For independent single-qubit depolarizing on every qubit of a multi-qubit
    gate, use ``PauliError({"X": p / 3, "Y": p / 3, "Z": p / 3})``.

    Examples
    --------
    ::

        from mpstab.noise import DepolarizingError

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

    def _events(self, index: int, qubits: tuple[int, ...], nqubits: int):
        k = len(qubits)
        labels = ("".join(p) for p in product("IXYZ", repeat=k))
        probs = {label: self.lam / 4**k for label in labels if set(label) != {"I"}}
        yield PauliChannel(index, qubits, probs)


@dataclass(frozen=True)
class CoherentError:
    """
    Coherent error ``exp(-i angle/2 P)``, such as an over-rotation or crosstalk.

    A single-qubit ``generator`` acts on each qubit of the gate; one as wide as
    the gate acts jointly on its qubits, in the gate's qubit order.

    Examples
    --------
    ::

        from mpstab.noise import CoherentError

        crosstalk = CoherentError("ZZ", 0.02)
        over_rotation = CoherentError("X", 0.01)
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

    def _events(self, index: int, qubits: tuple[int, ...], nqubits: int):
        for support in _supports(self, qubits):
            labels = ["I"] * nqubits
            for q, p in zip(support, self.generator):
                labels[q] = p
            yield CoherentRotation(index, "".join(labels), self.angle)


@dataclass(frozen=True)
class ReadoutError:
    """
    Symmetric readout bit flip: each measured bit is reported flipped with
    probability ``p``.

    A Pauli expectation value is estimated from the bits read out in the
    term's eigenbasis, so every non-identity factor of a term on a noisy
    qubit picks up a factor ``1 - 2 p``. Add it with no gate, optionally
    restricted to some qubits; errors on the same qubit compose.

    Examples
    --------
    ::

        from mpstab.noise import NoiseModel, ReadoutError

        noise = NoiseModel()
        noise.add(ReadoutError(0.02))              # every qubit
        noise.add(ReadoutError(0.05), qubits=[0])  # qubit 0 is worse
    """

    p: float

    def __post_init__(self):
        if not 0 <= self.p <= 1:
            raise ValueError(f"ReadoutError p={self.p} outside [0, 1].")

    def scaled(self, factor: float) -> ReadoutError:
        return ReadoutError(factor * self.p)


Error = Union[PauliError, DepolarizingError, CoherentError, ReadoutError]


def _supports(error, qubits: tuple[int, ...]):
    """The qubit tuples ``error`` acts on after a gate on ``qubits``."""
    if error.width == 1:
        return [(q,) for q in qubits]
    if error.width == len(qubits):
        return [qubits]
    raise ValueError(
        f"{type(error).__name__} of width {error.width} cannot act on a "
        f"{len(qubits)}-qubit gate: use single-qubit labels or labels as wide as "
        "the gate."
    )


@dataclass(frozen=True)
class _Rule:
    error: Error
    gate: tuple[type, ...] | None
    qubits: frozenset[int] | None

    def events(self, index: int, gate, nqubits: int) -> Iterable[NoiseEvent]:
        if isinstance(self.error, ReadoutError):
            return ()
        if self.gate is not None and not isinstance(gate, self.gate):
            return ()
        qubits = tuple(gate.qubits)
        if self.qubits is not None:
            if self.error.width != 1:  # spans the whole gate
                if not set(qubits) <= self.qubits:
                    return ()
            else:
                qubits = tuple(q for q in qubits if q in self.qubits)
        if not qubits:
            return ()
        return self.error._events(index, qubits, nqubits)


@dataclass
class NoiseModel:
    """
    Noise attached to gates, built rule by rule with :meth:`add` as in qibo.

    Every rule puts its error right after each matching gate. Pass the model
    to :meth:`~mpstab.HSMPO.noisy_expectation`; :meth:`apply` gives the same
    noisy circuit as a qibo density-matrix circuit, for exact reference values.

    Examples
    --------
    ::

        from qibo import gates
        from mpstab import HSMPO
        from mpstab.models.ansatze import KickedIsing
        from mpstab.noise import CoherentError, DepolarizingError, NoiseModel, PauliError

        noise = NoiseModel()
        noise.add(DepolarizingError(1e-3))                     # after every gate
        noise.add(PauliError({"ZZ": 5e-4}), gates.CNOT)        # correlated, on CNOTs
        noise.add(CoherentError("ZZ", 0.02), gates.CNOT)       # crosstalk
        noise.add(PauliError({"Z": 1e-3}), gates.RX, qubits=[0, 1])

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

        Parameters
        ----------
        error : PauliError, DepolarizingError or CoherentError
            The error to insert.
        gate : qibo gate class or sequence of them, optional
            Only gates of these types get the error. Measurements never do.
        qubits : int or sequence of int, optional
            Restrict the error to these qubits. A single-qubit error acts only
            on the gate's qubits in this set; an error spanning the whole gate
            applies only to gates entirely inside it.

        Returns
        -------
        NoiseModel
            ``self``, so calls can be chained.
        """
        if not isinstance(
            error, (PauliError, DepolarizingError, CoherentError, ReadoutError)
        ):
            raise TypeError(
                f"{type(error).__name__} is not a mpstab error; use PauliError, "
                "DepolarizingError, CoherentError or ReadoutError."
            )
        if isinstance(error, ReadoutError) and gate is not None:
            raise ValueError(
                "ReadoutError acts on the measured bits, not after a gate: add it "
                "without a gate, optionally with qubits=."
            )
        if gate is not None and isinstance(gate, type):
            gate = (gate,)
        if isinstance(qubits, (int, np.integer)):
            qubits = (int(qubits),)
        self.rules.append(
            _Rule(
                error,
                tuple(gate) if gate is not None else None,
                frozenset(qubits) if qubits is not None else None,
            )
        )
        return self

    def scaled(self, factor: float) -> NoiseModel:
        """
        A copy with every probability and coherent angle multiplied by ``factor``,
        for noise-strength sweeps such as zero-noise extrapolation.
        """
        return NoiseModel(
            [_Rule(r.error.scaled(factor), r.gate, r.qubits) for r in self.rules]
        )

    def events(self, circuit: Circuit) -> list[NoiseEvent]:
        """The model resolved on ``circuit``: one event per inserted error."""
        events = []
        for rule in self.rules:
            for index, gate in enumerate(circuit.queue):
                if isinstance(gate, gates.M):
                    continue
                events += rule.events(index, gate, circuit.nqubits)
        return events

    def readout_attenuation(self, nqubits: int) -> np.ndarray:
        """
        Factor ``prod (1 - 2 p)`` of the readout errors on each qubit, shape
        ``(nqubits,)``; ones where there is no readout noise.
        """
        factors = np.ones(nqubits)
        for rule in self.rules:
            if isinstance(rule.error, ReadoutError):
                qubits = range(nqubits) if rule.qubits is None else rule.qubits
                for q in qubits:
                    if not 0 <= q < nqubits:
                        raise ValueError(
                            f"ReadoutError on qubit {q}, outside a {nqubits}-qubit circuit."
                        )
                    factors[q] *= 1 - 2 * rule.error.p
        return factors

    def readout_observable(self, observable, nqubits: int) -> tuple[float, dict]:
        """
        ``observable`` as ``(constant, {pauli: coefficient})``, as
        :func:`~mpstab.hamiltonians.pauli_terms` returns it, with every
        coefficient multiplied by the readout attenuation of the qubits its
        term acts on. The constant is not measured, so it is left unchanged.
        """
        from mpstab.hamiltonians import pauli_terms

        constant, terms = pauli_terms(observable, nqubits)
        factors = self.readout_attenuation(nqubits)
        scaled = {
            pauli: coefficient
            * float(np.prod([f for f, p in zip(factors, pauli) if p != "I"]))
            for pauli, coefficient in terms.items()
        }
        return constant, scaled

    def apply(self, circuit: Circuit) -> Circuit:
        """
        ``circuit`` with the noise inserted, as a qibo density-matrix circuit:
        Pauli errors as ``PauliNoiseChannel``, coherent ones as unitaries.

        Readout errors act on the measured bits, not on the state, so they are
        not part of this circuit: apply them to the observable with
        :meth:`readout_observable`.
        """
        by_gate = {}
        for event in self.events(circuit):
            by_gate.setdefault(event.after_gate, []).append(event)
        noisy = Circuit(circuit.nqubits, density_matrix=True)
        for index, gate in enumerate(circuit.queue):
            noisy.add(gate)
            for event in by_gate.get(index, ()):
                noisy.add(_qibo_gate(event))
        return noisy


def _qibo_gate(event: NoiseEvent):
    if isinstance(event, PauliChannel):
        return gates.PauliNoiseChannel(event.qubits, list(event.probs.items()))
    support = [q for q, p in enumerate(event.generator) if p != "I"]
    matrix = np.array([[1.0]])
    for q in support:
        matrix = np.kron(matrix, PAULI_MATRICES[event.generator[q]])
    unitary = (
        np.cos(event.angle / 2) * np.eye(len(matrix))
        - 1j * np.sin(event.angle / 2) * matrix
    )
    return gates.Unitary(unitary, *support)
