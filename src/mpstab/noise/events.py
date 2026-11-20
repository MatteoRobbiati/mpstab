"""Noise events: what happens, and after which gate of the circuit.

Every event is anchored to a position in ``ansatz.circuit.queue``: it acts right
after the gate at index ``after_gate``, or on the initial state when
``after_gate`` is ``-1``. Events sharing an anchor act in the order they are
listed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from mpstab.pauli import PAULI_MATRICES, validate_pauli_string


@dataclass(frozen=True)
class NoiseEvent:
    """
    Base class of every noise event.

    :func:`~mpstab.noise.compile_noise` dispatches on the concrete type and
    rejects any it does not know: a new kind of event is added as a subclass
    plus a branch there.

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


@dataclass(frozen=True, eq=False)
class KrausChannel(NoiseEvent):
    """
    A general channel ``rho -> sum_k K_k rho K_k^dag`` on a few qubits, such as
    amplitude damping or thermal relaxation.

    In the Pauli basis the channel is ``sum_{A, B} chi[A, B] A rho B``
    (:attr:`chi`). The diagonal of ``chi`` is a Pauli channel, its Pauli
    twirl; the off-diagonal entries are what makes it non-unital or
    otherwise non-Pauli, and the estimator evaluates them as transition
    amplitudes between two sign patterns of the rotation chain.

    Attributes:
        after_gate: index of the gate the channel follows, or ``-1`` for the
            initial state.
        qubits: the qubits the channel acts on. ``qubits[0]`` is the most
            significant factor of the Kraus matrices, as in qibo.
        kraus: Kraus operators, each ``2**k x 2**k`` for ``k = len(qubits)``.
            They must satisfy ``sum_k K_k^dag K_k = I``.

    Example::

        # Amplitude damping of qubit 0 after the first gate.
        import numpy as np
        from mpstab.noise.events import KrausChannel

        gamma = 0.05
        damping = KrausChannel(
            after_gate=0,
            qubits=(0,),
            kraus=(
                np.array([[1, 0], [0, np.sqrt(1 - gamma)]]),
                np.array([[0, np.sqrt(gamma)], [0, 0]]),
            ),
        )
        print(damping.chi[("I", "Z")])  # gamma / 4
    """

    qubits: tuple[int, ...]
    kraus: tuple[np.ndarray, ...]

    def __post_init__(self):
        object.__setattr__(self, "qubits", tuple(int(q) for q in self.qubits))
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"KrausChannel qubits {self.qubits} repeat a qubit.")
        dim = 2 ** len(self.qubits)
        kraus = tuple(np.asarray(k, dtype=complex) for k in self.kraus)
        if not kraus or any(k.shape != (dim, dim) for k in kraus):
            raise ValueError(
                f"KrausChannel on {len(self.qubits)} qubit(s) needs {dim}x{dim} "
                "Kraus operators."
            )
        completeness = sum(k.conj().T @ k for k in kraus)
        if not np.allclose(completeness, np.eye(dim), atol=1e-10):
            raise ValueError(
                "KrausChannel operators are not trace preserving: "
                "sum_k K_k^dag K_k != I."
            )
        object.__setattr__(self, "kraus", kraus)

    @property
    def chi(self) -> dict[tuple[str, str], complex]:
        """
        The process matrix in the Pauli basis, ``{(A, B): chi[A, B]}``, with
        ``A`` and ``B`` labels on :attr:`qubits`; entries below ``1e-15`` in
        magnitude are left out. It is Hermitian, and its diagonal sums to 1.
        """
        k = len(self.qubits)
        labels = ["".join(p) for p in product("IXYZ", repeat=k)]
        matrices = []
        for label in labels:
            matrix = np.array([[1.0]], dtype=complex)
            for p in label:
                matrix = np.kron(matrix, PAULI_MATRICES[p])
            matrices.append(matrix)
        # K = sum_A c[A] A with c[A] = Tr(A K) / 2**k, as Paulis are Hermitian.
        coefficients = np.array(
            [[np.trace(m @ op) / 2**k for m in matrices] for op in self.kraus]
        )
        chi = coefficients.T @ coefficients.conj()
        return {
            (a, b): complex(chi[i, j])
            for i, a in enumerate(labels)
            for j, b in enumerate(labels)
            if abs(chi[i, j]) > 1e-15
        }
