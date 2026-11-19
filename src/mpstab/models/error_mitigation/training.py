"""Training circuits for Clifford data regression, and noise amplification."""

from __future__ import annotations

import random
import warnings

from qibo import Circuit, gates

from mpstab.models.ansatze import Ansatz, CircuitAnsatz


def training_circuits(
    ansatz: Ansatz | Circuit,
    n_circuits: int,
    replacement_probability: float = 0.9,
    replacement_method: str = "closest",
    seed: int | None = None,
) -> list[Circuit]:
    """
    Low-magic variants of ``ansatz`` to train a CDR noise map on.

    Each non-Clifford gate is replaced by a Clifford one with probability
    ``replacement_probability`` and kept otherwise
    (:meth:`~mpstab.models.ansatze.Ansatz.partitionate_circuit`). Unlike the
    fully Clifford circuits of the original CDR, the gates that are kept leave
    some magic in each circuit, which HSMPO still simulates cheaply, so the
    training set stays closer to the target circuit. Every circuit keeps the
    target's gate sequence, only with other angles, so noise placed by gate
    type or position acts on it as on the target.

    Args:
        ansatz: the circuit to mitigate.
        n_circuits: training circuits to sample.
        replacement_probability: chance of replacing each non-Clifford gate.
            Lower values keep more magic: closer to the target, costlier to
            simulate.
        replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
        seed: seed of the replacement draws.

    Returns:
        The sampled circuits, as qibo ``Circuit`` objects.
    """
    ansatz = as_ansatz(ansatz)
    if replacement_probability >= 1 and replacement_method == "closest":
        warnings.warn(
            "With replacement_probability=1 and the 'closest' method every "
            "training circuit is the same circuit; lower the probability or "
            "use replacement_method='random'.",
            stacklevel=2,
        )
    rng = random.Random(seed)
    return [
        ansatz.partitionate_circuit(
            replacement_probability, replacement_method, rng=rng
        )[1]
        for _ in range(n_circuits)
    ]


def fold_circuit(circuit: Circuit, level: int) -> Circuit:
    r"""
    Amplify the noise of ``circuit`` by global unitary folding.

    Returns :math:`U (U^\dagger U)^k` with ``level = 2k + 1``: the same unitary,
    with ``level`` times as many gates, hence about ``level`` times the noise.
    Measurement gates are dropped.

    Args:
        circuit: the circuit :math:`U` to fold.
        level: the noise level, an odd positive integer. ``1`` returns a copy.

    Raises:
        ValueError: if ``level`` is not an odd positive integer.
    """
    if level < 1 or level != int(level) or int(level) % 2 == 0:
        raise ValueError(f"Folding needs an odd positive level, got {level}.")
    unitary = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        if not isinstance(gate, gates.M):
            unitary.add(gate)
    folded = unitary.copy(deep=True)
    for _ in range((int(level) - 1) // 2):
        folded += unitary.invert() + unitary.copy(deep=True)
    return folded


def as_ansatz(ansatz: Ansatz | Circuit) -> Ansatz:
    """``ansatz``, wrapped in a :class:`~mpstab.models.ansatze.CircuitAnsatz` if it is a circuit."""
    if isinstance(ansatz, Circuit):
        return CircuitAnsatz(qibo_circuit=ansatz)
    if not isinstance(ansatz, Ansatz):
        raise TypeError(f"Expected an Ansatz or a qibo Circuit, got {type(ansatz)}.")
    return ansatz
