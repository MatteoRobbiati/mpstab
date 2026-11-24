"""Training circuits for Clifford data regression."""

from __future__ import annotations

import random
import warnings

from qibo import Circuit

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


def as_ansatz(ansatz: Ansatz | Circuit) -> Ansatz:
    """``ansatz``, wrapped in a :class:`~mpstab.models.ansatze.CircuitAnsatz` if it is a circuit."""
    if isinstance(ansatz, Circuit):
        return CircuitAnsatz(qibo_circuit=ansatz)
    if not isinstance(ansatz, Ansatz):
        raise TypeError(f"Expected an Ansatz or a qibo Circuit, got {type(ansatz)}.")
    return ansatz
