"""Noise amplification by global unitary folding."""

from __future__ import annotations

from qibo import Circuit, gates


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
