"""Helpers shared by the evolutors."""

import numpy as np

#: The Pauli each supported magic gate rotates about.
GATE_GENERATORS = {"rx": "X", "ry": "Y", "rz": "Z", "t": "Z"}

#: Rotation angles of magic gates that carry no parameter.
FIXED_ANGLES = {"t": np.pi / 4}


def gate_angle(gate) -> float:
    """The rotation angle of a magic gate, including fixed-angle ones like ``T``."""
    if gate.name in FIXED_ANGLES:
        return FIXED_ANGLES[gate.name]
    return gate.parameters[0]


def gate_generator(gate, nqubits: int) -> str:
    """
    The full-width Pauli string generating ``gate``'s rotation.

    Raises:
        ValueError: if ``gate`` is not one of the supported rotations.
    """
    if gate.name not in GATE_GENERATORS:
        raise ValueError(
            f"Gate {gate.name!r} is not a supported magic gate; mpstab handles "
            f"{sorted(GATE_GENERATORS)}."
        )
    return "".join(
        GATE_GENERATORS[gate.name] if q in gate.target_qubits else "I"
        for q in range(nqubits)
    )
