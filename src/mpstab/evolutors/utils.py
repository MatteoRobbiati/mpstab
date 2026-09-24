"""Helpers shared by the evolutors."""

from typing import List, Tuple

import numpy as np
import stim
from qibo import Circuit

from mpstab.pauli import conjugate

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


def dressed_rotations(
    nqubits: int,
    stab_engine,
    magic_gates: List[Tuple[int, object]],
    clifford_circuit: Circuit,
) -> List[Tuple[str, float]]:
    """
    Every magic gate's dressed ``(generator, signed_angle)``, in circuit order.

    Each gate's generator is backpropagated through the Clifford sub-circuit
    preceding it, then scaled by the gate's rotation angle. The Clifford circuit
    is simulated once into a running ``stim.TableauSimulator`` and each generator
    is conjugated at its own breakpoint, rather than re-simulating the prefix per
    magic gate.

    Requires a :class:`~mpstab.engines.StimEngine`, but only when there is magic
    to dress: a Clifford-only circuit has no dressed rotations whatever engine is
    in use.

    Raises:
        NotImplementedError: if there are magic gates and ``stab_engine`` is not a
            ``StimEngine``.
    """
    if not magic_gates:
        return []

    from mpstab.engines.stabilizers.stim import StimEngine

    if not isinstance(stab_engine, StimEngine):
        raise NotImplementedError(
            "Dressed-rotation extraction requires StimEngine (this circuit has "
            f"{len(magic_gates)} magic gate(s)). Call "
            "set_engines(stab_engine=StimEngine()) to enable it."
        )

    simulator = stim.TableauSimulator()
    simulator.do(stim.Circuit(f"I {nqubits - 1}"))

    rotations: List[Tuple[str, float]] = []
    pending = iter(magic_gates)
    next_gate = next(pending, None)

    def dress_gates_at(breakpoint_index: int):
        nonlocal next_gate
        while next_gate is not None and next_gate[0] == breakpoint_index:
            gate = next_gate[1]
            # current_inverse_tableau() is U^dag for the Clifford prefix U.
            generator, sign = conjugate(
                gate_generator(gate, nqubits), simulator.current_inverse_tableau()
            )
            rotations.append((generator, gate_angle(gate) * sign))
            next_gate = next(pending, None)

    dress_gates_at(0)
    for breakpoint_index, gate in enumerate(clifford_circuit.queue, start=1):
        one_gate = Circuit(nqubits)
        one_gate.add(gate)
        simulator.do(stab_engine.to_stim(one_gate))
        dress_gates_at(breakpoint_index)

    return rotations
