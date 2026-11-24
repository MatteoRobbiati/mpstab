"""Circuit transpilation and Clifford-replacement helpers."""

import random
from copy import deepcopy
from typing import List, Optional

import networkx as nx
import numpy as np
from qibo import Circuit, gates
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import Passes
from qibo.transpiler.placer import Random
from qibo.transpiler.router import ShortestPaths
from qibo.transpiler.unroller import NativeGates, Unroller

from mpstab.models.topology import GraphLike, Topology

DEFAULT_NATIVE_GATES = (gates.GPI2, gates.RZ, gates.Z, gates.CZ)

# Clifford angles a non-Clifford gate may be snapped to, per gate family.
_PAULI_ROTATIONS = ("rx", "ry", "rz", "rxx", "ryy", "rzz")
_ROTATION_CLIFFORD_ANGLES = np.arange(-2, 3) * np.pi / 2.0
_GPI2_CLIFFORD_ANGLES = np.arange(4) * np.pi / 2.0


def hardware_compatible_circuit(
    circuit: Circuit,
    native_gates: Optional[List] = DEFAULT_NATIVE_GATES,
    connectivity: GraphLike | None = None,
) -> Circuit:
    """
    Unroll ``circuit`` into ``native_gates`` and, given a ``connectivity``,
    place and route it on that topology too.

    ``connectivity`` is the coupling map of the device: a ``networkx.Graph``
    or anything :meth:`.Topology.from_device` reads (a name is built on the
    qubits of ``circuit``).
    """
    unroller = Unroller(NativeGates(0).from_gatelist(list(native_gates)))
    if connectivity is None:
        return unroller(circuit)
    if not isinstance(connectivity, nx.Graph):
        connectivity = Topology.from_device(connectivity, circuit.nqubits).to_networkx()
    pipeline = Passes(
        [Preprocessing(), Random(), ShortestPaths(), unroller],
        connectivity=connectivity,
    )
    transpiled, _ = pipeline(circuit)
    return transpiled


def get_closest_angle(old_angle, candidates):
    """Return the closest angle to `old_angle` among some `candidates`."""
    differences = np.abs(
        np.arctan2(np.sin(candidates - old_angle), np.cos(candidates - old_angle))
    )
    return candidates[np.argmin(differences)]


def replace_non_clifford_gate(gate, replacement_method, candidates=None, rng=None):
    """
    Replace a non-Clifford RX/RY/RZ, RXX/RYY/RZZ or GPI2 gate with a Clifford one.

    Args:
        gate: the gate to replace.
        replacement_method: ``"closest"`` Clifford angle, or a ``"random"`` one.
        candidates: the Clifford angles to choose from. Defaults to the
            multiples of pi/2 legal for the gate.
        rng: source of the ``"random"`` draw, such as a seeded
            ``random.Random``. Defaults to the global ``random`` module.
    """
    if candidates is None:
        if gate.name in _PAULI_ROTATIONS:
            candidates = _ROTATION_CLIFFORD_ANGLES
        elif isinstance(gate, gates.GPI2):
            candidates = _GPI2_CLIFFORD_ANGLES
        else:
            raise NotImplementedError(
                f"replace_non_clifford_gate does not support gate type {gate.name}"
            )

    if replacement_method == "random":
        new_angle = (random if rng is None else rng).choice(candidates)
    elif replacement_method == "closest":
        new_angle = get_closest_angle(gate.parameters[0], candidates)
    else:
        raise ValueError(f"Unknown method {replacement_method!r}")

    new_gate = deepcopy(gate)
    new_gate.parameters = [new_angle]
    return new_gate
