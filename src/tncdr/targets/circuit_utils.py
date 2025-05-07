from typing import List, Optional
from copy import deepcopy
import random

import numpy as np

import networkx as nx

from qibo import Circuit, gates

from qibo.transpiler.pipeline import Passes
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.router import ShortestPaths
from qibo.transpiler.unroller import Unroller, NativeGates
from qibo.transpiler.placer import Random

def hardware_compatible_circuit(
    circuit: Circuit,
    native_gates: Optional[List] = [gates.GPI2, gates.RZ, gates.Z, gates.CZ],
    connectivity: Optional[nx.Graph] = None,
):
    
    natives = NativeGates(0).from_gatelist(native_gates)
    unroller = Unroller(natives)

    if connectivity is None:
        circuit = unroller(circuit)
        return circuit
    else:
        custom_passes = [Preprocessing(), Random(), ShortestPaths(), unroller]
        custom_pipeline = Passes(custom_passes, connectivity=connectivity)
        transpiled_circ, final_layout = custom_pipeline(circuit)    
        return transpiled_circ
    
def get_closest_angle(old_angle, candidates):
    """Return the closest angle to `old_angle` among some `candidates`."""
    differences = np.abs(
        np.arctan2(
            np.sin(candidates - old_angle),
            np.cos(candidates - old_angle)
        )
    )
    closest_index = np.argmin(differences)
    return candidates[closest_index]

def replace_non_clifford_gate(gate, candidates=None, method="random"):
    """Replace non‐Clifford RX/RY/RZ or GPI2 gate with a Clifford one."""
    # RX/RY/RZ branch
    if gate.name in ("rx", "ry", "rz"):
        if candidates is None:
            candidates = np.arange(-2, 3) * np.pi/2.0
        old_angle = gate.parameters[0]
        if method == "random":
            new_angle = random.choice(candidates)
        elif method == "closest":
            new_angle = get_closest_angle(old_angle, candidates)
        else:
            raise ValueError(f"Unknown method {method!r}")
        new_gate = deepcopy(gate)
        new_gate.parameters = [new_angle]
        return new_gate

    # GPI2 branch
    elif isinstance(gate, gates.GPI2):
        # only multiples of π/2 are legal Clifford GPI2 angles
        if candidates is None:
            candidates = np.arange(4) * (np.pi/2.0)   # [0, π/2, π, 3π/2]
        # extract the current angle; assuming you stored it as gate.angle
        old_angle = gate.parameters[0]
        if method == "random":
            new_angle = random.choice(candidates)
        elif method == "closest":
            new_angle = get_closest_angle(old_angle, candidates)
        else:
            raise ValueError(f"Unknown method {method!r}")
        # re-instantiate a fresh GPI2 at the chosen Clifford angle
        new_gate = deepcopy(gate)
        new_gate.parameters = [new_angle]
        return new_gate

    else:
        raise NotImplementedError(
            f"replace_non_clifford_gate does not support gate type {gate.name}"
        )
