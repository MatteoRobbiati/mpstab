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
    """Replace non Clifford gate with a Clifford gate."""

    if gate.name not in ["rx", "ry", "rz"]:
        raise NotImplementedError(
            f"This function does not support gate of type {gate.name}"
        )

    if candidates is None:
        candidates = np.arange(-2,3,1) * np.pi / 2.
    
    new_gate = deepcopy(gate)
    if gate.name in ["rx", "ry", "rz"]:
        old_angle = gate.parameters[0]
        if method == "random":
            new_gate.parameters = random.choice(candidates)
        elif method == "closest":
            new_gate.parameters = get_closest_angle(old_angle=old_angle, candidates=candidates)

    return new_gate
