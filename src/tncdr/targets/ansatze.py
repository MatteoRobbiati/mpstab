from typing import Optional, List
from copy import deepcopy
import random 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from qibo import Circuit, gates
from qibo.noise import NoiseModel

from tncdr.targets.circuit_utils import (
    hardware_compatible_circuit,
    replace_non_clifford_gate,
)

@dataclass
class Ansatz(ABC):
    """Abstract ansatz to generate quantum states."""
    nqubits: int
    density_matrix: bool = False

    def __post_init__(self):
        self._circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)
        self.noise_model = None
        self.noisy_circuit = None

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        if not isinstance(value, Circuit):
            raise TypeError("Expected a Circuit instance")
        self._circuit = value

    @property
    def nparams(self):
        return len(self.circuit.get_parameters())
    
    def execute(self, nshots: int = None, initial_state: Circuit = None, with_noise : bool = False):
        """Execute the circuit and return the outcome."""
        # Default empty initial circuit
        if initial_state is None:
            initial_state = Circuit(self.nqubits, density_matrix=self.density_matrix)
        
        if with_noise:
            if self.noisy_circuit is None:
                raise ValueError(
                    f"Before asking for noisy simulation, ensure the noise model is set via the `update_noise_model` method."
                )
            if len(initial_state.queue) != 0:
                initial_state.density_matrix = True
                initial_state = self.noise_model.apply(initial_state)
            result = (initial_state + self.noisy_circuit)(nshots=nshots)
        else:
            result = (initial_state + self.circuit)(nshots=nshots)
        return result
        
    def update_noise_model(self, noise_model: NoiseModel):
        """Construct an attribute which is the noisy version of circuit."""
        self.noise_model = noise_model
        self.noisy_circuit = noise_model.apply(self.circuit)

    @abstractmethod
    def partitionate_circuit(self):
        raise NotImplementedError(
            f"Partitioning strategy not implemented for Ansatz {self}."
        )

    def prepare_partitions(self, n_partitions):
        """Helper method to prepare partitions."""
        magic_layers = [Circuit(self.nqubits, density_matrix=self.density_matrix) for _ in range(n_partitions)]
        stabilizer_layers = [Circuit(self.nqubits, density_matrix=self.density_matrix) for _ in range(n_partitions)]
        partitioned_circuit = Circuit(self.nqubits, density_matrix=self.density_matrix)
        return partitioned_circuit, magic_layers, stabilizer_layers
    
    def partitionate_circuit(self, replacement_probability: float):
        """
        Partitionate the circuit replacing non-Clifford (magic) gates with a given probability.
        
        For each gate in the original circuit:
        - If the gate is non-Clifford, with probability replacement_probability it is
            replaced by a Clifford gate via replace_non_clifford_gate.
        - The processed gate (whether replaced or not) is added to the overall circuit.
        - Simultaneously, consecutive gates of the same type (Clifford or non-Clifford)
            are collected into blocks.
        
        Returns:
            partitioned_circuit (Circuit): the complete processed circuit.
            clifford_blocks (List[Circuit]): list of circuits, each a block of consecutive Clifford gates.
            non_clifford_blocks (List[Circuit]): list of circuits, each a block of consecutive non-Clifford gates.
        """
        original_circuit = self.circuit
        # Initialise partitioned circuit as an empty circuit of nqubits
        partitioned_circuit = Circuit(self.nqubits)
        stabilizer_layers = []
        magic_layers = []
        # Useful to process blocks of gates defined as Clifford gates 
        # between single magic rotations
        current_block = Circuit(self.nqubits)
        current_block_type = None

        for gate in original_circuit.queue:
            # Check whether a gate is magic and, in case, try to replace
            # Add the gate if Clifford, try to replace if not
            if not gate.clifford and not isinstance(gate, gates.M):
                r = random.random()
                if r < replacement_probability:
                    new_gate = replace_non_clifford_gate(gate, method="closest")
                else:
                    new_gate = gate
            else:
                new_gate = gate
            partitioned_circuit.add(new_gate)

            # Define the block type
            gate_type = 'clifford' if new_gate.clifford else 'non_clifford'

            # If gate is the first, sets the block type
            if current_block_type is None:
                current_block_type = gate_type
                current_block.add(new_gate)
            # Else, we have to check if gate is of the same type
            # of the current block 
            else:
                # If yes, add the gate to the block and continue
                if current_block_type == gate_type:
                    current_block.add(new_gate)
                # If not, archive the old block and start a new one 
                # of a different type
                else:
                    if current_block_type == 'clifford':
                        stabilizer_layers.append(current_block)
                    else:
                        magic_layers.append(current_block)
                    current_block = Circuit(self.nqubits)
                    current_block.add(new_gate)
                    current_block_type = gate_type

        # Append the last block
        if current_block.queue:
            if current_block_type == 'clifford':
                stabilizer_layers.append(current_block)
            else:
                magic_layers.append(current_block)

        return partitioned_circuit, magic_layers, stabilizer_layers

@dataclass
class HardwareEfficient(Ansatz):
    """Hardware Efficient ansatz."""
    nlayers: int = 1
    entangling: bool = True

    def __post_init__(self):
        super().__post_init__()
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            if self.entangling:
                self.circuit += self.entanglement_layer()
        #self.circuit.add(gates.M(*range(self.nqubits)))

    @property
    def parameters_per_layers(self):
        return int(self.nparams / self.nlayers)
    
    def parametric_layer(self, layer_index: int):
        """Return the gates composing a parametric layer."""
        # Start and end index for the layer in the circuit
        # Count as start the number of parametric gates per layer + the entanglement layer
        start_index = layer_index * self.parameters_per_layers + layer_index * self.nqubits
        end_index = start_index + self.parameters_per_layers
        return deepcopy(self.circuit.queue[start_index:end_index])
    
    def entanglement_layer(self):
        """Construct an entanglement layer compatible with the target quantum circuit."""
        ent_circuit = Circuit(self.nqubits, density_matrix=self.density_matrix)
        [ ent_circuit.add(gates.CZ(q0=q%self.nqubits, q1=(q+1)%self.nqubits)) for q in range(self.nqubits) ]
        return ent_circuit
    
    
@dataclass(kw_only=True)
class TranspiledAnsatz(Ansatz):
    """
    Any ansatz which is also transpiled into native gates of a given quantum device
    presenting a given connectivity.
    
    Args:
        original_circuit: The circuit to be transpiled.
        native_gates: Optional[List]: list of native gates of the used device.
            Default is [gates.GPI2, gates.RZ, gates.Z, gates.CZ].
        connectivity: Optional[nx.Graph]: graph representing the topology of the 
            used device. Default is None and in this case the transpilation 
            does not take into account any connectivity constraint.
    """
    original_circuit: Circuit
    native_gates: Optional[List] = field(default_factory=lambda: [gates.GPI2, gates.RZ, gates.Z, gates.CZ])
    connectivity: Optional[nx.Graph] = None
    # Override nqubits so it is not passed in __init__
    nqubits: int = field(init=False)

    def __post_init__(self):
        # Set nqubits from the provided circuit.
        self.nqubits = self.original_circuit.nqubits
        # Now call the parent's __post_init__ to initialize _circuit and other attributes.
        super().__post_init__()
        # Overwrite the circuit with the original one.
        self._circuit = hardware_compatible_circuit(self.original_circuit)

        # Freeze the GPI2 gates
        for g in self._circuit.parametrized_gates:
            if isinstance(g, gates.GPI2) and g.clifford:
                g.trainable = False

    @property
    def circuit(self):
        return self._circuit
    
    