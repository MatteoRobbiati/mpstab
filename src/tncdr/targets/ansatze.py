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
        self.circuit.add(gates.M(*range(self.nqubits)))

    
    @property
    def parameters_per_layers(self):
        return int(self.nparams / self.nlayers)
    

    def partitionate_circuit(self, n_partitions: int, magic_gates_per_partition: int = 1):
        """
        Construct a new circuit starting from the original so that it is in the form
        T3 - C3 - T2 - C2 - T1 - C1 - ... etc, 
        where C blocks are stabilizer circuits and T blocks are full of magic 🧙.
        T blocks are composed of some of the rotations which preserve their original 
        random angles, while C blocks are composed of the remaining rotations combined 
        with the layer of CZ.

        Args:
            n_partitions (int): number of partitions. For example if 2 is chosen,
                then the partitioned circuit is composed as T2 - C2 - T1 - C1;
            magic_gates_per_partition (int): since the partitions are defined 
                starting from the layered structure, we need to decide how many 
                gates in the target layers are kept as magic. The selected gates 
                in the target layer are then used to compose the magic layer T.

        Returns:
            partitioned_circuit (Circuit): a circuit of the same shape of the original one,
                but with some of the gates magic and the other Clifford, according 
                to the chosen strategy;
            magic_layers (List[Circuit]): list of circuits corresponding to each 
                magic layer T1, T2, T3, ...
            stabilizer_layers (List[Circuit]): list of circuits corresponding to 
                each stabilizer layer C1, C2, C3, ...
        """

        if n_partitions > self.nlayers:
            raise ValueError(
                f"Number of partitions (now {n_partitions}) have to be <= number of layers (which is {self.nlayers})"
            )

        # Layer 0 is selected
        target_layers = [0]
        # Select random `n_partitions - 1` layers from layers != layer 0
        target_layers.extend(list(random.sample(range(1, self.nlayers), n_partitions - 1)))
        target_layers = np.sort(target_layers)


        # Mapping layers into a dictionary
        layers_map = {str(value): index for index, value in enumerate(target_layers)}
        # These list will contain the circuits composing the partitionate circuit
        # Each list will be containing a sub-circuit
        partitioned_circuit, magic_layers, stabilizer_layers = self.prepare_partitions(n_partitions)

        partition_block = 0
        for i in range(self.nlayers):
            if i in target_layers:
                partition_block = layers_map[str(i)]
                # Sampling which rotations will remain magic
                target_gates = list(random.sample(range(0, self.parameters_per_layers - 1), magic_gates_per_partition))
                layer_gates = self.parametric_layer(layer_index=i)

                for j, gate in enumerate(layer_gates):
                    if j in target_gates:
                        partitioned_circuit.add(gate)
                        magic_layers[partition_block].add(gate)
                    else:
                        # new_gate = sample_random_pauli_gate(qubit=gate.qubits[0])
                        new_gate = gates.Y(gate.qubits[0])
                        partitioned_circuit.add(new_gate)
                        stabilizer_layers[partition_block].add(new_gate)
            else:
                layer_gates = self.parametric_layer(layer_index=i)
                for j, gate in enumerate(layer_gates):
                    # new_gate = sample_random_pauli_gate(qubit=gate.qubits[0])
                    new_gate = gates.Y(gate.qubits[0])
                    partitioned_circuit.add(new_gate)
                    stabilizer_layers[partition_block].add(new_gate)
            
            if self.entangling:
                partitioned_circuit += self.entanglement_layer()
                stabilizer_layers[partition_block] += self.entanglement_layer()
        
        partitioned_circuit.add(gates.M(*range(self.nqubits)))

        return partitioned_circuit, magic_layers, stabilizer_layers


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
        self._circuit = self.original_circuit

    @property
    def circuit(self):
        return hardware_compatible_circuit(
            circuit=self._circuit,
            native_gates=self.native_gates,
            connectivity=self.connectivity,
        )
    
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
        original_circuit = deepcopy(self.circuit)
        partitioned_circuit = Circuit(self.nqubits)
        stabilizer_layers = []
        magic_layers = []
        current_block = Circuit(self.nqubits)
        current_block_type = None

        for gate in original_circuit.queue:
            if not gate.clifford and not isinstance(gate, gates.M):
                if random.random() < replacement_probability:
                    new_gate = replace_non_clifford_gate(gate, method="closest")
                else:
                    new_gate = gate
            else:
                new_gate = gate

            partitioned_circuit.add(new_gate)
            gate_type = 'clifford' if new_gate.clifford else 'non_clifford'

            if current_block_type is None:
                current_block_type = gate_type
                current_block.add(new_gate)
            else:
                if current_block_type == gate_type:
                    current_block.add(new_gate)
                else:
                    if current_block_type == 'clifford':
                        stabilizer_layers.append(current_block)
                    else:
                        magic_layers.append(current_block)
                    current_block = Circuit(self.nqubits)
                    current_block.add(new_gate)
                    current_block_type = gate_type

        if current_block.queue:
            if current_block_type == 'clifford':
                stabilizer_layers.append(current_block)
            else:
                magic_layers.append(current_block)

        return partitioned_circuit, magic_layers[:-1], stabilizer_layers