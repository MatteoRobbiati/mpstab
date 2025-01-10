from abc import ABC
from dataclasses import dataclass
from copy import deepcopy
import random 

import numpy as np

from qibo import Circuit, gates

from dataclasses import dataclass
from abc import ABC

@dataclass
class Ansatz(ABC):
    """Abstract ansatz to generate quantum states."""
    nqubits: int
    density_matrix: bool = False

    def __post_init__(self):
        self._circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)

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
    
    def random_unitary(self, random_seed: int = 42):
        """Sample circuit's params in [-pi, pi], execute and return state."""
        np.random.seed(random_seed)
        self.circuit.set_parameters(np.random.uniform(-np.pi, np.pi, self.nparams))
        return self.circuit

    def random_quasi_clifford_unitary(self, cliff_fraction: float = 0.7, random_seed: int = 42):
        """
        Sample circuit's params so that a portion `0 < cliff_fraction < 1` 
        of the gates are cliffordized by setting an angle which is a multiple
        of pi/2.
        """
        np.random.seed(random_seed)
        new_parameters = []
        for _ in range(self.nparams):
            coin = np.random.uniform(0,1)
            if coin >= cliff_fraction:
                new_parameters.append(np.random.uniform(-np.pi, np.pi))
            else:
                new_parameters.append(np.random.randint(-2, 3) * np.pi / 2)
        self.circuit.set_parameters(new_parameters)
        return self.circuit
    
    def execute(self, nshots: int = None):
        """Execute the circuit and return the outcome."""
        return self.circuit(nshots=nshots)
        


@dataclass
class HardwareEfficient(Ansatz):
    """Hardware Efficient ansatz."""
    nlayers: int = 1

    def __post_init__(self):
        super().__post_init__()
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=0.))
                self.circuit.add(gates.RZ(q=q, theta=0.))
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
        with the layer of CNOTS.
        """
        target_layers = [0]
        target_layers.extend(list(random.sample(range(1, self.nlayers), n_partitions - 1)))
        target_layers = np.sort(target_layers)
        # Mapping layers into a dictionary
        layers_map = {str(value): index for index, value in enumerate(target_layers)}
        # These list will contain the circuits composing the partitionate circuit
        # Each list will be containing a sub-circuit
        magic_layers, stabilizer_layers = [Circuit(self.nqubits)] * n_partitions, [Circuit(self.nqubits)] * n_partitions

        partitioned_circuit = Circuit(self.nqubits)

        partition_block = 0
        for i in range(self.nlayers):
            if i in target_layers:
                partition_block = layers_map[str(i)]
                target_gates = list(random.sample(range(0, self.parameters_per_layers - 1), magic_gates_per_partition))
                layer_gates = self.parametric_layer(layer_index=i)

                for j, gate in layer_gates:
                    if j in target_gates:
                        partitioned_circuit.add(gate)
                        magic_layers[partition_block].add(gate)
                    else:
                        gate.parameters = np.random.randint(-2, 3) * np.pi / 2
                        partitioned_circuit.add(gate)
                        stabilizer_layers[partition_block].add(gate)
                





    def parametric_layer(self, layer_index: int):
        """Return the gates composing a parametric layer."""
        # Start and end index for the layer in the circuit
        # Count as start the number of parametric gates per layer + the entanglement layer
        start_index = layer_index * self.parameters_per_layers + layer_index * self.nqubits
        end_index = start_index + self.parameters_per_layers
        return deepcopy(self.circuit.queue[start_index:end_index])
    
    def entanglement_layer(self):
        """Construct an entanglement layer compatible with the target quantum circuit."""
        ent_circuit = Circuit(self.nqubits)
        [ ent_circuit.add(gates.CNOT(q0=q%self.nqubits, q1=(q+1)%self.nqubits)) for q in range(self.nqubits) ]
        return ent_circuit