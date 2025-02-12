"""Hybrid Stabilizer-MPO evolutor"""

from dataclasses import dataclass

from qibo import Circuit

from tncdr.evolutors.tensor_network import TensorNetwork
from tncdr.evolutors.tensor_network.w_utils import (
    _compute_all_w_tensors,
    # pauli_pauli_expansion,
    basis_pauli_expansion,
    X_pauli_expansion,
    theta_pauli_expansion
)
from tncdr.evolutors.utils import gate2generator
from tncdr.targets.ansatze import Ansatz


@dataclass
class HybridSurrogate:
    """
    Construct an hybrid stabilizer MPO surrogate of a given quantum circuit.

    Args:
        circuit (tncdr.targets.Ansatz): given quantum circuit in the form of a 
            tncdr ansatz class.
    """
    ansatz: Ansatz

    def __post_init__(self):
        # Initializing the tensor network
        self.tn = TensorNetwork()
        # Compute all the possible W tensors
        self.ws_map = _compute_all_w_tensors()
        # Add the initial state, which is |0> by default
        for q in range(self.nqubits):
            self.tn.add_tensor(f"T{q}", tensor=basis_pauli_expansion('0'))

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    def sample_partition(self):
        """Sample a partition of the given ansatz."""
        pass

    def build_one_rotation_layer(self, gate, layer_number):
        """Construct one horizontal layer of Ws, according to a given rotational gate."""
        
        # Add theta projection and X projection (very left and very right nodes of each row)
        self.tn.add_tensor(f"Theta{layer_number}", tensor=theta_pauli_expansion(gate.parameters[0]))
        self.tn.add_tensor(id=f"X{layer_number}", tensor=X_pauli_expansion())

        # Add all the W operators
        for q in range(self.nqubits):

            # Add Pauli W only where the rotation is applied
            if gate.target_qubits == q:
                    self.tn.add_tensor(f"W{layer_number}{q}", tensor=self.ws_map[gate2generator[gate.name]])
            # Add identity elsewhere
            else:
                self.tn.add_tensor(f"W{layer_number}{q}", tensor=self.ws_map["I"])

            # If it is the first layer, connect to initial state
            if layer_number == 0:
                self.tn.add_edge(f"T{q}", f"W0{q}", "v_link", (0,2))
            # Else, connect with the previous layer
            else:
                self.tn.add_edge(f"W{layer_number - 1}{q}", f"W{layer_number}{q}", "v_link", (0,2))

        # Connect the first W of the row with the Theta node
        self.tn.add_edge(f"Theta{layer_number}", f"W{layer_number}0", 'h_link', (0,1))

        # Horizontal links
        for q in range(self.nqubits - 1):
            self.tn.add_edge(f"W{layer_number}{q}", f"{layer_number}{q + 1}", "h_link", (0,1))

        # Link with X projection
        self.tn.add_edge(f"W{layer_number}{self.nqubits - 1}", f"X{layer_number}", "h_link", (0,0))
  

    def mpo_from_magic_circuit(self, magic_circuit:Circuit):
        """
        Construct an MPO layer according to: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150604
        given a specific circuit made of magic gates only.

        Args:
            magic_circuit (qibo.Circuit): a circuit which is supposed to be composed 
                of only magic gates.
        """

        # Sanity check - only magic gates are allowed here
        for gate in magic_circuit.queue:
            if gate.clifford:
                raise ValueError("Ensure all the gates of the given circuit are magic!")
            
        # construct the Network 
        for i, gate in enumerate(magic_circuit.queue):
            self.build_one_rotation_layer(gate, layer_number=i)




        
        