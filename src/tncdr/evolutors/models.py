"""Hybrid Stabilizer-MPO evolutor"""

from dataclasses import dataclass

from qibo import Circuit

from tncdr.evolutors.tensor_network import TensorNetwork
from tncdr.evolutors.tensor_network.w_utils import (
    _compute_all_w_tensors,
    pauli_pauli_expansion,
    basis_pauli_expansion,
    X_pauli_expansion,
    theta_pauli_expansion
)
from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer import tableaus
from tncdr.evolutors.utils import gate2generator, gate2tableau
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

    def expectation_from_partition(self, n_partitions, magic_gates_per_partition, observable):
        """Sample a partition of the given ansatz."""
        _, mag_layers, stab_layers = self.ansatz.partitionate_circuit(
            n_partitions=n_partitions,
            magic_gates_per_partition=magic_gates_per_partition,
        )
        # TODO: fix this! because this works only then npartitions is 1!
        # TODO: fix the problem of sign! It is counted as element in the list
        new_observable = self.backpropagate_pauli(observable, stab_layers[0])
        stab_layers[0].draw()
        self.mpo_from_magic_circuit(magic_circuit=mag_layers[0])
        return self.contract_mpo_on_obs(new_observable)
        

    def build_one_rotation_layer(self, gate, layer_number):
        """Construct one horizontal layer of Ws, according to a given rotational gate."""

        if gate.name not in ["rx", "ry", "rz"]:
            raise ValueError("tncdr currently supports only rotational gates.")
        
        # Add theta projection and X projection (very left and very right nodes of each row)
        self.tn.add_tensor(f"Angle{layer_number}", tensor=theta_pauli_expansion(theta=gate.parameters[0]))
        self.tn.add_tensor(id=f"X{layer_number}", tensor=X_pauli_expansion())

        # Add all the W operators
        for q in range(self.nqubits):

            # Add Pauli W only where the rotation is applied
            if q in gate.target_qubits:
                self.tn.add_tensor(f"W{layer_number}{q}", tensor=self.ws_map[gate2generator[gate.name]])
            # Add identity elsewhere
            else:
                self.tn.add_tensor(f"W{layer_number}{q}", tensor=self.ws_map["I"])

            # If it is the first layer, connect to initial state
            if layer_number == 0:
                self.tn.add_edge(f"T{q}", f"W0{q}", "v_link", (0,2))
            # Else, connect with the previous layer
            else:
                self.tn.add_edge(f"W{layer_number - 1}{q}", f"W{layer_number}{q}", "v_link", (3,2))

        # Connect the first W of the row with the Angle node
        self.tn.add_edge(f"Angle{layer_number}", f"W{layer_number}0", 'h_link', (0,1))

        # Horizontal links
        for q in range(self.nqubits - 1):
            # Adding custom name to the edges because we need to collect all of 
            # them at the end of the contraction
            self.tn.add_edge(f"W{layer_number}{q}", f"W{layer_number}{q + 1}", f"h_link{layer_number}{q}", (0,1))

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
            
        # Number of rotation layers
        self.n_rotation_layers = len(magic_circuit.queue)
            
        # construct the Network 
        for i, gate in enumerate(magic_circuit.queue):
            self.build_one_rotation_layer(gate, layer_number=i)


    def contract_rotation_layer(self, layer_number: int):
        """
        Contract one rotation layer of the MPO. The upper directions are considered 
        already contracted.
        """
        # Contract Theta with the first W
        self.tn.contract(
            f"Angle{layer_number}", f"temp{layer_number}0", "h_link", f"temp_left"
        )
        # Contract the last W with X
        self.tn.contract(
            f"temp{layer_number}{self.nqubits - 1}", f"X{layer_number}", "h_link", f"temp_right"
        )
        # Contract the vertical edges of the nodes in the middle
        for q in range(self.nqubits):
            # Isolating the three cases (Theta, X and Ws)
            if q == 0:
                up_label = "temp_left"
            elif q == self.nqubits - 1:
                up_label = "temp_right"
            else:
                up_label = f"temp{layer_number}{q}"
            # Distinguishing a general case from last layer case (involving initial state)
            if layer_number ==  0:
                down_label = f"T{q}"
                new_label = f"F{q}"
            else:
                down_label = f"W{layer_number - 1}{q}"
                new_label = f"temp{layer_number - 1}{q}"
            self.tn.draw(title="after_obs")
            # Contracting
            self.tn.contract(down_label, up_label, "v_link", new_label)


    def contract_mpo_on_obs(self, observable: str):
        """
        Append the observable to the MPO and compute the contraction of 
        the whole structure to get the expectation value. 
        
        Args:
            observable (str): expected to be a string of Paulis.
        """

        # TODO: shall we make a copy and re-use the same network structure?

        if observable[0] == "-":
            final_sign = -1.
            observable = observable[1:]
        else:
            final_sign = 1.

        # Connect to the observable
        for n, pauli in enumerate(observable):
            self.tn.add_tensor(f'O{n}', tensor=pauli_pauli_expansion(pauli))
            self.tn.add_edge(f'W{self.n_rotation_layers - 1}{n}', f'O{n}', 'v_link', (3,0))

        self.tn.draw(title="start")

        # Contract the O on the last layer of Ws
        # Replace all Ws with temp nodes
        for n in range(len(observable)):
            self.tn.contract(f"W{self.n_rotation_layers - 1}{n}", f"O{n}", "v_link", f"temp{self.n_rotation_layers - 1}{n}")

        # Now loop over the rotation layers and contract them all 
        # (in the Land of Mordor, where the Shadows lie)
        for l in reversed(range(self.n_rotation_layers)):
            self.contract_rotation_layer(layer_number=l)
            self.tn.draw(title=f"h_layer{l}")

        
        # Contract all the final h_links
        for q in range(self.nqubits - 1):
            if q == 0:
                temp_label = "F0"
            else:
                temp_label = f"temp{q - 1}"
            self.tn.contract(temp_label, f"F{q + 1}", self._retrieve_h_links(qubit=q), f"temp{q}")
        
        result = final_sign * float(self.tn.tensornet.nodes[f"temp{self.nqubits-2}"]["tensor"])
        return result

    def backpropagate_pauli(self, observable: str, stabilizer_circuit: Circuit):
        """
        Process a given Pauli string applying a `stabilizer_circuit` in 
        Heisenberg picture.
        """
        # Construct the propagator and apply the inverse of the circuit gate by gate
        propagator = Pauli(observable)
        for gate in stabilizer_circuit.invert().queue:
            if len(gate.parameters) != 0:
                params = {"angle": gate.parameters[0]}
            else:
                params = {}
            propagator.apply(getattr(tableaus, gate2tableau[gate.name])(*gate.qubits, **params))
        return propagator.__repr__()


    def _retrieve_h_links(self, qubit: int):
        """Construct list of all existing horizontal links for `qubit`."""
        return [f"h_link{n}{qubit}" for n in range(self.n_rotation_layers)]






        
        