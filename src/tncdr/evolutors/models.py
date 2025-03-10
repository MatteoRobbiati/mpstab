"""Hybrid Stabilizer-MPO evolutor"""

from dataclasses import dataclass
from copy import deepcopy

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
        self._init_tn()
    
    def _init_tn(self):

        #TODO Add first layer if available
        for q in range(self.nqubits):
            self.tn.add_tensor(f"T{q}", tensor=basis_pauli_expansion('0'))
            
            #Add dummy nodes D to track the free edges mu
            self.tn.add_tensor(f'D{q}', tensor=basis_pauli_expansion('0'))
            self.tn.add_edge(f"T{q}", f"D{q}", f"mu{q}", (0,0))

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    def expectation_from_partition(
            self, 
            n_partitions, 
            magic_gates_per_partition, 
            observable,
            return_partitions=False,
        ):
        """Sample a partition of the given ansatz."""

        # Partitionate circuit
        full_circuit, mag_layers, stab_layers = self.ansatz.partitionate_circuit(
            n_partitions=n_partitions,
            magic_gates_per_partition=magic_gates_per_partition,
        )
        
        # Compute tn of the ansatz
        self._build_tn_from_partition(mag_layers, stab_layers)
        # Compute the conjugate of the observable
        phase, new_observable = self.backpropagate_pauli(observable, sum(stab_layers, start=Circuit(self.nqubits)))
        # Collect partitions into a dictionary in case we want to return it
        if return_partitions:
            partitions = {
                "magic_layers": mag_layers,
                "stabilizer_layers": stab_layers,
                "full_circuit": full_circuit,
            }
        else:
            partitions = None

        return phase*self.contract_tn_on_obs(new_observable), partitions
        
    def _build_tn_from_partition(self, mag_layers, stab_layers):
        """
        Construct an MPO layer according to: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150604
        and contract it from left to right.
        """

        # check the stab layers, might be wrong
        for i, magic_circuit in enumerate(mag_layers):
            for gate in magic_circuit.queue:

                phase, generator = self._conjugate_generator(gate, stab_layers[:i])
                self._build_rotation_layer(-phase*gate.parameters[0], generator)
                self._contract_rotation_layer()

    def _conjugate_generator(self, gate, stabs):
        """Conjugate a given gate generator by a sequence of Clifford circuits."""
        
        if gate.name not in ["rx", "ry", "rz"]:
            raise ValueError("tncdr currently supports only rotational gates.")
        
        generator = ''.join([gate2generator[gate.name] if q in gate.target_qubits else 'I' for q in range(self.nqubits)])
        return self.backpropagate_pauli(generator, sum(stabs, start=Circuit(self.nqubits)))

    def _build_rotation_layer(self, angle, generator):
        """Construct one horizontal layer of Ws, according to a given generator."""

        # Add theta projection and X projection (very left and very right nodes of each row)
        self.tn.add_tensor(f"Angle", tensor=theta_pauli_expansion(theta=angle))
        self.tn.add_tensor(id=f"X", tensor=X_pauli_expansion())

        # Generate the rotation tensors
        for q in range(self.nqubits):
            self.tn.add_tensor(f"W{q}", tensor=self.ws_map[generator[q]])

        # Add Angle and X edges
        self.tn.add_edge(f"Angle", f"W0", "w_link", (0,1))
        self.tn.add_edge(f"W{self.nqubits-1}", f"X", "w_link", (0,0))

        # Add horizontal edges
        for q in range(self.nqubits-1):
            self.tn.add_edge(f"W{q}", f"W{q+1}", "w_link", (0,1))
        
        # Add vertical edges and shift mu
        for q in range(self.nqubits):
            mu_direction = self.tn.tensornet.edges[f"T{q}", f"D{q}", f"mu{q}"]["directions"][0]
            self.tn.remove_edge(f"T{q}", f"D{q}", f"mu{q}")
            self.tn.add_edge(f"T{q}", f"W{q}", "v_link", (mu_direction,2))
            self.tn.add_edge(f"W{q}", f"D{q}", f"mu{q}", (3,0))

    def _contract_rotation_layer(self, bond_dimension=None):

        # Contract Angle and X edges
        self.tn.contract("Angle", "W0", "w_link", f"W0")
        self.tn.contract(f"W{self.nqubits-1}","X","w_link",f"W{self.nqubits-1}")
        # Contract and SVD all the rest
        self.tn.contract("T0", "W0", "v_link", f"T0")

        for q in range(1,self.nqubits):
            
            horizontal_links, left_links, right_links = self._links(q)

            self.tn.contract(f"T{q}", f"W{q}", "v_link", f"T{q}")
            self.tn.contract(f"T{q-1}", f"T{q}", horizontal_links, f"tmp")
            self.tn.svd_decomposition(
                node='tmp',
                left_node_id=f'T{q-1}',
                left_node_edges=left_links,
                right_node_id=f'T{q}',
                right_node_edges=right_links,
                middle_edge_left=f'chi{q}',
                middle_edge_right='tmp_link',
                max_bond_dimension=bond_dimension,
            )
            self.tn.contract(f"T{q}", "Lambda" ,'tmp_link', f"T{q}")

    def _links(self, q):

        horizontal_links = ['w_link']
        left_links = [f'mu{q-1}']
        right_links = [f'mu{q}']

        if self.tn.tensornet.number_of_edges(f'T{q-1}',f'T{q}'):
            horizontal_links.append(f'chi{q}')

        if q > 1 and self.tn.tensornet.number_of_edges(f'T{q-2}',f'T{q-1}'):
            left_links.append(f'chi{q-1}')

        if q < self.nqubits-1: 
            right_links.append(f'w_link')
            if self.tn.tensornet.number_of_edges(f'T{q}',f'T{q+1}'):
                right_links.append(f'chi{q+1}')
            
        return horizontal_links, left_links, right_links

    def contract_tn_on_obs(self, observable: str):
        """
        Append the observable to the MPO and compute the contraction of 
        the whole structure to get the expectation value. 
        
        Args:
            observable (str): expected to be a string of Paulis.
        """

        tn = deepcopy(self.tn)

        # Connect to the observable and remove dummy tensor
        for n, pauli in enumerate(observable):
            tn.add_tensor(f'O{n}', tensor=pauli_pauli_expansion(pauli))
            
            mu_direction = tn.tensornet.edges[f"T{n}", f"D{n}", f"mu{n}"]["directions"][0]
            tn.remove_edge(f"T{n}", f"D{n}", f"mu{n}")
            tn.add_edge(f'T{n}', f'O{n}', 'v_link', (mu_direction,0))
            tn.tensornet.remove_node(f"D{n}")

        # Contract the last layer
        for n in range(len(observable)-1):
            tn.contract(f"T{n}", f"O{n}", "v_link", f"T{n}")
            tn.contract(f"T{n}", f"T{n+1}", f"chi{n+1}", f"T{n+1}")

        tn.contract(f"T{len(observable)-1}", f"O{len(observable)-1}", "v_link", f"F")
        return float(tn.tensornet.nodes["F"]["tensor"])

    def backpropagate_pauli(self, observable: str, stabilizer_circuit: Circuit):
        """
        Process a given Pauli string applying a `stabilizer_circuit` in 
        Heisenberg picture.
        """
        # Construct the propagator and apply the inverse of the circuit gate by gate
        propagator = Pauli(observable)
        for gate in stabilizer_circuit.invert().queue:
            propagator.apply(getattr(tableaus, gate2tableau[gate.name])(*gate.qubits))

        return propagator.complex_phase(), propagator.to_string(ignore_phase=True)
