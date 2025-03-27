from copy import deepcopy
from typing import Optional, Union

import numpy as np
import networkx as nx

from tncdr.evolutors.tensor_network import TensorNetwork
from tncdr.evolutors.tensor_network.s_utils import basis

from mpo import MPO
from mpo import CNOT, CNOT_inv, CZ, SWAP, H, S, X, Y, Z, T

class CircuitMPS(TensorNetwork):
    # Basic features still missing:
    # - Add docstrings
     
    def __init__(self, n:int, initial_state:Optional[str]=None, max_bond_dimension:Optional[int]=None):
        
        self.n_qubits = n
        self.max_bond_dimension = max_bond_dimension

        if initial_state is None: initial_state = n*'0'
        assert n >=2, 'This implementation only supports 2-qubit or more MPSs'
        assert n == len(initial_state), f'Intial state qubits ({len(initial_state)}) and circuit qubits ({n}) must match.'

        super().__init__()

        # Add qubits (3-legged tensors)
        self.add_tensor('T0', tensor = np.reshape(basis(initial_state[0]), (2,1)))
        self.add_measurement('D0')
        self.add_edge('T0', 'D0', 'phyisical0', (0,0))

        for q, s in enumerate(initial_state[1:], 1):
            self.add_tensor(f'T{q}', tensor = np.reshape(basis(s), (2,1,1)))
            self.add_tensor(f'L{q-1}', tensor = np.reshape(np.array([1]), (1,1)))
            self.add_measurement(f'D{q}')

            self.add_edge(f'T{q}', f'D{q}', f'phyisical{q}', (0,0))
            self.add_edge(f'T{q}', f'L{q-1}', f'chi{q-1}_r', (2,0))
            self.add_edge(f'T{q-1}', f'L{q-1}', f'chi{q-1}_l', (1,1))
        
        # Contract the trivial unused leg
        self.add_tensor('tmp', tensor=np.array([1]))
        self.add_edge(f'T{n-1}','tmp','link', (1,0))
        self.contract(f'T{n-1}','tmp','link', f'T{n-1}')

    def bipartite_entanglement_entropy(self, cut:int):

        assert cut < self.n_qubits-1, 'Both partitions must be non-empty'
        d = np.diagonal(self.tensornet.nodes[f'L{cut}']['tensor'])**2
        return np.sum(-d*np.log(d))
    
    def amplitude(self, basis_element):
        
        assert len(basis_element) == self.n_qubits, 'Basis elements have the wrong number of qubits'
        
        mps = deepcopy(self)
        for q, state in enumerate(basis_element):
            mps.add_tensor('measurement', tensor=basis(state))
            mps._link_to_dummy(f'D{q}', 'measurement', 0)
            mps.contract(f'T{q}', 'measurement', 'v_link',f'T{q}')

        mps.contract('T0', 'L0', 'chi0_l', 'F')
        for q,q_next in zip(range(mps.n_qubits), range(1,mps.n_qubits-1)):
            mps.contract(f'T{q_next}', 'F', f'chi{q}_r', 'F')
            mps.contract('F', f'L{q_next}', f'chi{q_next}_l', 'F')

        mps.contract(f'T{self.n_qubits-1}', 'F', f'chi{self.n_qubits-2}_r', 'F')
        return mps.tensornet.nodes['F']['tensor'].item()

    def cnot(self, control, target):
        gate = CNOT if control < target else CNOT_inv
        return self.apply(gate, sorted([control, target]))
    
    def cz(self, control, target):
        return self.apply(CZ, sorted([control, target]))
    
    def swap(self, control, target):
        return self.apply(SWAP, sorted([control, target]))

    def h(self, qubit):
        self.apply(H, [qubit])
    
    def x(self, qubit):
        self.apply(X, [qubit])
     
    def y(self, qubit):
        self.apply(Y, [qubit])
    
    def z(self, qubit):
        self.apply(Z, [qubit])
    
    def s(self, qubit):
        self.apply(S, [qubit])
    
    def t(self, qubit):
        self.apply(T, [qubit])
    
    def expval(self, obs:MPO, sites:Optional[list[int]]=None):

        if sites is None: 
            sites = [i for i in range(self.n_qubits)]
        else:
            for s, s_next in zip(sites, sites[1:]):
                assert s_next-s == 1, f'All qubits in the MPO must be adjacent and ascending order. Given link {s}->{s_next}.'
        
        # "Ket" MPS
        tn = deepcopy(self)

        # Cut unnecessary sites, and contract the remaning central nodes
        if sites[0] > 0:
            tn.remove_edge(f"T{sites[0]-1}", f"L{sites[0]-1}", f'chi{sites[0]-1}_l')
            tn.contract(f"T{sites[0]}", f"L{sites[0]-1}", f'chi{sites[0]-1}_r', f"T{sites[0]}")

        if sites[-1] < self.n_qubits-1:
            tn.remove_edge(f"T{sites[-1]+1}", f"L{sites[-1]}", f'chi{sites[-1]}_r')
            tn.contract(f"T{sites[-1]}", f"L{sites[-1]}", f'chi{sites[-1]}_l', f"T{sites[-1]}")

        # Contract middle node
        for s in sites[1:]:
            tn.contract(f"T{s}", f"L{s-1}", f"chi{s-1}_r", f"T{s}")        

        # "Bra" MPS
        bra = deepcopy(tn)
        bra.complex_conjugate()

        # Whole TensorNetework
        tn.tensornet = nx.union(tn.tensornet, obs.tensornet)
        tn.tensornet = nx.union(tn.tensornet, bra.tensornet)
        
        # Connect to the observable and remove dummy tensor
        for i, s in enumerate(sites):
            tn._link_to_dummy(f'D{s}', obs.prefix+f'{i}', obs.physical_directions[i][0])
            tn._link_to_dummy(f'D{s}_dg', obs.prefix+f'{i}', obs.physical_directions[i][1])
        
        # If necessary, reattach the extremal nodes
        if sites[0] > 0:
            free_d=tn.tensornet.nodes[f'T{sites[0]}']['free_directions'].index(True)
            tn.add_edge(f"T{sites[0]}", f"T{sites[0]}_dg", f'left_link', (free_d, free_d))

        if sites[-1] < self.n_qubits-1:
            free_d=tn.tensornet.nodes[f'T{sites[-1]}']['free_directions'].index(True)
            tn.add_edge(f"T{sites[-1]}", f"T{sites[-1]}_dg", f'right_link', (free_d, free_d))

        # Contract the first layer
        tn.contract(f"T{sites[0]}", obs.prefix+f"0", "v_link", "F")
        tn.contract(f"T{sites[0]}_dg", "F", "v_link", "F")
        
        if sites[0] > 0:
            tn.contract(f"F", "F", "left_link", "F")

        for i, s in enumerate(sites[1:], start=1):
            tn.contract(f"T{s}", obs.prefix+f"{i}", "v_link", "tmp")
            tn.contract("F", f"tmp", [f"chi{s-1}_l", obs.link], "F")
            tn.contract("F", f"T{s}_dg", f"chi{s-1}_l", "F")
            tn.contract("F", "F", "v_link", "F")
        
        if sites[-1] < self.n_qubits-1:
            tn.contract(f"F", "F", "right_link", "F")

        res = tn.tensornet.nodes["F"]["tensor"].item()
        return np.real(res) if np.imag(res) < 1e-10 else res

    def apply(self, mpo:MPO, sites:Optional[list[int]]=None):

        if sites is None: 
            sites = [i for i in range(self.n_qubits)]
        else:
            for s, s_next in zip(sites, sites[1:]):
                assert s_next-s == 1, f'All qubits in the MPO must be adjacent and ascending order. Given link {s}->{s_next}.'
        
        # Link the MPS and MPO along the physical direction
        self.tensornet = nx.union(self.tensornet, mpo.tensornet)
        for i, s in enumerate(sites):
            self._move_dummy(f'D{s}', mpo.prefix+f'{i}', mpo.physical_directions[i], out_edge_id=f'physical{s}')
        
        # If only one site is affected, apply simple matrix multiplication
        if len(sites) == 1:
            return self.contract(f'T{sites[0]}', mpo.prefix+f'0', 'v_link', f'T{sites[0]}')

        # Otherwise, perform pairwise contraction
        # -- Step 1: Save the Schmidt coefficients of the right extreme, if applicable
        if sites[-1] < self.n_qubits-1:
            L_last = self.tensornet.nodes[f'L{sites[-1]}']['tensor']

        # -- Step 2: Prepare the leftmost node to be contracted
        self.contract(f'T{sites[0]}', f'L{sites[0]}', f'chi{sites[0]}_l', f'T{sites[0]}')
        self.contract(f'T{sites[0]}', mpo.prefix+f'0', 'v_link', f'T{sites[0]}')

        # -- Step 3: Sequentially contract the MPO
        for i, (s, s_next) in enumerate(zip(sites, sites[1:]), start=1):

            # -- Step 3.1: Contract the nodes involving two consecutive sites
            if s > 0: 
                L = self.tensornet.nodes[f'L{s-1}']['tensor']
                self.contract(f'T{s}',f'L{s-1}', f'chi{s-1}_r', f'T{s}')

            self.contract(f'T{s_next}', f'T{s}', f'chi{s}_r', f'T{s}')
            self.contract(f'T{s}', mpo.prefix+f'{i}', [mpo.link, 'v_link'], f'T{s}')

            if s_next < self.n_qubits-1:
                self.contract(f'T{s}', f'L{s_next}', f'chi{s_next}_l', f'T{s}')
            
            # -- Step 3.2: SVD
            self.svd_decomposition(
                node=f'T{s}',
                left_node_id=f'T{s}',
                left_node_edges=[f'physical{s}'] + ([f'chi{s-1}_l'] if s > 0 else []),
                right_node_id=f'T{s_next}',
                right_node_edges= ([f'chi{s_next}_r'] if s_next < self.n_qubits-1 else []) + ([mpo.link] if s_next < sites[-1] else []) + [f'physical{s_next}'],
                middle_node_id=f'L{s}',
                middle_edge_left=f'chi{s}_l',
                middle_edge_right=f'chi{s}_r',
                max_bond_dimension=self.max_bond_dimension,
            )

            # -- Step 3.3: Insert the Schmidt coefficients back, to restore the normal form for the MPS
            if s > 0:
                self._insert_square_matrix(f'T{s-1}', f'T{s}', f'chi{s-1}_l', L, f'L{s-1}', right_edge_name=f'chi{s-1}_r')
                T = self.tensornet.nodes[f'T{s}']['tensor']
                self.tensornet.nodes[f'T{s}']['tensor'] = np.linalg.solve(L, T)

        # -- Step 4: Complete restoring the normal form of the MPS
        if sites[-1] < self.n_qubits-1:
            self._insert_square_matrix(f'T{sites[-1]+1}', f'T{sites[-1]}', f'chi{sites[-1]}_r', L_last, f'L{sites[-1]}', right_edge_name=f'chi{sites[-1]}_l')
            T = self.tensornet.nodes[f'T{sites[-1]}']['tensor']
            self.tensornet.nodes[f'T{sites[-1]}']['tensor'] = np.linalg.solve(L_last, T)

    """
    def apply(self, gate:MPSGateTensor, qubits:Union[int,list[int]]):
        
        if type(qubits) is int: qubit = qubits
        else: qubit = qubits[0]

        if gate.n == 1: return self._apply_single_site_tensor(gate, qubit)
        if gate.n == 2: return self._apply_two_site_tensor(gate, qubits)

        raise NotImplementedError
    
    def _apply_two_site_tensor(self, gate:MPSGateTensor, sites:int):
        
        assert np.abs(sites[0]-sites[1]) == 1, f'Only adjacent qubits are supported, given {sites}.'
        self.add_tensor('gate', tensor=gate.tensor)

        for site, din, dout in zip(sites, gate.in_directions, gate.out_directions):
            self._move_dummy(f'D{site}', 'gate', (din, dout), out_edge_id=f'physical{site}')

        if sites[0] > sites[1]: sites = [s for s in reversed(sites)]     

        if sites[0] > 0:
            Lleft = self.tensornet.nodes[f'L{sites[0]-1}']['tensor']
            self.contract(f'T{sites[0]}', f'L{sites[0]-1}', f'chi{sites[0]-1}_r', f'T{sites[0]}')

        if sites[1] < self.n_qubits-1:
            Lright = self.tensornet.nodes[f'L{sites[1]}']['tensor']
            self.contract(f'T{sites[1]}', f'L{sites[1]}', f'chi{sites[1]}_l', f'T{sites[1]}')

        self.contract(f'T{sites[0]}', f'L{sites[0]}', f'chi{sites[0]}_l', f'T{sites[0]}')
        self.contract(f'T{sites[0]}', 'gate', 'v_link', 'tmp')
        self.contract(f'T{sites[1]}', 'tmp', ['v_link', f'chi{sites[0]}_r'], 'tmp')

        self.svd_decomposition(
            node='tmp',
            left_node_id=f'T{sites[0]}',
            left_node_edges=[f'physical{sites[0]}'] + ([f'chi{sites[0]-1}_l'] if sites[0] > 0 else []),
            right_node_id=f'T{sites[1]}',
            right_node_edges=[f'physical{sites[1]}'] + ([f'chi{sites[1]}_r'] if sites[1] < self.n_qubits-1 else []),
            middle_node_id=f'L{sites[0]}',
            middle_edge_left=f'chi{sites[0]}_l',
            middle_edge_right=f'chi{sites[0]}_r',
            max_bond_dimension=self.max_bond_dimension,
        )

        if sites[0] > 0:
            self._add_gauge_matrix(f'T{sites[0]-1}', f'T{sites[0]}', f'chi{sites[0]-1}_l', Lleft, f'L{sites[0]-1}', f'chi{sites[0]-1}_r')
            self.contract(f'T{sites[0]}', f'L{sites[0]-1}^-1', f'chi{sites[0]-1}_l', f'T{sites[0]}')

        if sites[1] < self.n_qubits-1:
            self._add_gauge_matrix(f'T{sites[1]+1}', f'T{sites[1]}', f'chi{sites[1]}_r', Lright, f'L{sites[1]}', f'chi{sites[1]}_l')
            self.contract(f'T{sites[1]}', f'L{sites[1]}^-1', f'chi{sites[1]}_r', f'T{sites[1]}')

    def _apply_single_site_tensor(self, gate:MPSGateTensor, site:int):
        
        self.add_tensor('gate', tensor=gate.tensor)
        self._move_dummy(f'D{site}', 'gate', (gate.in_directions[0], gate.out_directions[0]), out_edge_id=f'physical{site}')
        self.contract(f'T{site}', f'gate', 'v_link', f'T{site}')
        
    def _add_gauge_matrix(
            self, 
            left_node:str, 
            right_node:str, 
            edge:str, 
            gauge_matrix:np.ndarray,
            gauge_matrix_name:str,
            middle_edge_name:str, 
        ):
        
        dleft, dright = self.tensornet.edges[left_node, right_node, edge]['directions']
        self.add_tensor(gauge_matrix_name, tensor=gauge_matrix)
        self.add_tensor(gauge_matrix_name+'^-1', tensor=np.linalg.inv(gauge_matrix))
        self.add_edge(gauge_matrix_name+'^-1', gauge_matrix_name, middle_edge_name, (1,0))
        self.remove_edge(left_node, right_node, edge)
        self.add_edge(left_node, gauge_matrix_name, edge, (dleft, 1))
        self.add_edge(right_node, gauge_matrix_name+'^-1', edge, (dright,0))
    """

    def _insert_square_matrix(
            self, 
            left_node:str, 
            right_node:str, 
            edge:str, 
            matrix:np.ndarray,
            matrix_name:str,
            right_edge_name:Optional[str]=None,
            left_edge_name:Optional[str]=None,
        ):

        dleft, dright = self.tensornet.edges[left_node, right_node, edge]['directions']
        self.add_tensor(matrix_name, tensor=matrix)
        self.remove_edge(left_node, right_node, edge)
        self.add_edge(left_node, matrix_name, left_edge_name if left_edge_name is not None else edge, (dleft, 1))
        self.add_edge(right_node, matrix_name, right_edge_name if right_edge_name is not None else edge, (dright,0))

    def _move_dummy(self, dummy:str, tensor:str, tensor_directions:tuple[int, int], in_edge_id:str='v_link', out_edge_id:str='physical'):

        self._link_to_dummy(dummy, tensor, tensor_directions[0], in_edge_id)
        self.add_edge(tensor, dummy, out_edge_id, (tensor_directions[1],0))
    
    def _link_to_dummy(self, dummy:str, tensor:str, tensor_direction:int, edge_id:str='v_link'):

        T,d,e,data = list(self.tensornet.in_edges(dummy, data=True, keys=True))[0]
        dummy_direction = data["directions"][0]
        self.remove_edge(T,d,e)
        self.add_edge(T, tensor, edge_id, (dummy_direction,tensor_direction))
