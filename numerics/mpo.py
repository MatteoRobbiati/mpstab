from copy import deepcopy
from typing import Optional, Union

import numpy as np
import networkx as nx

from tncdr.evolutors.tensor_network import TensorNetwork
from tncdr.evolutors.tensor_network.tn_utils import paulis
from tncdr.evolutors.tensor_network.s_utils import _compute_all_s_tensors, theta_state, X_state

from tncdr.evolutors.stabilizer.pauli_string import Pauli

class MPO(TensorNetwork):

    def __init__(
            self, 
            tensors:list[np.ndarray], 
            link_directions:Optional[list[tuple[int, int]]]=None,
            physical_directions:Optional[list[tuple[int, int]]]=None,
            tensor_prefix:str='O', 
            link_name:str='h_link',
        ):

        n_tensors = len(tensors)
 
        if link_directions is None: 
            link_directions = [(2,3) if i < n_tensors-2 else (2,2) for i in range(n_tensors-1)] if n_tensors > 1 else []
        
        if physical_directions is None: 
            physical_directions = [(0,1) for i in range(n_tensors)]
        
        assert n_tensors == len(physical_directions), f'Mismatch in the number of tensors and physical legs, {n_tensors}!={len(physical_directions)}.'
        assert n_tensors == len(link_directions) + 1, f'Mismatch in the number of tensors ({n_tensors}) and link directions ({len(link_directions)}, should be {n_tensors-1}).' 

        self.physical_directions = physical_directions
        self.prefix = tensor_prefix
        self.link = link_name

        super().__init__()

        self.add_tensor(self.prefix+'0', tensor=tensors[0])
        for q, (t, link_dir) in enumerate(zip(tensors[1:], link_directions), start=1):
            self.add_tensor(self.prefix+f'{q}', tensor=t)
            self.add_edge(self.prefix+f'{q-1}', self.prefix+f'{q}', link_name, link_dir)

#OBSERVABLES
class PauliMPO(MPO):

    def __init__(self, pauli_string:Union[Pauli, str]):

        if type(pauli_string) is str: 
            pauli_string = Pauli(pauli_string)
        
        phase = pauli_string.complex_phase()
        desc = pauli_string.to_string(ignore_phase=True)

        tensors = [phase*np.reshape(paulis[desc[0]], (2,2,1))]
        for d in desc[1:-1]:
            tensors.append(np.reshape(paulis[d], (2,2,1,1)))
        
        if len(desc) > 1:
            tensors.append(np.reshape(paulis[desc[-1]], (2,2,1)))
        else:
            tensors[0] = np.squeeze(tensors[0])

        return super().__init__(tensors)

# GATES
s_tensors = _compute_all_s_tensors()

class PauliExp(MPO):

    def __init__(self, pauli_string:Union[Pauli, str], theta:float):
    
        if type(pauli_string) is str: 
            pauli_string = Pauli(pauli_string)
        
        phase = pauli_string.complex_phase()
        desc = pauli_string.to_string(ignore_phase=True)

        tensors = [s_tensors[desc[0]]]
        for d in desc[1:-1]:
            tensors.append(s_tensors[d])
        
        if len(tensors) > 1:
            tensors.append(s_tensors[desc[-1]])
    
        super().__init__(tensors, tensor_prefix=f'exp(-i{theta/2:.2f}{desc})')

        self.add_tensor('Angle', tensor=theta_state(phase*theta))
        self.add_edge('Angle', self.prefix+'0', 'tmp', (0,3))
        self.add_tensor('X', tensor=X_state())
        self.add_edge('X', self.prefix+f'{len(desc)-1}', 'tmp', (0,3 if len(tensors) > 1 else 2))

        self.contract('Angle', self.prefix+'0', 'tmp', self.prefix+'0')
        self.contract('X', self.prefix+f'{len(desc)-1}', 'tmp', self.prefix+f'{len(desc)-1}')

CNOT = MPO(
    tensors=[
        np.array([[[1,0],[0,0]], [[0,0],[0,1]]]), 
        np.array([[[1,0],[0,1]], [[0,1],[1,0]]]),
    ],
    tensor_prefix='CNOT',
)

CNOT_inv = MPO(
    tensors=[
        np.array([[[1,0],[0,1]], [[0,1],[1,0]]]),
        np.array([[[1,0],[0,0]], [[0,0],[0,1]]]),
    ],
    tensor_prefix='CNOT_inv',
)

CZ = MPO(
    tensors=[
        np.array([[[1,0],[0,0]], [[0,0],[0,1]]]), 
        np.array([[[1,1],[0,0]], [[0,0],[1,-1]]]),
    ],
    tensor_prefix='CZ',
)

SWAP = MPO(
    tensors=[
        np.array([[[1,0,0,0],[0,1,0,0]], [[0,0,1,0],[0,0,0,1]]]),
        np.array([[[1,0,0,0],[0,0,1,0]], [[0,1,0,0],[0,0,0,1]]]),
    ],
    tensor_prefix='SWAP'
)

H = MPO(tensors=[np.array([[1,1],[1,-1]])/np.sqrt(2)], tensor_prefix='Hadamard')

X = MPO(tensors=[np.array([[0,1],[1,0]])], tensor_prefix='X_gate')

Y = MPO(tensors=[np.array([[0,-1j],[1j,0]])], tensor_prefix='Y_gate')

Z = MPO(tensors=[np.array([[1,0],[0,-1]])], tensor_prefix='Z_gate')

S = MPO(tensors=[np.array([[1,0],[0,1j]])], tensor_prefix='S_gate')

T = MPO(tensors=[np.array([[1,0],[0,1.0/np.sqrt(2)+1j/np.sqrt(2)]])], tensor_prefix='T_gate')

