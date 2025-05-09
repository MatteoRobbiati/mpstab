from typing import Union

import numpy as np

from tncdr.evolutors.stabilizer.pauli_string import Pauli

from tncdr.evolutors.tensor_network.operators import MPO
from tncdr.evolutors.tensor_network.operators.utils import _compute_all_s_tensors, theta_state, X_state

s_tensors = _compute_all_s_tensors()

class PauliExp(MPO):
    """
    MPO implementation of the unitary operator exp(-i theta/2 P), for some real parameter theta and a pauli string P
    """

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

        self.add_tensor('Angle', tensor=theta_state(-phase*theta))
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