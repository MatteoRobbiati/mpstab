from functools import lru_cache

import numpy as np

from tncdr.evolutors.tensor_network.tensor_network import TensorNetwork 
from tncdr.evolutors.tensor_network.tn_utils import paulis

# Compute S tensors
def _compute_all_s_tensors()->np.ndarray:
    return {p:_S(p) for p in paulis.keys()}

# Measurements

@lru_cache(maxsize=None)
def basis(initial_state_name:str)->np.ndarray:
    if initial_state_name=='0': return np.array([1.0,0.0])
    if initial_state_name=='1': return np.array([0.0,1.0])
    if initial_state_name=='+': return np.array([1.0,1.0])/np.sqrt(2)
    if initial_state_name=='-': return np.array([1.0,-1.0])/np.sqrt(2)
    raise ValueError(f'Basis element descriptor must be either 0,1,+ or -. "Given {initial_state_name}".')

@lru_cache(maxsize=None)
def theta_state(theta:float)->np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([c,-1j*s])

@lru_cache(maxsize=None)
def X_state()->np.ndarray:
    return np.array([1.0,1.0])

# @lru_cache(maxsize=None)
def generic_state(alpha:complex, beta:complex)->np.ndarray:
    
    # Projection in *normalized* the pauli basis
    return np.array([alpha,beta])

def _S(p:str)->TensorNetwork:

    # Index order: i,j, alpha, beta

    tn = TensorNetwork()
    tn.add_pauli_pair('gamma', p0='I', p1=p)
    tn.add_copy_tensor('copy', n=2)

    tn.add_edge('gamma','copy', 'link', (0,0))
    tn.contract('gamma', 'copy', 'link', 'S')
    return tn.tensornet.nodes['S']['tensor']
