from functools import lru_cache

import numpy as np

from tncdr.evolutors.tensor_network.tensor_network import TensorNetwork 
from tncdr.evolutors.tensor_network.tn_utils import paulis

# Tensor containg the full pauli basis
all_pauli_tensor = np.array([paulis['I'], paulis['X'], paulis['Y'],paulis['Z']])

# Compute W tensors
def _compute_all_w_tensors()->np.ndarray:
    Ws = {}
    for p in paulis.keys():
        tn = _W_tn(p)
        Ws[p] = _W_contract(tn)
    return Ws

# Measurements
@lru_cache(maxsize=None)
def pauli_pauli_expansion(p:str)->np.ndarray:
    tensor = np.zeros(4)
    tensor['IXYZ'.find(p)] = np.sqrt(2)
    return tensor

@lru_cache(maxsize=None)
def basis_pauli_expansion(initial_state_name:str)->np.ndarray:
    if initial_state_name=='0': return np.array([1.0,0.0,0.0,1.0])/np.sqrt(2)
    if initial_state_name=='1': return np.array([1.0,0.0,0.0,-1.0])/np.sqrt(2)
    if initial_state_name=='+': return np.array([1.0,1.0,0.0,0.0])/np.sqrt(2)
    if initial_state_name=='-': return np.array([1.0,-1.0,0.0,0.0])/np.sqrt(2)
    raise ValueError(f'Basis element descriptor must be either 0,1,+ or -. "Given {initial_state_name}".')

@lru_cache(maxsize=None)
def theta_pauli_expansion(theta:float)->np.ndarray:
    c, s = np.cos(theta/2), np.sin(theta/2)
    # Projection in *normalized* the pauli basis
    return np.array([1.0,0.0,2*c*s,c**2-s**2])/np.sqrt(2)

@lru_cache(maxsize=None)
def X_pauli_expansion():
    return np.array([1.0,1.0,0.0,0.0])*np.sqrt(2)

def _W_tn(p:str)->TensorNetwork:

    # Create the TensorNetwork object
    tn = TensorNetwork()

    #Add the fundamental tensors
    tn.add_copy_tensor('C', n=2)
    tn.add_pauli_pair('P', p0='I', p1=p)
    tn.add_copy_tensor('C+', n=2)
    tn.add_pauli_pair('P+', p0='I', p1=p)

    # Create the nodes responsible for the projection into the Pauli basis
    tn.add_tensor('mu_basis', all_pauli_tensor)
    tn.add_tensor('nu_basis', all_pauli_tensor)
    tn.add_tensor('alpha_basis', all_pauli_tensor)
    tn.add_tensor('beta_basis', all_pauli_tensor)

    # Connect the nodes
    tn.add_edge('P', 'C', 'X', (0,0))
    tn.add_edge('P+', 'C+', 'X+', (0,0))
        
    tn.add_edge('mu_basis', 'P', 'mu_link', (2,1))
    tn.add_edge('mu_basis', 'P+', 'mu_link+', (1,2))

    tn.add_edge('nu_basis', 'P', 'nu_link', (2,2))
    tn.add_edge('nu_basis', 'P+', 'nu_link+', (1,1))
    
    tn.add_edge('alpha_basis', 'C', 'alpha_link', (2,1))
    tn.add_edge('alpha_basis', 'C+', 'alpha_link+', (1,2))
    
    tn.add_edge('beta_basis', 'C', 'beta_link', (2,2))
    tn.add_edge('beta_basis', 'C+', 'beta_link+', (1,1))

    return tn

def _W_contract(tn:TensorNetwork)->np.ndarray:
    # This contraction order yields W with tensor direction ordered
    # as follows: 0->beta, 1->alpha, 2->nu, 3->mu

    tn.contract('mu_basis', 'P', 'mu_link', 't1')
    tn.contract('t1', 'P+', 'mu_link+', 't2')

    tn.contract('nu_basis', 't2', 'nu_link', 't1')
    tn.contract('t1', 't1', 'nu_link+', 't2')

    tn.contract('t2', 'C', 'X', 't1')
    tn.contract('t1', 'C+', 'X+', 't2')
    
    tn.contract('alpha_basis', 't2', ['alpha_link', 'alpha_link+'], 't1')
    tn.contract('beta_basis', 't1', ['beta_link','beta_link+'], 'F')

    # Division by 4 restores the normalization w.r.t. to the Pauli basis, 
    # which in the _w_tn() are un-normalized. It is done here rather than
    # earlier to improve numerical stability.
    return np.real(tn.tensornet.nodes['F']['tensor']).astype(float)/4