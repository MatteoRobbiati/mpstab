import numpy as np

from tncdr.stabilizer_mps.tensor_network import TensorNetwork

# Import W tensor generator
from tncdr.stabilizer_mps.w_utils import _compute_all_w_tensors

# Import 'spatial' measurements (i.e. initial state and observable)
from tncdr.stabilizer_mps.w_utils import pauli_pauli_expansion, basis_pauli_expansion
# Import 'temporal' measurements (i.e. theta and X fictitous states)
from tncdr.stabilizer_mps.w_utils import  X_pauli_expansion, theta_pauli_expansion

Ws = _compute_all_w_tensors()

def sample_random_pauli(n:int):
    possible_symbols = list('IXYZ')
    return ''.join([possible_symbols[np.random.randint(len(possible_symbols))] for _ in range(n)])

def pauli_tensors(pauli_string:str):

    for p in pauli_string:
        if 'IXYZ'.find(p) >= 0:
            yield Ws[p]

def _pointwise_expectation(state, pauli):

    expect = 1
    for s,p in zip(list(state), list(pauli)):
        if s == '1':
            if p == 'I': continue
            if p == 'Z': expect *= -1
            else: return 0
        if s == '0':
            if p in list('IZ'): continue
            else: return 0

        if s == '-':
            if p == 'I': continue
            if p == 'X': expect *= -1
            else: return 0
        if s == '+':
            if p in list('IX'): continue
            else: return 0

        #assert False, f'State must composed only of "10+-", given {s}, {p}'
    return expect

def _get_phase_and_string(pauli):
    string_repr = str(pauli)
    phase = 1
    if string_repr[0]=='-':
        phase *= -1
        string_repr = string_repr[1:]
    if string_repr[0]=='i':
        phase *= 1j
        string_repr = string_repr[1:]

    return phase, string_repr
    

def analytical_solution(rotation_pauli, observable, theta, initial_state):

    from tncdr.stabilizer_mps.pauli_string import Pauli
    f = Pauli(rotation_pauli)@Pauli(observable)@Pauli(rotation_pauli)
    l = Pauli(observable)@Pauli(rotation_pauli)
    r = Pauli(rotation_pauli)@Pauli(observable)

    pf, sf = _get_phase_and_string(f)
    pl, sl = _get_phase_and_string(l)
    pr, sr = _get_phase_and_string(r)

    c,s = np.cos(theta/2), np.sin(theta/2)
    solution = c**2*_pointwise_expectation(initial_state, observable)
    solution += s**2*_pointwise_expectation(initial_state, sf)*pf
    solution += 1j*s*c*_pointwise_expectation(initial_state, sl)*pl
    solution -= 1j*s*c*_pointwise_expectation(initial_state, sr)*pr
    return np.real(solution)


def main():

    N=3
    
    rotation_pauli = 'YXZ'#sample_random_pauli(N)
    observable = 'XXZ'#sample_random_pauli(N)
    theta =  2.5943124162115114#2*np.pi*np.random.rand()
    intial_state = '1'*N

    print(f'Rotation generator: {rotation_pauli}')
    print(f'H: {observable}')
    print(f'rho_0: |{intial_state}X{intial_state}|')
    print(f'theta: {theta}')
    print(f'Analytical solution: {analytical_solution(rotation_pauli, observable, theta, intial_state)}')

    # Build TensorNetwork
    tn = TensorNetwork()

    # Add initial state
    for n, state in enumerate(intial_state):
        tn.add_tensor(f'T{n}', tensor=basis_pauli_expansion(state))
    
    # Add one rotation
    for n, w_tensor in enumerate(pauli_tensors(rotation_pauli)):
        tn.add_tensor(id=f'W{n}', tensor=w_tensor)
        tn.add_edge(f'T{n}', f'W{n}', 'v_link', (0,2))

    tn.add_tensor(id='theta', tensor=theta_pauli_expansion(theta=theta))
    tn.add_edge('theta', 'W0', 'h_link', (0,1))

    for n in range(N-1):
        tn.add_edge(f'W{n}', f'W{n+1}', 'h_link', (0,1))

    tn.add_tensor(id='X', tensor=X_pauli_expansion())
    tn.add_edge(f'W{N-1}', 'X', 'h_link', (0,0))

    # Connect to the observable
    for n, o in enumerate(observable):
        tn.add_tensor(f'O{n}', tensor=pauli_pauli_expansion(o))
        tn.add_edge(f'O{n}', f'W{n}', 'v_link', (0,3))

    # Contract the TensorNetwork
    tn.contract('theta', 'W0', 'h_link', 'tmp')
    for n in range(N-1):
        tn.contract(f'T{n}', 'tmp', 'v_link', 'tmp2')
        tn.contract(f'O{n}', 'tmp2', 'v_link', 'tmp3')
        tn.contract(f'tmp3', f'W{n+1}', 'h_link', 'tmp')
    
    tn.contract(f'T{N-1}', 'tmp', 'v_link', 'tmp2')
    tn.contract(f'O{N-1}', 'tmp2', 'v_link', 'tmp3')
    tn.contract('tmp3', 'X', 'h_link', 'F')

    # Output the result
    print(f'TN solution: {tn.tensornet.nodes['F']['tensor'].item()}')

if __name__ == '__main__':
    main()

