from copy import copy

# CONVERSION TABLES

# The two-bit encoding of each Pauli operator is implemented as a python integer. 
# Pauli strings are also python integers, but consist of 2n bits, where n is the number of qubits.
single_pauli_to_xz = {'I':0, 'X':1, 'Z':2, 'Y':3}
xz_to_single_pauli = {0:'I', 1:'X', 2:'Z', 3:'Y'}

phase_to_xz = {'i':1, '-':2, '-i':3}
xz_to_phase = {0:'', 1:'i', 2:'-', 3:'-i'}

def string_to_xz(description:str)->int:
    """
    Convert a Pauli string, provided as an explicit string of 'I', 'X', 'Y' and 'Z' into the corresponding XZ encoding.
    """
    return sum((single_pauli_to_xz[p]<<(2*q) for q,p in enumerate(reversed(description))), start=0)

def xz_to_string(xz_desc:int)->str:
    """
    Converts a python integer, interpreted as the XZ encoding of a Pauli string, into the corresponfing explicit string.
    """
    return ''.join([xz_to_single_pauli[3 & xz_desc >> (2*q)] for q in range(num_qubits(xz_desc)-1,-1,-1)])

def xz_to_string_phase(xz_desc:int, phase:int, n:int)->str:
    """
    Given an XZ encoding and a phase, it computes the phase of the explicit pauli string
    """
    shift = 0
    for q in range(n):
        shift += phase_to_xz['-i'] if ((xz_desc >> 2*q) & (xz_desc >> (2*q+1))) & 1 else 0
    phase = update_phase(phase, shift)
    return xz_to_phase[phase]

# IMPLEMENTATIONS OF THE BASIC OPERATIONS AT THE BINARY LEVEL

def xz_prod(xz_desc1:int, xz_desc2:int)->int:
    """
    Implements the matrix product (up to a global phase) of two Pauli strings in the XZ encoding, i.e. the bitwise XOR.
    """
    return xz_desc1 ^ xz_desc2

def has_X(xz_desc:int, qubit:int)->bool:
    """
    Returns wether the qubit-th Pauli operator has an 'X' in its encoding, i.e. if it is either 'X' or 'Y'
    """
    return bool(1 << (2*qubit) & xz_desc)

def has_Z(xz_desc:int, qubit:int)->bool:
    """
    Returns wether the `qubit`-th Pauli operator has a 'Z' in its encoding, i.e. if it is either 'Z' or 'Y'
    """
    return bool(2 << (2*qubit) & xz_desc)

def reset_qubit(xz_desc:int, qubit:int)->int:
    """
    Returns a copy of the Pauli string, but replacing the `qubit`-th qubit with the identity 'I'
    """
    return xz_desc & (~(3 << (2*qubit)))

def ith_qubit(xz_desc:int, qubit:int)->int:
    """
    Returns the XZ encoding of the Pauli operator in position `qubit`
    """
    return 3 & (xz_desc >> (2*qubit))

def replace_qubit(xz_desc:int, qubit:int, replacement:int)->int:
    """
    Replaces the encoding of the Pauli operator in position `qubit` with `replacement`
    """
    return reset_qubit(xz_desc, qubit) | (replacement << 2*qubit)

def num_qubits(xz_desc:int)->int:
    """
    Estimates the number of qubit encoded in the XZ encoding of a Pauli string, ignoring the identities appering on the right.

    Example: num_qubits(XIZXYII) = 5 and not 7, since the last two 'I' are ignored, i.e. XIZXYII -> XIZXY 
    """
    return int((len(bin(xz_desc))-1)//2)

def phase_filp(xz_desc1:int, xz_desc2:int, n:int)->bool:
    """
    Computes whether the sign should be flipped after multiplication of two pauli strings, determining the change in the global
    phase due to the product.
    """
    det = ((xz_desc1 >> 1) & xz_desc2)
    flip = False
    for q in range(n):
        flip = not flip if ((det>>(2*q)) & 1) else flip
    return flip

def initial_phase(xz_desc:int, n:int, phase0:int=0)->int:
    """
    Explicitely compute the hidden phase of the pauli string in the XZ encoding, due to the fact that XZ = -iY and not Y.
    """
    shift = 0
    for q in range(n):
        shift += phase_to_xz['i'] if ((xz_desc >> 2*q) & (xz_desc >> (2*q+1))) & 1 else 0
    return update_phase(phase0, shift)

def update_phase(phase, shift):
    return (phase + shift) & 3

# HUMAN FRIENDLY INTERFACE

class Pauli():
    """
    Pauli string XZ representation, including a global phase in {1,-1,i,-i}.
    """

    def __init__(self, description:str|int, n:int|None=None)->None:
        
        if type(description) is int:
            self.xz = description
            self.n = n if n is not None else num_qubits(description)
            self.phase = initial_phase(self.xz, self.n)
            return
        
        phase0 = 0
        if description[0] in phase_to_xz.keys():
            phase0 += phase_to_xz[description[0]]
            description = description[1:]
            if description[0] in phase_to_xz.keys():
                phase0 += phase_to_xz[description[0]]
                description = description[1:]

        self.xz = string_to_xz(description)
        self.n = len(description)
        self.phase = initial_phase(self.xz, self.n, phase0)

    def __repr__(self)->str:
        return xz_to_string_phase(self.xz, self.phase, self.n)+xz_to_string(self.xz)

    def __matmul__(self, other:'Pauli')->'Pauli':

        # This operation is not in-place, it will create a new Pauli instance.
        result = copy(self)
        result.xz = xz_prod(self.xz, other.xz)
        result._update_phase(other.phase)
        
        if phase_filp(self.xz, other.xz, self.n):
            result._update_phase(phase_to_xz['-'])
        return result
    
    def __getitem__(self, qubit:int)->int:
        return ith_qubit(self.xz, qubit)
    
    def __setitem__(self, qubit:int, pauli:int)->None:
        self.xz = replace_qubit(self.xz, qubit, pauli)

    def _has_X(self, qubit:int)->bool:
        return has_X(self.xz, qubit)
    
    def _has_Z(self, qubit:int)->bool:
        return has_Z(self.xz, qubit)
    
    def _update_phase(self, new_phase:int)->None:
        self.phase = update_phase(self.phase, new_phase)

    def apply(self, T)->None:
        """
        Update the Pauli string according to a Tableau `T`.
        """

        # Create a workspace to collect the changes happening to the involved qubits
        new_sub_space = Pauli(0, n=len(T.qubits))
        
        # Apply the tableau updates if the current state has a X or Z respectively
        for i,q in enumerate(T.qubits):
            if self._has_X(q):
                new_sub_space = new_sub_space@T.XTableau.conjugates[i]
                #new_sub_space._update_phase(T.XTableau.signs[i])
            if self._has_Z(q):
                new_sub_space = new_sub_space@T.ZTableau.conjugates[i]
                #new_sub_space._update_phase(T.ZTableau.signs[i])

        # Replace the qubits to be updated into the full Pauli string
        for i,q in enumerate(T.qubits):
            self[q] = new_sub_space[i]
        
        # Update the global phase
        self._update_phase(new_sub_space.phase) 