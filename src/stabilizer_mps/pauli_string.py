from copy import copy

# CONVERSION TABLES

# The two-bit encoding of each Pauli operator is implemented as a python integer. 
# Pauli strings are also python integers, but consist of 2n bits, where n is the number of qubits.
single_pauli_to_xz = {'I':0, 'X':1, 'Z':2, 'Y':3}
xz_to_single_pauli = {0:'I', 1:'X', 2:'Z', 3:'Y'}

def string_to_xy(description:str)->int:
    """
    Converts a Pauli string, provided as an explicit string of 'I', 'X', 'Y' and 'Z' into the corresponding XZ encoding.
    """
    return sum((single_pauli_to_xz[p]<<(2*q) for q,p in enumerate(description)), start=0)

def xy_to_string(xz_desc:int)->str:
    """
    Converts a python integer, interpreted as the XZ encoding of a Pauli string, into the corresponfing explicit string.
    """
    return ''.join([xz_to_single_pauli[3 & xz_desc >> (2*q)] for q in range(num_qubits(xz_desc))])

# IMPLEMENTATIONS OF THE BASIC OPERATIONS AT THE BINARY LEVEL

def xy_prod(xz_desc1:int, xz_desc2:int)->int:
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

# HUMAN FRIENDLY INTERFACE

class Pauli():
    """
    Pauli string XZ representation (up to a global phase)
    
    #TODO Add the tracking of the phase. Is it really relevant? Idk...
    """

    def __init__(self, description:str|int)->None:

        self.xz = description if type(description) is int else string_to_xy(description)

    def __repr__(self)->str:
        return xy_to_string(self.xz)

    def __matmul__(self, other)->'Pauli':

        # This operation is not in-place, it will create a new Pauli instance.
        result = copy(self)
        result.xz = xy_prod(self.xz, other.xz)
        return result
    
    def __getitem__(self, qubit:int)->int:
        return ith_qubit(self.xz, qubit)
    
    def __setitem__(self, qubit:int, pauli:int)->None:
        self.xz = replace_qubit(self.xz, qubit, pauli)

    def _has_X(self, qubit:int)->bool:
        return has_X(self.xz, qubit)
    
    def _has_Z(self, qubit:int)->bool:
        return has_Z(self.xz, qubit)

    def apply(self, T)->None:
        """
        Update the Pauli string according to a Tableau `T`.
        """

        # Create a workspace to collect the changes happening to the involved qubits
        new_sub_space = Pauli(0)
        
        # Apply the tableau updates if the current state has a X or Z respectively
        for i,q in enumerate(T.qubits):
            new_sub_space = new_sub_space@T.XTableau.conjugates[i] if self._has_X(q) else new_sub_space
            new_sub_space = new_sub_space@T.ZTableau.conjugates[i] if self._has_Z(q) else new_sub_space
        
        # Replace the qubits to be updated into the full Pauli string
        for i,q in enumerate(T.qubits):
            self[q] = new_sub_space[i]