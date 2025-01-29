import numpy as np

paulis = {
    'I':np.eye(2, dtype=np.complex64),
    'X':np.array([[0,1],[1,0]], dtype=np.complex64),
    'Y':np.array([[0,-1j],[1j,0]], dtype=np.complex64),
    'Z':np.array([[1,0],[0,-1]], dtype=np.complex64)
}