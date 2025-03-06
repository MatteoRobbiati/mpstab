import random
from qibo import gates   

gate2generator = {
    "rx": "X",
    "ry": "Y",
    "rz": "Z",
}

gate2tableau = {
    "cx": "CNOT",
    "h": "H",
    "s": "S",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "swap": "SWAP",
    "cz": "CZ",
    "ry": "RY",
}

one_qubit_cliff = "HXYZ"

def sample_random_pauli_gate(qubit):
    """
    Sample a random one-qubit gate applyed to a given qubit.
    """
    random_letter = random.choice(one_qubit_cliff)
    return getattr(gates, random_letter)(q=qubit)