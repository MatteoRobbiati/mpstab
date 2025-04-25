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
    "rz": "RZ",
    "gpi2": "GPI2",
}

one_qubit_cliff = "HXYZ"

def sample_random_pauli_gate(qubit):
    """
    Sample a random one-qubit gate applyed to a given qubit.
    """
    random_letter = random.choice(one_qubit_cliff)
    return getattr(gates, random_letter)(q=qubit)

def _link_to_dummy(tn, dummy, tensor, tensor_direction, edge_id='v_link'):

    T,d,e,data = list(tn.tensornet.in_edges(dummy, data=True, keys=True))[0]
    dummy_direction = data["directions"][0]
    tn.remove_edge(T,d,e)
    tn.add_edge(T, tensor, edge_id, (dummy_direction,tensor_direction))
    tn.tensornet.remove_node(dummy)