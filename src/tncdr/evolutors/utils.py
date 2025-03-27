import random
from qibo import gates

from tncdr.evolutors.tensor_network import TensorNetwork

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

def sample_random_pauli_gate(qubit:int):
    """
    Sample a random one-qubit gate applyed to a given qubit.
    """
    random_letter = random.choice(one_qubit_cliff)
    return getattr(gates, random_letter)(q=qubit)

def _link_to_dummy(tn:TensorNetwork, dummy:str, tensor:str, tensor_direction:int, edge_id:str='v_link'):
    """
    Given a TensorNetwork, replace the dummy node with a new tensor, connecting the edge to a specified index.

    Args:
        tn (TensorNetwork): TensorNetwork containing both dummy and tensor
        dummy (str): Dummy node id
        tensor (str): New tensor id
        tensor_direction (str): Index of the tensor to be connected in replacement of dummy
        edge_id (str): name of the link replacing the dummy link.
    """

    T,d,e,data = list(tn.tensornet.in_edges(dummy, data=True, keys=True))[0]
    dummy_direction = data["directions"][0]
    tn.remove_edge(T,d,e)
    tn.add_edge(T, tensor, edge_id, (dummy_direction,tensor_direction))
    tn.tensornet.remove_node(dummy)