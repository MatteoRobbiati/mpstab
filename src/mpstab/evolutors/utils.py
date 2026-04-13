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
    "ry": "RY",
    "rx": "RX",
    "gpi2": "GPI2",
    "sdg": "Sdg",
}

one_qubit_cliff = "HXYZ"


def sample_random_pauli_gate(qubit):
    """
    Sample a random one-qubit gate applyed to a given qubit.
    """
    random_letter = random.choice(one_qubit_cliff)
    return getattr(gates, random_letter)(q=qubit)


def _link_to_dummy(tn, dummy, tensor, tensor_direction, edge_id="v_link"):

    T, d, e, data = list(tn.tensornet.in_edges(dummy, data=True, keys=True))[0]
    dummy_direction = data["directions"][0]
    tn.remove_edge(T, d, e)
    tn.add_edge(T, tensor, edge_id, (dummy_direction, tensor_direction))
    tn.tensornet.remove_node(dummy)


def validate_pauli_observable(observable: str, nqubits: int) -> None:
    """
    Validate that a Pauli observable string is well-formed.

    Args:
        observable: Pauli observable string (e.g., "ZZZZZ", "XYZIX")
        nqubits: Number of qubits in the system

    Raises:
        ValueError: If observable contains invalid characters or has incorrect length
    """
    # Validate observable string contains only Pauli operators
    valid_paulis = set("IXYZ")
    invalid_chars = set(observable) - valid_paulis
    if invalid_chars:
        raise ValueError(
            f"Observable string contains invalid characters: {invalid_chars}. "
            f"Observable strings should only contain Pauli operators: I, X, Y, Z. "
            f"Do not include signs, coefficients, or other characters. "
            f"Examples: 'ZZZZZ' or 'XYZIX', not '2*ZZZZZ' or '-ZZ'."
        )

    # Validate observable string length matches number of qubits
    if len(observable) != nqubits:
        raise ValueError(
            f"Observable string length ({len(observable)}) does not match "
            f"the number of qubits ({nqubits}). "
            f"Expected a Pauli string of length {nqubits}, "
            f"e.g., '{'Z'*nqubits}' for measuring Z operators on all qubits."
        )
