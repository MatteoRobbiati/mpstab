from __future__ import annotations
from typing import Any
from numpy import cos, sin
from mpstab.backends.tensor_networks.abstract import TensorNetworkEngine

from quimb.gates import X, Y, Z, I
from quimb.tensor import CircuitMPS, MPO_product_operator, MPO_identity
from qibo.gates.abstract import ParametrizedGate


GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}


pauli_map = {'X': X, 'Y': Y, 'Z': Z, 'I': I}


def _qibo_circuit_to_quimb(nqubits, qibo_circ, **circuit_kwargs):
    """
    Convert a Qibo Circuit to a Quimb Circuit. Measurement gates are ignored. If are given gates not supported by Quimb, an error is raised.

    Parameters
    ----------
    qibo_circ : qibo.models.circuit.Circuit
        The circuit to convert.
    quimb_circuit_type : type
        The Quimb circuit class to use (Circuit, CircuitMPS, etc).
    circuit_kwargs : dict
        Extra arguments to pass to the Quimb circuit constructor.

    Returns
    -------
    circ : quimb.tensor.circuit.Circuit
        The converted circuit.
    """

    circ = CircuitMPS(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        gate_name = getattr(gate, "name", None)
        quimb_gate_name = GATE_MAP.get(gate_name, None)
        if quimb_gate_name == "measure":
            continue
        if quimb_gate_name is None:
            raise ValueError(f"Gate {gate_name} not supported in Quimb backend.")

        params = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())

        is_parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if is_parametrized:
            circ.apply_gate(
                quimb_gate_name, *params, *qubits, parametrized=is_parametrized
            )
        else:
            circ.apply_gate(
                quimb_gate_name,
                *params,
                *qubits,
            )

    return circ


def PauliExp(pauli_string, theta):
    """
    Returns the MPO for exp(i * theta * P) where P is a Pauli string. The euler formula is used:
    exp(i * theta * P) = cos(theta) * I + i * sin(theta) * P.
    """
    L = len(pauli_string)
    
    pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]
    
    id_mpo = MPO_identity(L, phys_dim=2, dtype='complex128')
    pauli_mpo = MPO_product_operator(pauli_matrices)
    rotation_mpo = (cos(theta) * id_mpo).add_MPO(1j * sin(theta) * pauli_mpo)
    
    return rotation_mpo


class QuimbEngine(TensorNetworkEngine):
    """
    Thin adapter that exposes the minimal API required by HybridSurrogate
    while reusing the existing evolutors.tensor_network implementation.
    """


    def build_circuit_mps(self, n: int, initial_state_amplitudes: Any, initial_state_circuit: Any, max_bond_dimension: int | None = None):
        """
        Builds a Circuit MPS object in Quimb. The underlying tensor network is a Matrix Product State.
        """

        if initial_state_circuit is not None:
            return _qibo_circuit_to_quimb(nqubits=n, qibo_circ=initial_state_circuit, max_bond=max_bond_dimension)
        else:
            raise NotImplementedError("Building a CircuitMPS from state amplitudes is not implemented in the QuimbEngine.")


    def pauli_mpo(self, pauli_string: str | object):
        """
        Build a Matrix Product Operator (MPO) representing a given Pauli string.
        """

        pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]
        pauli_mpo = MPO_product_operator(pauli_matrices)
        pauli_mpo.add_tag('MPO')
        
        return pauli_mpo


    def expval(self, state_circuit: CircuitMPS, operator: any):
        """
        Compute the expectation value of `operator` on `state_circuit`.
        - state_circuit: CircuitMPS representing the state of the system
        - operator: MatrixProductOperator representing the observable whose expectation value we want to compute
        """

        state = state_circuit.psi
        circuit_tn_dag = state.reindex({f'k{i}': f'b{i}' for i in range(state_circuit.N)})
        
        return (circuit_tn_dag.H & operator & state).contract(optimize='auto-hq').real
    

    def pauli_rot(self, state_circuit: CircuitMPS, generator: str, angle: float) -> CircuitMPS:
        """
        Apply a Pauli string rotation MPO to a CircuitMPS and return the updated object.
        """
        rotation_mpo = PauliExp(generator, angle)
        
        # Apply the MPO directly to the underlying MPS state.
        # We use the circuit's default gate_opts (max_bond, cutoff, etc.)
        state_circuit.psi.gate_with_mpo_(
            rotation_mpo, 
            inplace=True, 
            max_bond=state_circuit.gate_opts['max_bond'],
        )
        
        return state_circuit


