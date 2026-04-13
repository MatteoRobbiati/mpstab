from __future__ import annotations

from typing import Any

from qibo.gates.abstract import ParametrizedGate
from quimb.gates import I, X, Y, Z
from quimb.tensor import (
    CircuitMPS,
    MatrixProductOperator,
    MatrixProductState,
    MPO_identity,
    MPO_product_operator,
)
import cotengra as ctg

from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine

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


pauli_map = {"X": X, "Y": Y, "Z": Z, "I": I}


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


class QuimbEngine(TensorNetworkEngine):
    """
    Tensor network engine using Quimb for tensor network manipulations and contractions. 
    The engine supports caching of contraction paths using cotengra's ReusableOptimizer. 
    """
    def __init__(self, backend: str = "numpy", cache: bool = False, cache_directory: str | None = "contractions_cache"):
        """
        Initialize the engine with backend and persistent contraction optimizer.
        
        Parameters
        ----------
        backend : str, optional
            Quimb backend: Numpy (default), Jax, Torch
        cache : bool, optional
            If true, the optimizer caches contraction paths
        cache_directory : str, optional
            The directory where contraction paths will be saved. 
            If it doesn't exist, cotengra will create it.
        """
        if backend == "jax":
            import jax.numpy as jnp
            self.np = jnp
        
        elif backend == "numpy":
            import numpy as np
            self.np = np
        
        elif backend == "torch":
            import torch
            self.np = torch

        else: raise ValueError(f"Unsupported quimb backend: {backend}")

        self.backend = backend

        if cache == True:
            self.optimizer = ctg.ReusableHyperOptimizer(
                        directory=cache_directory, 
                        minimize='flops',          
                        max_repeats=128,           
                        progbar=False
                    )        
        else : self.optimizer = "auto-hq"

    def PauliExp(self, pauli_string, theta):
        """
        Returns the MPO for exp(-i * theta/2 * P) where P is a Pauli string. The euler formula is used:
        exp(-i * theta/2 * P) = cos(theta/2) * I + i * sin(theta/2) * P.
        """
        L = len(pauli_string)

        pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]

        id_mpo = MPO_identity(L, phys_dim=2)
        pauli_mpo = MPO_product_operator(pauli_matrices)

        if self.backend == "torch":
            pauli_mpo.apply_to_arrays(lambda x: self.np.as_tensor(x))
            id_mpo.apply_to_arrays(lambda x: self.np.as_tensor(x))
            theta = self.np.as_tensor(theta)

        rotation_mpo = (self.np.cos(theta / 2) * id_mpo).add_MPO(-1j * self.np.sin(theta / 2) * pauli_mpo)

        return rotation_mpo

    def build_circuit_mps(
        self,
        n: int,
        initial_state_amplitudes: Any,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """
        Builds a Circuit MPS object in Quimb. The underlying tensor network is a Matrix Product State. truncation_fidelity is
        initialized.
        """

        if initial_state_circuit is not None:
            
            return _qibo_circuit_to_quimb(
                nqubits=n, 
                qibo_circ=initial_state_circuit, 
                max_bond=max_bond_dimension,
                to_backend=self.np.asarray
            ).psi
        
        else: raise NotImplementedError("Building a CircuitMPS from state amplitudes is not implemented in the QuimbEngine.")


    def pauli_mpo(self, pauli_string: str | object):
        """
        Build a Matrix Product Operator (MPO) representing a given Pauli string.
        """

        pauli_matrices = [pauli_map[s.upper()] for s in pauli_string]
        pauli_mpo = MPO_product_operator(pauli_matrices)
        pauli_mpo.add_tag("MPO")
        if self.backend == "torch":
            pauli_mpo.apply_to_arrays(lambda x: self.np.as_tensor(x))            

        return pauli_mpo


    def expval(
        self, state_circuit: MatrixProductState, operator: MatrixProductOperator
    ):
        """
        Compute the expectation value of `operator` on `state_circuit`.
        - state_circuit: MatrixProductState representing the state of the system
        - operator: MatrixProductOperator representing the observable whose expectation value we want to compute
        Due to truncation we loose unitary norm, so normalizing is needed when computing expectation.
        """
        self.norm = state_circuit.norm(squared=True).real
        circuit_tn_dag = state_circuit.reindex(
            {f"k{i}": f"b{i}" for i in range(state_circuit.L)}
        )
        return (circuit_tn_dag.H & operator & state_circuit).contract(
                backend=self.backend, 
                optimize=self.optimizer ).real / self.norm
        

    def pauli_rot(
        self,
        state_circuit: MatrixProductState,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """
        Apply a Pauli string rotation MPO to an MPS and return the updated object. SVD is performed with compression
        given by specified bond dimension.
        """
        rotation_mpo = self.PauliExp(generator, angle)

        if self.backend == "torch":
            rotation_mpo.apply_to_arrays(lambda x: self.np.as_tensor(x))            

        state_circuit.gate_with_mpo(
            rotation_mpo, 
            inplace=True, 
            max_bond=max_bond_dimension
        )
