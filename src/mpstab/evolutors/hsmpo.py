from dataclasses import dataclass
from typing import Union

import numpy as np
from qibo import Circuit
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.engines import (
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.evolutors.utils import gate2generator
from mpstab.models.ansatze import Ansatz, CircuitAnsatz


@dataclass
class HSMPO:
    """
    Construct an hybrid stabilizer MPO surrogate of a given quantum circuit.

    The tensor-network part is now engine-pluggable via the TensorNetworkEngine API.
    """

    ansatz: Union[Ansatz, Circuit]
    max_bond_dimension: int = None
    initial_state: Circuit = None

    def __post_init__(self):
        # Wrap the qibo circuit with our ansatz in case a pure qibo
        # circuit is provided
        if isinstance(self.ansatz, Circuit):
            self.ansatz = CircuitAnsatz(qibo_circuit=self.ansatz)

        # Initial state is zero by default
        if self.initial_state is None:
            self.initial_state = Circuit(self.ansatz.nqubits)

        # Default engines will be set here (stabilizers + tensor-network)
        # and TN will be initialised by _init_tn.
        self.set_engines()

        # Add the initial state, which is |0> by default
        self._init_tn(self.max_bond_dimension)

    def _init_tn(self, max_bond_dimension: int | None = None):
        """
        Initialize the tensor network (MPS) to the factorized state specified
        by the initial_state circuit.

        The code still builds the per-qubit amplitudes from the qibo Circuit as
        before, but the actual MPS object is created through the TN engine.

        Passing both amplitudes for native engine and and full initial circuit for quimb.
        """

        amplitudes = []
        for q in range(self.nqubits):

            light_circ, light_dict = self.initial_state.light_cone(q)

            # Check whether the initial state is constructed properly
            # (with one-qubit gates only)
            if len(light_dict.items()) > 1:
                raise ValueError(
                    "Ensure only 1-qubit gates compose the initial state preparation circuit."
                )

            # Add the initial state tensor (single-qubit amplitude vector)
            amplitudes.append(light_circ().state())

        self.mps = self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_amplitudes=np.array(amplitudes),
            initial_state_circuit=self.initial_state,
            max_bond_dimension=max_bond_dimension,
        )

    def expectation(self, observable: Union[str, SymbolicHamiltonian]) -> float:
        """
        Compute the expectation value of an observable with respect
        to the full ansatz circuit (no partitioning).
        """

        if isinstance(observable, SymbolicHamiltonian):
            coeffs, pauli_terms, sites = observable.simple_terms
            return self._expectation_from_symbolic_hamiltonian(
                coefficients_list=coeffs,
                operators_list=pauli_terms,
                sites_list=sites,
                constant=observable.constant.real,
            )

        elif isinstance(observable, str):
            return self.expectation_from_partition(
                observable, replacement_probability=0.0
            )[0]

        else:
            raise ValueError(
                f"Given observable of type {type(observable)}, but only list or Qibo's SymbolicHamiltonian are supported"
            )

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    def expectation_from_partition(
        self,
        observable: Union[str, SymbolicHamiltonian],
        replacement_probability: float,
        replacement_method: str = "closest",
        return_partitions: bool = False,
    ):
        """
        Sample a lower-magic circuit from the ansatz, and compute its expectation value w.r.t. the observable.
        """

        # Reset MPS to initial state
        self._init_tn(self.max_bond_dimension)

        # Partitionate circuit
        (magic_gates, clifford_circuit), full_circuit = (
            self.ansatz.partitionate_circuit(
                replacement_probability=replacement_probability,
                replacement_method=replacement_method,
            )
        )

        # Apply pauli rotations (generated from dropped magic gates) on the MPS
        for k, magic_gate in magic_gates:

            clifford_subcircuit = self._clifford_subcircuit(clifford_circuit, k)
            generator = self._conjugate_generator(magic_gate, clifford_subcircuit)

            self.tn_engine.pauli_rot(
                state_circuit=self.mps,
                generator=generator,
                angle=magic_gate.parameters[0],
                max_bond_dimension=self.max_bond_dimension,
            )

        # Compute the conjugate of the observable via the stabilizer engine
        new_observable = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=clifford_circuit
        )

        # Collect partitions into a dictionary in case we want to return it
        if return_partitions:
            partitions = {
                "magic_gates": magic_gates,
                "only_cliffords": clifford_circuit,
                "full_circuit": full_circuit,
            }
        else:
            partitions = None

        # mpo is created through the TN engine, and expectation value is computed via the TN engine as well.
        mpo = self.tn_engine.pauli_mpo(new_observable)
        return self.tn_engine.expval(state_circuit=self.mps, operator=mpo), partitions

    def _conjugate_generator(self, gate, clifford_circuit):
        """Conjugate a given gate generator by a sequence of Clifford circuits."""

        if gate.name not in ["rx", "ry", "rz"]:
            raise ValueError("mpstab currently supports only rotational gates.")

        generator = "".join(
            [
                gate2generator[gate.name] if q in gate.target_qubits else "I"
                for q in range(self.nqubits)
            ]
        )
        return self.stab_engine.backpropagate(generator, clifford_circuit)

    def set_engines(
        self,
        stab_engine: StabilizersEngine | None = None,
        tn_engine: TensorNetworkEngine | None = None,
    ):
        """
        Set both stabilizers and tensor-network engines.

        - stab_engine: instance of StabilizersEngine (if None, StimEngine is used)
        - tn_engine: instance of TensorNetworkEngine (if None, NativeTensorNetworkEngine is used)
        """

        # ---- stabilizers engine (existing behaviour) ----
        if stab_engine is None:
            stab_engine = StimEngine()

        if not isinstance(stab_engine, StabilizersEngine):
            raise ValueError(
                f"Provided stabilizers engine {stab_engine} is not supported."
            )

        self.stab_engine = stab_engine

        # ---- tensor-network engine (new) ----
        if tn_engine is None:
            tn_engine = QuimbEngine()

        if not isinstance(tn_engine, TensorNetworkEngine):
            raise ValueError(
                f"Provided tensor-network engine {tn_engine} is not supported."
            )

        self.tn_engine = tn_engine
        self._init_tn(max_bond_dimension=self.max_bond_dimension)

    def _clifford_subcircuit(self, clifford_circuit: Circuit, k: int = 0) -> Circuit:
        """Return a sub-circuit of a given Clifford circuit, cut at index `k`."""
        cut_queue = (
            clifford_circuit.queue[:k] if k is not None else clifford_circuit.queue
        )

        clifford_subcircuit = Circuit(clifford_circuit.nqubits)
        for gate in cut_queue:
            clifford_subcircuit.add(gate)

        return clifford_subcircuit

    def _expectation_from_symbolic_hamiltonian(
        self,
        coefficients_list: list,
        operators_list: list,
        sites_list: list,
        constant: float = None,
        nqubits: int = None,
    ) -> float:
        """
        Compute the expectation value of a Qibo SymbolicHamiltonian.

        Args:
            hamiltonian (SymbolicHamiltonian): a Qibo Hamiltonian object.

        Returns:
            float: The total expectation value, computed as sum of single contributions.
        """

        if nqubits is None:
            nqubits = self.nqubits

        total_expval = constant if constant is not None else 0

        # Computing the contributions
        for coeff, p_name, targets in zip(
            coefficients_list, operators_list, sites_list
        ):

            # For now mpstab requires padding with identities
            full_pauli_list = ["I"] * self.ansatz.circuit.nqubits

            # Fill in the specific Pauli operators at the correct positions
            for i, qubit_idx in enumerate(targets):
                full_pauli_list[qubit_idx] = p_name[i]

            full_pauli_string = "".join(full_pauli_list)

            # Contraction for the Hamiltonian term
            # Each expectation is re-initialising the network to avoid errors
            # TODO: check the performance
            term_expval = self.expectation(full_pauli_string)

            # Accumulate (handle complex coefficients if necessary, though usually real for H)
            total_expval += coeff.real * term_expval

        return total_expval
