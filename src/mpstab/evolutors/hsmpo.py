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
from mpstab.engines.tensor_networks.quimb import _qibo_circuit_to_quimb
from mpstab.evolutors.optimization import (
    minimize_expectation_dmrg as _minimize_expectation_dmrg,
)
from mpstab.evolutors.utils import gate2generator, validate_pauli_observable
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

        # Precompute MPS evolved once with replacement_probability=0
        # This is cached for reuse without reconstruction
        # Also caches magic_gates and clifford_circuit during the precomputation
        self.original_circuit_mps = self._precompute_original_mps()

    def _precompute_original_mps(self):
        """
        Precompute the MPS evolved once using the magic/clifford decomposition
        with replacement_probability=0.

        Also caches magic_gates and clifford_circuit for later use in expectation
        evaluations, avoiding redundant circuit partitioning.

        Returns the evolved MPS for caching.
        """
        import copy

        # Start from a clean copy of the initial MPS
        self._init_tn(self.max_bond_dimension)
        evolved_mps = copy.deepcopy(self.mps)

        # Get magic gates and clifford structure with no replacements
        # (cache these for later use)
        (self.magic_gates, self.clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=0.0,
            replacement_method="closest",
        )

        # Apply each magic gate evolution
        for k, magic_gate in self.magic_gates:
            clifford_subcircuit = self._clifford_subcircuit(self.clifford_circuit, k)
            generator, sign = self._conjugate_generator(magic_gate, clifford_subcircuit)

            # Apply pauli rotation to the evolved MPS
            self.tn_engine.pauli_rot(
                state_circuit=evolved_mps,
                generator=generator,
                angle=magic_gate.parameters[0] * sign,
                max_bond_dimension=self.max_bond_dimension,
            )

        return evolved_mps

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

    def expectation(
        self, observable: Union[str, SymbolicHamiltonian], return_fidelity: bool = False
    ):
        """
        Compute the expectation value of an observable with respect
        to the full ansatz circuit (no partitioning).
        """

        if isinstance(observable, SymbolicHamiltonian):
            if return_fidelity:
                return (
                    self._expectation_from_symbolic_hamiltonian(hamiltonian=observable),
                    self.tn_engine.norm,
                )
            else:
                return self._expectation_from_symbolic_hamiltonian(
                    hamiltonian=observable
                )

        elif isinstance(observable, str):
            # Validate observable string format and length
            validate_pauli_observable(observable, self.nqubits)

            # Use cached clifford circuit for observable backpropagation
            # (computed once during MPS initialization)
            backprop_observable, sign = self.stab_engine.backpropagate(
                observable=observable, clifford_circuit=self.clifford_circuit
            )

            mpo = self.tn_engine.pauli_mpo(backprop_observable)
            expval = self.tn_engine.expval(
                state_circuit=self.original_circuit_mps, operator=mpo
            )
            expval = float(expval) * sign

            if return_fidelity:
                return (
                    expval,
                    self.original_circuit_mps.norm(squared=True),
                )
            else:
                return expval
        else:
            raise ValueError(
                f"Given observable of type {type(observable)}, but only list or Qibo's SymbolicHamiltonian are supported"
            )

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    @property
    def nparams(self):
        return self.ansatz.nparams

    @property
    def truncation_fidelity_pure_tn(self) -> float:
        return _qibo_circuit_to_quimb(
            nqubits=self.nqubits,
            qibo_circ=self.initial_state + self.ansatz.circuit,
            max_bond=self.max_bond_dimension,
        ).fidelity_estimate()

    def truncation_fidelity(
        self,
        replacement_probability: float = 0.0,
        replacement_method: str = "closest",
    ) -> float:
        """
        Truncation fidelity between truncated and original state :math:`|\\langle\\Psi_t|\\Psi_t\rangle|^2/|\\langle\\Psi|\\Psi\rangle|^2 = |\\langle\\Psi_t|\\Psi_t\rangle|^2`, being Ψ normalized.
        """

        # Reset MPS to initial state
        self._init_tn(self.max_bond_dimension)

        # Partitionate circuit
        (magic_gates, clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=replacement_probability,
            replacement_method=replacement_method,
        )

        # Apply pauli rotations (generated from dropped magic gates) on the MPS
        for k, magic_gate in magic_gates:

            clifford_subcircuit = self._clifford_subcircuit(clifford_circuit, k)
            generator, sign = self._conjugate_generator(magic_gate, clifford_subcircuit)

            self.tn_engine.pauli_rot(
                state_circuit=self.mps,
                generator=generator,
                angle=magic_gate.parameters[0] * sign,
                max_bond_dimension=self.max_bond_dimension,
            )

        return self.mps.norm(squared=True)

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
            generator, sign = self._conjugate_generator(magic_gate, clifford_subcircuit)

            self.tn_engine.pauli_rot(
                state_circuit=self.mps,
                generator=generator,
                angle=magic_gate.parameters[0] * sign,
                max_bond_dimension=self.max_bond_dimension,
            )

        # Compute the conjugate of the observable via the stabilizer engine
        new_observable, sign = self.stab_engine.backpropagate(
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
        return (
            self.tn_engine.expval(state_circuit=self.mps, operator=mpo) * sign,
            partitions,
        )

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
        self, hamiltonian: SymbolicHamiltonian
    ) -> float:
        """
        Compute the expectation value of a Qibo SymbolicHamiltonian.

        Args:
            hamiltonian (SymbolicHamiltonian): a Qibo Hamiltonian object.

        Returns:
            float: The total expectation value, computed as sum of single contributions.
        """

        # Leveraging Qibo's features
        coeffs, pauli_names, target_qubits = hamiltonian.simple_terms
        constant = hamiltonian.constant.real

        total_expval = constant

        # Computing the contributions using the precomputed evolved MPS
        # and cached clifford circuit
        for coeff, p_name, targets in zip(coeffs, pauli_names, target_qubits):

            # For now mpstab requires padding with identities
            full_pauli_list = ["I"] * hamiltonian.nqubits

            # Fill in the specific Pauli operators at the correct positions
            for i, qubit_idx in enumerate(targets):
                full_pauli_list[qubit_idx] = p_name[i]

            full_pauli_string = "".join(full_pauli_list)

            # Backpropagate the specific term through the cached clifford circuit
            new_observable, sign = self.stab_engine.backpropagate(
                observable=full_pauli_string, clifford_circuit=self.clifford_circuit
            )

            # Contraction for the Hamiltonian term using the precomputed evolved MPS
            mpo = self.tn_engine.pauli_mpo(new_observable)
            term_expval = self.tn_engine.expval(
                state_circuit=self.original_circuit_mps, operator=mpo
            )

            total_expval += coeff.real * term_expval * sign

        return total_expval

    def set_parameters(self, parameters: np.ndarray):
        """
        Set circuit parameters and automatically re-precompute the MPS cache.

        This ensures the MPS state always matches the current circuit parameters.

        Args:
            parameters: Array of circuit parameter values.
        """
        self.ansatz.circuit.set_parameters(parameters)
        # Re-precompute MPS with updated parameters
        self.original_circuit_mps = self._precompute_original_mps()

    def get_parameters(self) -> np.ndarray:
        """Get current circuit parameters."""
        return self.ansatz.circuit.get_parameters()

    def minimize_expectation(
        self,
        observables: Union[str, list, dict, SymbolicHamiltonian],
        method: str = "dmrg",
        bond_dims: Union[int, list] = None,
        cutoff: float = 1e-9,
        tol: float = 1e-6,
        max_sweeps: int = 10,
        verbosity: int = 1,
    ):
        """
        Minimize expectation value(s) using the specified method.

        This delegates to the optimization module (defaults to DMRG, much more
        efficient than circuit-based VQE).

        Args:
            observables: Hamiltonian to minimize. Can be:
                - str: Single observable (e.g., "ZZZZ")
                - list of str: Multiple observables (e.g., ["ZZZZ", "XXXX"])
                - dict: Observable -> coefficient (e.g., {"ZZZZ": 1.0, "XXXX": 0.5})
                - SymbolicHamiltonian: Qibo Hamiltonian object
            method: 'dmrg' (default, recommended), or 'circuit' for gradient-free VQE
            bond_dims: Max bond dimensions (DMRG only)
            cutoff: SVD truncation (DMRG only)
            tol: Energy convergence tolerance (DMRG only)
            max_sweeps: Maximum sweeps (DMRG only)
            verbosity: Verbosity level

        Returns:
            dict with 'ground_state', 'energy', 'converged', etc.
        """
        if method.lower() == "dmrg":
            return _minimize_expectation_dmrg(
                self,
                observables=observables,
                bond_dims=bond_dims,
                cutoff=cutoff,
                tol=tol,
                max_sweeps=max_sweeps,
                verbosity=verbosity,
            )
        elif method.lower() == "circuit":
            raise NotImplementedError(
                "Circuit-based VQE removed. Use DMRG instead via method='dmrg'"
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dmrg' or 'circuit'")
