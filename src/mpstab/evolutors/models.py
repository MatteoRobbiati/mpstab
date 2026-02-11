
from dataclasses import dataclass

import numpy as np
from qibo import Circuit

from mpstab.backends.stabilizers.abstract import StabilizersEngine
from mpstab.backends.stabilizers.stim import StimEngine
from mpstab.backends.tensor_networks.abstract import TensorNetworkEngine
from mpstab.backends.tensor_networks.native import NativeTensorNetworkEngine
from mpstab.backends.tensor_networks.quimb import QuimbEngine
from mpstab.evolutors.stabilizer import tableaus
from mpstab.evolutors.stabilizer.pauli_string import Pauli
from mpstab.evolutors.utils import gate2generator, gate2tableau
from mpstab.models.ansatze import Ansatz


@dataclass
class HybridSurrogate:
    """
    Construct an hybrid stabilizer MPO surrogate of a given quantum circuit.

    The tensor-network part is now engine-pluggable via the TensorNetworkEngine API.
    """

    ansatz: Ansatz
    max_bond_dimension: int = None
    initial_state: Circuit = None

    def __post_init__(self):
        # Initial state is zero by default
        if self.initial_state is None:
            self.initial_state = Circuit(self.ansatz.nqubits)

        # Default engines will be set here (stabilizers + tensor-network)
        # and TN will be initialised by _init_tn.
        self.set_engine()

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
            max_bond_dimension=max_bond_dimension
        )

    def expectation(self, observable: str) -> float:
        """
        Compute the expectation value of an observable with respect to the full ansatz circuit (no partitioning).
        """
        # Reset MPS to initial state
        self._init_tn(self.max_bond_dimension)

        expval = self.expectation_from_partition(observable, replacement_probability=0.0)[0]

        return expval

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    def expectation_from_partition(
        self,
        observable: str,
        replacement_probability: float,
        replacement_method: str = "closest",
        return_partitions: bool = False,
    ):
        """
        Sample a lower-magic circuit from the ansatz, and compute its expectation value w.r.t. the observable.
        """

        # Partitionate circuit
        (magic_gates, clifford_circuit), full_circuit = (
            self.ansatz.partitionate_circuit(
                replacement_probability=replacement_probability,
                replacement_method=replacement_method,
            )
        )

        # Apply pauli rotations (generated from dropped magic gates) on the MPS
        for k, magic_gate in magic_gates:

            generator = self._conjugate_generator(magic_gate, clifford_circuit, k)
            self.mps = self.tn_engine.pauli_rot(state_circuit=self.mps, generator=generator, angle=magic_gate.parameters[0])

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
        return self.tn_engine.expval(state_circuit = self.mps, operator = mpo), partitions

    def _conjugate_generator(self, gate, clifford_circuit, k):
        """Conjugate a given gate generator by a sequence of Clifford circuits."""

        if gate.name not in ["rx", "ry", "rz"]:
            raise ValueError("mpstab currently supports only rotational gates.")

        generator = "".join(
            [
                gate2generator[gate.name] if q in gate.target_qubits else "I"
                for q in range(self.nqubits)
            ]
        )
        return self.backpropagate_pauli(generator, clifford_circuit, k)

    def set_engine(
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
            raise ValueError(f"Provided stabilizers engine {stab_engine} is not supported.")

        self.stab_engine = stab_engine

        # ---- tensor-network engine (new) ----
        if tn_engine is None:
            tn_engine = QuimbEngine()

        if not isinstance(tn_engine, TensorNetworkEngine):
            raise ValueError(f"Provided tensor-network engine {tn_engine} is not supported.")

        self.tn_engine = tn_engine