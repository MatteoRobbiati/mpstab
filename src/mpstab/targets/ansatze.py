import random
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np
from qibo import Circuit, gates
from qibo.noise import NoiseModel

from mpstab.targets.utils import hardware_compatible_circuit, replace_non_clifford_gate


@dataclass
class Ansatz(ABC):
    """Abstract ansatz to generate quantum states."""

    nqubits: int
    density_matrix: bool = False

    def __post_init__(self):
        self._circuit = Circuit(
            nqubits=self.nqubits, density_matrix=self.density_matrix
        )
        self.noise_model = None
        self.noisy_circuit = None

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        if not isinstance(value, Circuit):
            raise TypeError("Expected a Circuit instance")
        self._circuit = value

    @property
    def nparams(self):
        return len(self.circuit.get_parameters())

    def execute(
        self,
        nshots: int = None,
        initial_state: Circuit = None,
        with_noise: bool = False,
    ):
        """Execute the circuit and return the outcome."""
        # Default empty initial circuit
        if initial_state is None:
            initial_state = Circuit(self.nqubits, density_matrix=self.density_matrix)

        if with_noise:
            if self.noisy_circuit is None:
                raise ValueError(
                    f"Before asking for noisy simulation, ensure the noise model is set via the `update_noise_model` method."
                )
            if len(initial_state.queue) != 0:
                initial_state.density_matrix = True
                initial_state = self.noise_model.apply(initial_state)
            result = (initial_state + self.noisy_circuit)(nshots=nshots)
        else:
            result = (initial_state + self.circuit)(nshots=nshots)
        return result

    def update_noise_model(self, noise_model: NoiseModel):
        """Construct an attribute which is the noisy version of circuit."""
        self.noise_model = noise_model
        self.noisy_circuit = noise_model.apply(self.circuit)

    def partitionate_circuit(
        self, replacement_probability: float, replacement_method: str
    ):
        """
        Partitionate the circuit replacing non-Clifford (magic) gates with a given probability.

        For each gate in the original circuit:
        - If the gate is non-Clifford, with probability replacement_probability it is
            replaced by a Clifford gate via replace_non_clifford_gate.
        - The processed gate (whether replaced or not) is added to the overall circuit.
        - Simultaneously, consecutive gates of the same type (Clifford or non-Clifford)
            are collected into blocks.

        Returns:
            partitioned_circuit (Circuit): the complete processed circuit.
            clifford_blocks (List[Circuit]): list of circuits, each a block of consecutive Clifford gates.
            non_clifford_blocks (List[Circuit]): list of circuits, each a block of consecutive non-Clifford gates.
        """
        magic_gates = []
        clifford_only_circuit = Circuit(
            nqubits=self.nqubits, density_matrix=self.density_matrix
        )
        full_circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)

        break_point = 0
        for gate in self.circuit.queue:

            if not gate.clifford and not isinstance(gate, gates.M):
                r = random.random()
                if r > replacement_probability:
                    magic_gates.append((break_point, gate))
                    full_circuit.add(gate)
                    continue

                gate = replace_non_clifford_gate(
                    gate, replacement_method=replacement_method
                )

            break_point += 1
            clifford_only_circuit.add(gate)
            full_circuit.add(gate)

        return (magic_gates, clifford_only_circuit), full_circuit


@dataclass
class HardwareEfficient(Ansatz):
    """Hardware Efficient ansatz."""

    nlayers: int = 1
    entangling: bool = True

    def __post_init__(self):
        super().__post_init__()
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            if self.entangling:
                self.circuit += self.entanglement_layer()
        # self.circuit.add(gates.M(*range(self.nqubits)))

    @property
    def parameters_per_layers(self):
        return int(self.nparams / self.nlayers)

    def parametric_layer(self, layer_index: int):
        """Return the gates composing a parametric layer."""
        # Start and end index for the layer in the circuit
        # Count as start the number of parametric gates per layer + the entanglement layer
        start_index = (
            layer_index * self.parameters_per_layers + layer_index * self.nqubits
        )
        end_index = start_index + self.parameters_per_layers
        return deepcopy(self.circuit.queue[start_index:end_index])

    def entanglement_layer(self):
        """Construct an entanglement layer compatible with the target quantum circuit."""
        ent_circuit = Circuit(self.nqubits, density_matrix=self.density_matrix)
        [
            ent_circuit.add(gates.CZ(q0=q % self.nqubits, q1=(q + 1) % self.nqubits))
            for q in range(self.nqubits)
        ]
        return ent_circuit


@dataclass(kw_only=True)
class TranspiledAnsatz(Ansatz):
    """
    Any ansatz which is also transpiled into native gates of a given quantum device
    presenting a given connectivity.

    Args:
        original_circuit: The circuit to be transpiled.
        native_gates: Optional[List]: list of native gates of the used device.
            Default is [gates.GPI2, gates.RZ, gates.Z, gates.CZ].
        connectivity: Optional[nx.Graph]: graph representing the topology of the
            used device. Default is None and in this case the transpilation
            does not take into account any connectivity constraint.
    """

    original_circuit: Circuit
    native_gates: Optional[List] = field(
        default_factory=lambda: [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    )
    connectivity: Optional[nx.Graph] = None
    # Override nqubits so it is not passed in __init__
    nqubits: int = field(init=False)

    def __post_init__(self):
        # Set nqubits from the provided circuit.
        self.nqubits = self.original_circuit.nqubits
        # Now call the parent's __post_init__ to initialize _circuit and other attributes.
        super().__post_init__()
        # Overwrite the circuit with the original one.
        self._circuit = hardware_compatible_circuit(self.original_circuit)

        # Freeze the GPI2 gates
        for g in self._circuit.parametrized_gates:
            if isinstance(g, gates.GPI2) and g.clifford:
                g.trainable = False

    @property
    def circuit(self):
        return self._circuit


@dataclass
class FloquetAnsatz(Ansatz):
    """
    Floquet echo: U = (FL)^t · Rz(theta) on qubit q · (FL^t)†.

    Args:
        nlayers (int): number of Floquet layers.
        b (float): parameter controlling the magic in the circuit;
        theta (float): controls the signal;
        target_qubit (int): this is the qubit, in the circuit, on which we perform
            the final local measurement (and the one on which we apply H and RZ).
        decompose_rzz (bool): if ``True``, RZZ is decomposed into CNOTs and RZ.
    """

    nlayers: int = 2
    b: float = 0.4 * np.pi
    theta: float = 0.5 * np.pi
    target_qubit: int = 1
    decompose_rzz: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Save attribute for replacements
        self.half_sandwich = Circuit(self.nqubits)
        for _ in range(self.nlayers):
            self.half_sandwich += self._build_floquet_layer()

        # Len of the first half of the sandwich + 1 Hadamard
        # # These attributes will be useful later for partitioning
        self.rz_index = len(self.half_sandwich.queue) + 1
        self.start_inverted_half = self.rz_index + 1

        # First, we add an Hadamard to the target qubit
        self.circuit.add(gates.H(self.target_qubit))
        # Then append nlayers Floquet layers
        self.circuit += self.half_sandwich
        # Add RZ
        self.circuit.add(gates.RZ(q=self.target_qubit, theta=self.theta))
        # Add the inverted Floquet layers
        self.circuit += self.half_sandwich.invert()

    def _build_floquet_layer(self) -> Circuit:
        """Construct one Floquet layer over all links."""
        layer = Circuit(self.nqubits, density_matrix=self.density_matrix)
        layer += self._build_sublayer("even")
        layer += self._build_sublayer("odd")
        return layer

    def _build_sublayer(self, parity: str):
        """Helper function to build a sub-layer composing a Floquet layer."""
        layer = Circuit(self.nqubits, density_matrix=self.density_matrix)

        if parity == "even":
            qubits = range(0, self.nqubits - 1, 2)
        elif parity == "odd":
            qubits = range(1, self.nqubits - 1, 2)
        else:
            raise ValueError(f"Please set `parity` to be 'odd' or 'even'.")

        for q1 in qubits:
            q2 = q1 + 1
            layer.add(gates.RZ(q=q1, theta=0.25 * np.pi))
            layer.add(gates.RX(q=q1, theta=self.b))
            layer.add(gates.RZ(q=q2, theta=0.25 * np.pi))
            layer.add(gates.RX(q=q2, theta=self.b))
            if not self.decompose_rzz:
                layer.add(gates.RZZ(q0=q1, q1=q2, theta=0.5 * np.pi))
            else:
                layer += self._decomposed_rzz(q0=q1, q1=q2, theta=0.5 * np.pi)

        return layer

    def _decomposed_rzz(self, q0, q1, theta):
        layer = Circuit(self.nqubits, density_matrix=self.density_matrix)
        layer.add(gates.CNOT(q0, q1))
        layer.add(gates.RZ(q=q1, theta=theta))
        layer.add(gates.CNOT(q0, q1))
        return layer

    def partitionate_sub_circuit(
        self, circuit: Circuit, replacement_probability: float, replacement_method: str
    ):
        """
        Partitionate a *sub*-circuit replacing non-Clifford (magic) gates with a given probability.
        Returns ((magic_gates, clifford_only_circuit), full_circuit) where
        magic_gates is List of (local_bp, gate) with local_bp starting at 0.
        """
        magic_gates = []
        clifford_only_circuit = Circuit(
            self.nqubits, density_matrix=self.density_matrix
        )
        full_circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)

        break_point = 0
        for gate in circuit.queue:
            if not gate.clifford and not isinstance(gate, gates.M):
                if random.random() > replacement_probability:
                    magic_gates.append((break_point, gate))
                    full_circuit.add(gate)
                    continue
                gate = replace_non_clifford_gate(
                    gate, replacement_method=replacement_method
                )

            break_point += 1
            clifford_only_circuit.add(deepcopy(gate))
            full_circuit.add(deepcopy(gate))

        return (magic_gates, clifford_only_circuit), full_circuit

    def partitionate_circuit(
        self, replacement_probability: float, replacement_method: str
    ):
        """
        Partitionate the full Floquet circuit U = H · half_sandwich · RZ · (half_sandwich)†,
        but only *sweep* the half_sandwich for magic gates.
        Returns ((magic_gates, clifford_only_circuit), full_circuit).
        """
        magic_gates = []
        clifford_only_circuit = Circuit(
            self.nqubits, density_matrix=self.density_matrix
        )
        full_circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)

        # 1) initial H
        H = gates.H(self.target_qubit)
        full_circuit.add(deepcopy(H))
        clifford_only_circuit.add(deepcopy(H))

        # 2) first half: collect both full_circuit_1 and its magic break-points
        (magic_gates_1, clifford_block_1), full_circuit_1 = (
            self.partitionate_sub_circuit(
                deepcopy(self.half_sandwich),
                replacement_probability,
                replacement_method,
            )
        )

        # shift each local_bp by +1 to account for the leading H at index 0
        magic_gates.extend([(bp + 1, gate) for bp, gate in magic_gates_1])
        clifford_only_circuit += clifford_block_1
        full_circuit += full_circuit_1

        print("theta: ", self.theta)

        # 3) the central RZ
        rz = gates.RZ(q=self.target_qubit, theta=self.theta)
        full_circuit.add(deepcopy(rz))
        if rz.clifford:
            clifford_only_circuit.add(deepcopy(rz))
        else:
            # its position is len(half_sandwich) + 1
            magic_gates.append((len(clifford_block_1.queue), deepcopy(rz)))

        # 4) inverted half + mirrored magic gates
        for bp, gate in magic_gates_1[::-1]:
            # mirror the *shifted* index
            magic_gates.append(
                (
                    len(clifford_block_1.queue) - bp + len(clifford_only_circuit.queue),
                    deepcopy(gate).dagger(),
                )
            )

        clifford_only_circuit += clifford_block_1.invert()
        full_circuit += full_circuit_1.invert()

        return (magic_gates, clifford_only_circuit), full_circuit
