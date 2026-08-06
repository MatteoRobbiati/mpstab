"""Circuit templates to simulate: variational patterns and textbook algorithms."""

import random
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import numpy as np
from qibo import Circuit, gates
from qibo.models import QFT as qibo_qft
from qibo.noise import NoiseModel

from mpstab.models.utils import hardware_compatible_circuit, replace_non_clifford_gate


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
        Split the circuit into a Clifford part and the magic gates left over.

        Each non-Clifford gate is replaced by the nearest Clifford one with
        probability ``replacement_probability``, and kept as a magic gate
        otherwise. Every magic gate is recorded with the number of Clifford gates
        preceding it, which is where it has to be reinserted later.

        Args:
            replacement_probability: chance of replacing each non-Clifford gate.
            replacement_method: ``"closest"`` or ``"random"`` Clifford angle.

        Returns:
            ``((magic_gates, clifford_circuit), full_circuit)``, where
            ``magic_gates`` is a list of ``(breakpoint_index, gate)`` and
            ``full_circuit`` is the sampled circuit including its magic gates.
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


@dataclass
class HardwareEfficientBrickwork(Ansatz):
    """
    Hardware Efficient ansatz with a full brickwork entanglement pattern.

    Each layer applies two sublayers of single-qubit RY+RZ rotations interleaved
    with CZ entanglement:

        sublayer 1: RY+RZ on all qubits → CZ on even bonds (0-1, 2-3, …)
        sublayer 2: RY+RZ on all qubits → CZ on odd bonds (1-2, 3-4, …) + CZ(0, n-1)

    A final layer of RY rotations closes the circuit.  The ring-closing CZ(0, n-1)
    in the odd sublayer creates long-range entanglement.

    Args:
        nqubits: Number of qubits.
        nlayers: Number of full brickwork layers.
    """

    nlayers: int = 1

    def __post_init__(self):
        super().__post_init__()
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(self.nqubits):
                self.circuit.add(gates.RZ(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(0, self.nqubits - 1, 2):
                self.circuit.add(gates.CZ(q0=q, q1=q + 1))
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(self.nqubits):
                self.circuit.add(gates.RZ(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(1, self.nqubits - 2, 2):
                self.circuit.add(gates.CZ(q0=q, q1=q + 1))
            self.circuit.add(gates.CZ(q0=0, q1=self.nqubits - 1))
        for q in range(self.nqubits):
            self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))


@dataclass
class HammingWeightPreserving(Ansatz):
    """
    Hamming-weight-preserving ansatz built from decomposed RBS gates.

    Each RBS gate rotates within the {|01⟩, |10⟩} subspace of a qubit pair,
    leaving |00⟩ and |11⟩ unchanged:

        RBS(θ)|01⟩ =  cos(θ)|01⟩ + sin(θ)|10⟩
        RBS(θ)|10⟩ = -sin(θ)|01⟩ + cos(θ)|10⟩

    RBS is decomposed into gates mpstab supports natively:

        CNOT(ctrl=q1, tgt=q0)
        RY(q1,  θ)              ← magic gate
        CNOT(ctrl=q0, tgt=q1)
        RY(q1, -θ)              ← magic gate
        CNOT(ctrl=q0, tgt=q1)
        CNOT(ctrl=q1, tgt=q0)

    The circuit always starts with n//2 X gates to prepare the half-filling
    state |1…10…0⟩, giving a fixed Hamming weight of n//2.

    Within each layer the connectivity pattern mirrors `hw_preserving` from
    Qibo examples: four `connect_qubits` passes with jump sizes 1 and 2 and
    different starting offsets to explore both nearest- and next-nearest-
    neighbour correlations with circular boundary conditions.

    Args:
        nqubits: Number of qubits (must be even).
        nlayers: Number of RBS layers.
    """

    nlayers: int = 1

    def __post_init__(self):
        if self.nqubits % 2 != 0:
            raise ValueError(
                "HammingWeightPreserving requires an even number of qubits."
            )
        super().__post_init__()
        for q in range(self.nqubits // 2):
            self.circuit.add(gates.X(q=q))
        for _ in range(self.nlayers):
            self._connect_qubits(jumpsize=1, start_from=0)
            self._connect_qubits(jumpsize=1, start_from=1)
            self._connect_qubits(jumpsize=2, start_from=0)
            self._connect_qubits(jumpsize=2, start_from=1)
            self._connect_qubits(jumpsize=2, start_from=3)

    def _connect_qubits(self, jumpsize: int, start_from: int) -> None:
        """Add RBS gates between qubit pairs (q, (q+jumpsize) % nqubits)."""
        for q in range(start_from, self.nqubits, jumpsize + 1):
            q0 = q
            q1 = (q + jumpsize) % self.nqubits
            theta = np.random.uniform(-np.pi, np.pi)
            self.circuit += self._rbs_decomposition(q0, q1, theta)

    def _rbs_decomposition(self, q0: int, q1: int, theta: float) -> Circuit:
        """
        Decompose RBS(theta) on qubits (q0, q1) into CNOT and RY gates.

        Equal to CNOT(q1→q0) · CRY(ctrl=q0, tgt=q1, 2θ) · CNOT(q1→q0),
        with CRY expanded as RY(θ)·CNOT·RY(-θ)·CNOT.  Verified analytically
        on all four computational basis states.
        """
        circ = Circuit(self.nqubits, density_matrix=self.density_matrix)
        circ.add(gates.CNOT(q1, q0))
        circ.add(gates.RY(q=q1, theta=theta))
        circ.add(gates.CNOT(q0, q1))
        circ.add(gates.RY(q=q1, theta=-theta))
        circ.add(gates.CNOT(q0, q1))
        circ.add(gates.CNOT(q1, q0))
        return circ


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


@dataclass(kw_only=True)
class CircuitAnsatz(Ansatz):
    """
    A simple wrapper that converts a Qibo Circuit into an Ansatz object.

    Args:
        input_circuit (Circuit): The Qibo circuit to wrap.
    """

    qibo_circuit: Circuit
    # We set nqubits as init=False because we derive it from input_circuit
    nqubits: int = field(init=False)

    def __post_init__(self):
        self.nqubits = self.qibo_circuit.nqubits
        super().__post_init__()
        self._circuit = deepcopy(self.qibo_circuit)

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, value):
        if not isinstance(value, Circuit):
            raise TypeError("Expected a Qibo Circuit instance")
        self._circuit = value
        self.nqubits = value.nqubits


# ---------------------------------------------------------------------------
# Circuit-library ansatze: textbook algorithms (QFT, Grover, QPE, QAE, Trotter)
# to benchmark HSMPO and HSynthSMPO against.
#
# They are written with high-level gates (CU1, SWAP, TOFFOLI, RZZ, ...) and
# transpiled to the native gate set on construction, which turns every
# non-Clifford gate into an `rz` rotation -- the form HSMPO understands, so
# nothing needs hand-decomposing. Pass ``transpile=False`` to keep the raw
# high-level circuit for inspection; HSMPO itself needs the transpiled form.
# ---------------------------------------------------------------------------
def _qft_circuit(n: int) -> Circuit:
    """QFT sub-circuit using H, controlled-phase (CU1) and SWAP (no transpilation)."""
    return qibo_qft(nqubits=n)


def _multi_controlled_z(circuit: Circuit, controls: List[int], target: int) -> None:
    """
    Append a multi-controlled-Z built from CZ and TOFFOLI, so that it unrolls.

    Exact for up to two controls (CZ, and CCZ = H.CCX.H). Beyond that it chains
    pairwise Toffolis sharing the target, which is a benchmark-oriented
    approximation rather than a faithful multi-controlled-Z: do not rely on
    Grover oracle correctness above 3 qubits.
    """
    n_ctrl = len(controls)
    if n_ctrl == 0:
        circuit.add(gates.Z(target))
    elif n_ctrl == 1:
        circuit.add(gates.CZ(controls[0], target))
    elif n_ctrl == 2:
        circuit.add(gates.H(target))
        circuit.add(gates.TOFFOLI(controls[0], controls[1], target))
        circuit.add(gates.H(target))
    else:
        circuit.add(gates.H(target))
        for c in controls[:-1]:
            circuit.add(gates.TOFFOLI(c, controls[-1], target))
        circuit.add(gates.H(target))


@dataclass(kw_only=True)
class _CompiledAnsatz(Ansatz):
    """
    Base for circuit-library ansatze that build a high-level circuit and (by
    default) transpile it to the native gate set so HSMPO can consume it.

    Subclasses implement :meth:`_build_circuit`.

    Args:
        transpile: If ``True`` (default), unroll the built circuit into
            ``native_gates`` (Clifford GPI2 gates are frozen, as in
            :class:`TranspiledAnsatz`). If ``False``, keep the raw high-level
            circuit.
        native_gates: Target native gate set for transpilation.
        connectivity: Optional device connectivity graph for transpilation.
    """

    transpile: bool = True
    native_gates: Optional[List] = field(
        default_factory=lambda: [gates.GPI2, gates.RZ, gates.Z, gates.CZ]
    )
    connectivity: Optional[nx.Graph] = None

    def __post_init__(self):
        super().__post_init__()
        circuit = self._build_circuit()
        if self.transpile:
            circuit = hardware_compatible_circuit(
                circuit, self.native_gates, self.connectivity
            )
            for g in circuit.parametrized_gates:
                if isinstance(g, gates.GPI2) and g.clifford:
                    g.trainable = False
        self._circuit = circuit

    def _build_circuit(self) -> Circuit:
        raise NotImplementedError

    @property
    def circuit(self):
        return self._circuit


@dataclass(kw_only=True)
class QFT(_CompiledAnsatz):
    """Quantum Fourier Transform (H, controlled-phase and SWAP)."""

    def _build_circuit(self) -> Circuit:
        return _qft_circuit(self.nqubits)


@dataclass(kw_only=True)
class QFTPhaseKernel(_CompiledAnsatz):
    """QFT followed by a diagonal RZ phase kernel (no closing inverse QFT).

    The trailing RZ layer's magic rotations get Clifford-backpropagated through
    the QFT's H/CU1/SWAP structure -- a stronger HSMPO test than a single
    trailing CZ chain.

    Args:
        coeffs: Per-qubit RZ angles. Default: binary-weighted ``2*pi/2^(j+1)``.
    """

    coeffs: Optional[np.ndarray] = None

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        coeffs = self.coeffs
        if coeffs is None:
            coeffs = np.array([2 * np.pi / 2 ** (j + 1) for j in range(n)])

        circuit = _qft_circuit(n)
        for j, theta in enumerate(coeffs):
            circuit.add(gates.RZ(j, theta=float(theta)))
        return circuit


@dataclass(kw_only=True)
class Grover(_CompiledAnsatz):
    """Grover search: H^n then n_iterations x (oracle + diffuser).

    The oracle phase-flips ``marked_state`` (default |1...1>) via a
    multi-controlled-Z; the diffuser is the standard inversion-about-the-mean.
    See :func:`_multi_controlled_z` for the oracle-correctness caveat at
    ``nqubits > 3``.

    Args:
        marked_state: Integer in [0, 2^nqubits). Default 2^nqubits - 1.
        n_iterations: Grover iterations. Default round(pi/4 * sqrt(2^nqubits)).
    """

    marked_state: Optional[int] = None
    n_iterations: Optional[int] = None

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        marked_state = self.marked_state
        n_iterations = self.n_iterations
        if marked_state is None:
            marked_state = (1 << n) - 1
        if n_iterations is None:
            n_iterations = max(1, int(round(np.pi / 4 * np.sqrt(2**n))))

        circuit = Circuit(n)
        for q in range(n):
            circuit.add(gates.H(q))

        for _ in range(n_iterations):
            # Oracle: phase-flip on |marked_state>.
            for q in range(n):
                if not (marked_state >> (n - 1 - q)) & 1:
                    circuit.add(gates.X(q))
            _multi_controlled_z(circuit, controls=list(range(n - 1)), target=n - 1)
            for q in range(n):
                if not (marked_state >> (n - 1 - q)) & 1:
                    circuit.add(gates.X(q))

            # Diffuser: H^n (2|0><0| - I) H^n.
            for q in range(n):
                circuit.add(gates.H(q))
                circuit.add(gates.X(q))
            _multi_controlled_z(circuit, controls=list(range(n - 1)), target=n - 1)
            for q in range(n):
                circuit.add(gates.X(q))
                circuit.add(gates.H(q))

        return circuit


@dataclass(kw_only=True)
class QPE(_CompiledAnsatz):
    """Quantum Phase Estimation for U = diag(1, e^{2 pi i phase}).

    Layout: ``n_counting`` counting qubits + 1 eigenstate qubit (prepared in
    |1>). Controlled-U^{2^k} is a controlled phase (CU1); the counting register
    is closed with an inverse QFT. ``nqubits`` is derived as ``n_counting + 1``.

    Args:
        n_counting: Number of counting qubits.
        phase: True phase in [0, 1); accuracy ~ 1/2^n_counting.
    """

    n_counting: int = 3
    phase: float = 0.375
    nqubits: int = field(init=False)

    def __post_init__(self):
        self.nqubits = self.n_counting + 1
        super().__post_init__()

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        eig = self.n_counting

        circuit = Circuit(n)
        circuit.add(gates.X(eig))
        for q in range(self.n_counting):
            circuit.add(gates.H(q))

        for k in range(self.n_counting):
            circuit.add(gates.CU1(k, eig, theta=2 * np.pi * self.phase * (2**k)))

        # Inverse QFT on the counting register (added gate by gate since it acts
        # on a sub-register of the full circuit).
        for gate in _qft_circuit(self.n_counting).invert().queue:
            circuit.add(gate)
        return circuit


@dataclass(kw_only=True)
class QAE(_CompiledAnsatz):
    """Canonical Quantum Amplitude Estimation.

    ``n_counting`` counting qubits + 1 work qubit. State preparation is
    A = R_Y(2*theta_A) on the work qubit (amplitude a = sin^2(theta_A)). The
    controlled Grover powers are modelled by controlled phases (CU1) with
    binary-scaled angles, closed with an inverse QFT on the counting register.
    ``nqubits`` is ``n_counting + 1``.

    Args:
        n_counting: Number of counting qubits.
        theta_A: Angle controlling the amplitude. Default 0.4 rad.
    """

    n_counting: int = 3
    theta_A: float = 0.4
    nqubits: int = field(init=False)

    def __post_init__(self):
        self.nqubits = self.n_counting + 1
        super().__post_init__()

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        work = self.n_counting

        circuit = Circuit(n)
        circuit.add(gates.RY(work, theta=2 * self.theta_A))
        for q in range(self.n_counting):
            circuit.add(gates.H(q))

        for k in range(self.n_counting):
            circuit.add(gates.CU1(k, work, theta=2 * (2**k) * self.theta_A))

        for gate in _qft_circuit(self.n_counting).invert().queue:
            circuit.add(gate)
        return circuit


@dataclass(kw_only=True)
class TrotterIsing(_CompiledAnsatz):
    """First-order Trotter evolution of the 1D transverse-field Ising model.

        H = -J sum_j Z_j Z_{j+1} - h sum_j X_j   (open boundaries)

    Each step applies ZZ rotations (via RZZ) then transverse-field RX rotations,
    producing a long structured sequence of magic rotations between Cliffords.

    Args:
        n_steps: Number of Trotter steps.
        dt: Time step.
        J: Coupling strength.
        h: Transverse field strength.
    """

    n_steps: int = 4
    dt: float = 0.2
    J: float = 1.0
    h: float = 0.5

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        circuit = Circuit(n)
        for _ in range(self.n_steps):
            for j in range(n - 1):
                circuit.add(gates.RZZ(j, j + 1, theta=-2 * self.J * self.dt))
            for j in range(n):
                circuit.add(gates.RX(j, theta=-2 * self.h * self.dt))
        return circuit
