"""The hybrid stabilizer-MPO surrogate of a quantum circuit."""

import copy
from dataclasses import dataclass
from typing import Union

import numpy as np
from qibo import Circuit
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.engines import (
    NativeTensorNetworkEngine,
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.engines.tensor_networks.quimb import _qibo_circuit_to_quimb
from mpstab.evolutors.optimization import minimize_expectation_dmrg
from mpstab.evolutors.utils import dressed_rotations, gate_angle, gate_generator
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.models.ansatze import Ansatz, CircuitAnsatz
from mpstab.models.entropies import stabilizer_renyi_entropy_mps
from mpstab.pauli import validate_pauli_string


@dataclass
class HSMPO:
    """
    Hybrid stabilizer-MPO surrogate of a quantum circuit.

    The circuit is split into a Clifford part, handled by a stabilizers engine,
    and the magic gates it cannot absorb, applied to an MPS as dressed Pauli
    rotations by a tensor-network engine. Both engines are pluggable through
    :meth:`set_engines`.

    Args:
        ansatz: the circuit to surrogate, as an :class:`~mpstab.models.ansatze.Ansatz`
            or a plain qibo ``Circuit``.
        max_bond_dimension: MPS truncation cap, or ``None`` for no cap.
        initial_state: single-qubit-gates-only circuit preparing the initial
            state. Defaults to ``|0...0>``.
    """

    ansatz: Union[Ansatz, Circuit]
    max_bond_dimension: int = None
    initial_state: Circuit = None

    def __post_init__(self):
        if isinstance(self.ansatz, Circuit):
            self.ansatz = CircuitAnsatz(qibo_circuit=self.ansatz)

        if self.initial_state is None:
            self.initial_state = Circuit(self.ansatz.nqubits)

        # set_engines installs the default engines, builds the MPS and evolves
        # it once, caching magic_gates and clifford_circuit along the way.
        self.set_engines()

    @classmethod
    def rotations_only(
        cls,
        ansatz: Union[Ansatz, Circuit],
        stab_engine: StabilizersEngine = None,
        tn_engine: TensorNetworkEngine = None,
    ):
        """
        Build an instance without evolving the MPS, for the dressed-rotation and
        resynthesis paths that never need it.

        Sets up only the ansatz, initial state, engines, ``magic_gates`` and
        ``clifford_circuit``, so construction stays cheap and never touches the
        (generally exponentially large) exact state. Anything needing
        ``original_circuit_mps``, :meth:`expectation` included, raises
        ``AttributeError``; call :meth:`set_engines` to upgrade to a full instance.

        Args:
            ansatz: the circuit to surrogate.
            stab_engine: stabilizers engine, ``StimEngine`` if ``None``.
            tn_engine: tensor-network engine, ``QuimbEngine`` if ``None``.
        """
        self = cls.__new__(cls)
        if isinstance(ansatz, Circuit):
            ansatz = CircuitAnsatz(qibo_circuit=ansatz)
        self.ansatz = ansatz
        self.max_bond_dimension = None
        self.initial_state = Circuit(ansatz.nqubits)
        self.stab_engine = _validated_stab_engine(stab_engine)
        self.tn_engine = _validated_tn_engine(tn_engine)
        (self.magic_gates, self.clifford_circuit), _ = ansatz.partitionate_circuit(
            replacement_probability=0.0, replacement_method="closest"
        )
        return self

    @property
    def nqubits(self):
        return self.ansatz.circuit.nqubits

    @property
    def nparams(self):
        return self.ansatz.nparams

    def set_engines(
        self,
        stab_engine: StabilizersEngine | None = None,
        tn_engine: TensorNetworkEngine | None = None,
    ):
        """
        Install the stabilizers and tensor-network engines, then rebuild the
        cached MPS with them.

        Args:
            stab_engine: stabilizers engine, ``StimEngine`` if ``None``.
            tn_engine: tensor-network engine, ``QuimbEngine`` if ``None``. The
                native engine supports :meth:`expectation` but not
                :meth:`minimize_expectation`.
        """
        self.stab_engine = _validated_stab_engine(stab_engine)
        self.tn_engine = _validated_tn_engine(tn_engine)
        self._init_tn(max_bond_dimension=self.max_bond_dimension)
        self.original_circuit_mps = self._precompute_original_mps()

    def _init_tn(self, max_bond_dimension: int | None = None):
        """
        Reset ``self.mps`` to the product state ``initial_state`` prepares.

        Both forms are handed to the engine, which takes whichever it prefers:
        the per-qubit amplitudes for the native engine, the circuit itself for
        quimb.

        Raises:
            ValueError: if ``initial_state`` entangles any pair of qubits.
        """
        amplitudes = []
        for qubit in range(self.nqubits):
            light_circuit, light_cone = self.initial_state.light_cone(qubit)
            if len(light_cone) > 1:
                raise ValueError(
                    "The initial-state circuit must be made of one-qubit gates "
                    f"only, but qubit {qubit}'s light cone covers {len(light_cone)}."
                )
            amplitudes.append(light_circuit().state())

        self.mps = self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_amplitudes=np.array(amplitudes),
            initial_state_circuit=self.initial_state,
            max_bond_dimension=max_bond_dimension,
        )

    def _dressed_rotations(self):
        """Every magic gate's ``(generator, signed_angle)``, in circuit order."""
        return dressed_rotations(
            self.nqubits, self.stab_engine, self.magic_gates, self.clifford_circuit
        )

    def _precompute_original_mps(self):
        """
        Evolve the MPS through every dressed rotation of the unmodified circuit.

        Also caches ``magic_gates`` and ``clifford_circuit``, so later expectation
        values do not re-partition the circuit.
        """
        self._init_tn(self.max_bond_dimension)
        evolved_mps = copy.deepcopy(self.mps)

        (self.magic_gates, self.clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=0.0, replacement_method="closest"
        )

        for generator, angle in self._dressed_rotations():
            self.tn_engine.pauli_rot(
                state_circuit=evolved_mps,
                generator=generator,
                angle=angle,
                max_bond_dimension=self.max_bond_dimension,
            )
        return evolved_mps

    def expectation(self, observable: Observable, return_fidelity: bool = False):
        """
        Expectation value of ``observable`` on the full circuit, with no
        magic-gate replacement.

        Args:
            observable: a Pauli string or any other format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            return_fidelity: also return the MPS's squared norm, which truncation
                pushes below 1.
        """
        constant, terms = pauli_terms(observable, self.nqubits)
        expval = constant + sum(
            coefficient * self._term_expectation(pauli, self.original_circuit_mps)
            for pauli, coefficient in terms.items()
        )

        if return_fidelity:
            return expval, self.original_circuit_mps.norm(squared=True)
        return expval

    def _term_expectation(self, pauli: str, state_mps) -> float:
        """
        Expectation value of one Pauli string, backpropagated through the cached
        Clifford circuit before being contracted against ``state_mps``.
        """
        backpropagated, sign = self.stab_engine.backpropagate(
            observable=pauli, clifford_circuit=self.clifford_circuit
        )
        mpo = self.tn_engine.pauli_mpo(backpropagated)
        return (
            np.real(self.tn_engine.expval(state_circuit=state_mps, operator=mpo)) * sign
        )

    def expectation_from_partition(
        self,
        observable: Observable,
        replacement_probability: float,
        replacement_method: str = "closest",
        return_partitions: bool = False,
    ):
        """
        Expectation value on a lower-magic circuit sampled from the ansatz.

        Args:
            observable: a Pauli string or any other format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            replacement_probability: chance of replacing each magic gate with a
                Clifford one.
            replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
            return_partitions: also return the magic gates, the Clifford-only
                circuit and the sampled full circuit.

        Returns:
            ``(expval, partitions)``, with ``partitions`` ``None`` unless
            ``return_partitions``.
        """
        self._init_tn(self.max_bond_dimension)
        (magic_gates, clifford_circuit), full_circuit = (
            self.ansatz.partitionate_circuit(
                replacement_probability=replacement_probability,
                replacement_method=replacement_method,
            )
        )
        self._evolve_magic_gates(self.mps, magic_gates, clifford_circuit)

        constant, terms = pauli_terms(observable, self.nqubits)
        expval = constant
        for pauli, coefficient in terms.items():
            backpropagated, sign = self.stab_engine.backpropagate(
                observable=pauli, clifford_circuit=clifford_circuit
            )
            mpo = self.tn_engine.pauli_mpo(backpropagated)
            expval += (
                coefficient
                * self.tn_engine.expval(state_circuit=self.mps, operator=mpo)
                * sign
            )

        partitions = (
            {
                "magic_gates": magic_gates,
                "only_cliffords": clifford_circuit,
                "full_circuit": full_circuit,
            }
            if return_partitions
            else None
        )
        return expval, partitions

    def _evolve_magic_gates(self, state_mps, magic_gates, clifford_circuit):
        """
        Apply each magic gate to ``state_mps`` as a dressed Pauli rotation,
        dressing it with the Clifford prefix that precedes it.
        """
        for breakpoint_index, magic_gate in magic_gates:
            prefix = self._clifford_subcircuit(clifford_circuit, breakpoint_index)
            generator, sign = self.stab_engine.backpropagate(
                gate_generator(magic_gate, self.nqubits), prefix
            )
            self.tn_engine.pauli_rot(
                state_circuit=state_mps,
                generator=generator,
                angle=gate_angle(magic_gate) * sign,
                max_bond_dimension=self.max_bond_dimension,
            )

    @staticmethod
    def _clifford_subcircuit(clifford_circuit: Circuit, k: int = 0) -> Circuit:
        """The first ``k`` gates of ``clifford_circuit``; all of them if ``k`` is ``None``."""
        queue = clifford_circuit.queue[:k] if k is not None else clifford_circuit.queue
        subcircuit = Circuit(clifford_circuit.nqubits)
        for gate in queue:
            subcircuit.add(gate)
        return subcircuit

    @property
    def n_magic_gates(self) -> int:
        """Number of non-Clifford ("magic") gates in the original AnsatzCircuit as split up by partitionate_circuit."""
        return len(self.magic_gates)

    @property
    def n_clifford_gates(self) -> int:
        """Number of Clifford gates in the precomputed Clifford-only Qibo Circuit."""
        return len(self.clifford_circuit.queue)

    @property
    def n_gates(self) -> int:
        """Total number of (non-measurement) gates in the original AnsatzCircuit."""
        return self.n_magic_gates + self.n_clifford_gates

    @property
    def truncation_fidelity_pure_tn(self) -> float:
        """
        Truncation fidelity of running the whole circuit as a plain MPS, with no
        stabilizer help: the baseline the hybrid representation improves on.
        """
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
        r"""
        Truncation fidelity of the hybrid MPS,
        :math:`|\langle\Psi_t|\Psi_t\rangle|^2`, which equals the fidelity against
        the untruncated state because :math:`\Psi` is normalised.
        """
        self._init_tn(self.max_bond_dimension)
        (magic_gates, clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=replacement_probability,
            replacement_method=replacement_method,
        )
        self._evolve_magic_gates(self.mps, magic_gates, clifford_circuit)
        return self.mps.norm(squared=True)

    def stabilizer_renyi_entropy(
        self, alpha: int = 2, n_samples: int = 1000, seed=None
    ):
        """
        Stochastic estimate of the stabilizer Renyi entropy of order ``alpha``.

        Args:
            alpha: Renyi order, at least 2.
            n_samples: random Pauli strings to sample.
            seed: RNG seed.
        """
        return stabilizer_renyi_entropy_mps(
            self, alpha=alpha, n_samples=n_samples, seed=seed
        )

    def get_parameters(self) -> np.ndarray:
        """The circuit's current parameters."""
        return self.ansatz.circuit.get_parameters()

    def set_parameters(self, parameters: np.ndarray):
        """Set the circuit's parameters and rebuild the cached MPS to match."""
        self.ansatz.circuit.set_parameters(parameters)
        self.original_circuit_mps = self._precompute_original_mps()

    def minimize_expectation(
        self,
        observables: Observable,
        method: str = "dmrg",
        bond_dims: Union[int, list] = None,
        cutoff: float = 1e-9,
        tol: float = 1e-6,
        max_sweeps: int = 10,
        verbosity: int = 1,
    ):
        """
        Minimize an observable over MPS tensors with DMRG, starting from the
        cached MPS.

        Args:
            observables: the Hamiltonian to minimize, in any format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            method: only ``"dmrg"`` is implemented.
            bond_dims: max bond dimension per sweep, or one value for all.
            cutoff: SVD truncation cutoff.
            tol: energy convergence tolerance.
            max_sweeps: maximum number of sweeps.
            verbosity: DMRG verbosity, 0 to 2.

        Returns:
            A dict with ``ground_state``, ``energy``, ``converged``,
            ``num_sweeps`` and ``energy_history``.

        Raises:
            NotImplementedError: with a tensor-network engine other than
                :class:`~mpstab.engines.QuimbEngine`, or an unimplemented method.
        """
        if isinstance(self.tn_engine, NativeTensorNetworkEngine):
            raise NotImplementedError(
                "DMRG optimization requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it; expectation() "
                "keeps working on NativeTensorNetworkEngine."
            )
        if method.lower() != "dmrg":
            raise ValueError(f"Unknown method {method!r}, expected 'dmrg'.")

        return minimize_expectation_dmrg(
            self,
            observables=observables,
            bond_dims=bond_dims,
            cutoff=cutoff,
            tol=tol,
            max_sweeps=max_sweeps,
            verbosity=verbosity,
        )


def _validated_stab_engine(stab_engine) -> StabilizersEngine:
    if stab_engine is None:
        return StimEngine()
    if not isinstance(stab_engine, StabilizersEngine):
        raise ValueError(
            f"{stab_engine} is not a StabilizersEngine; pass StimEngine() or "
            "NativeStabilizersEngine()."
        )
    return stab_engine


def _validated_tn_engine(tn_engine) -> TensorNetworkEngine:
    if tn_engine is None:
        return QuimbEngine()
    if not isinstance(tn_engine, TensorNetworkEngine):
        raise ValueError(
            f"{tn_engine} is not a TensorNetworkEngine; pass QuimbEngine() or "
            "NativeTensorNetworkEngine()."
        )
    return tn_engine
