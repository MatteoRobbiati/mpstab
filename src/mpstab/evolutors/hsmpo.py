"""The hybrid stabilizer-MPO surrogate of a quantum circuit."""

from dataclasses import dataclass, replace
from typing import List, Tuple, Union

import numpy as np
import stim
from qibo import Circuit
from qibo.hamiltonians import SymbolicHamiltonian
from tqdm.auto import tqdm

from mpstab.engines import (
    NativeTensorNetworkEngine,
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.engines.tensor_networks.quimb import _qibo_circuit_to_quimb
from mpstab.evolutors.optimization import minimize_expectation_dmrg
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.models.ansatze import Ansatz, CircuitAnsatz
from mpstab.models.metrics import stabilizer_renyi_entropy_mps
from mpstab.noise.compile import compile_noise
from mpstab.noise.mc_noise_estimator import noisy_expectation
from mpstab.pauli import conjugate
from mpstab.pauli import light_cone as pauli_light_cone
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
        progress: show a progress bar while the MPS is evolved through the
            dressed rotations. It only appears once the evolution has taken
            longer than about a second, so quick builds stay silent.
    """

    ansatz: Union[Ansatz, Circuit]
    max_bond_dimension: int = None
    initial_state: Circuit = None
    progress: bool = True

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
        self.progress = True
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
        """Reset ``self.mps`` to the product state ``initial_state`` prepares."""
        self.mps = self._product_state_mps(max_bond_dimension)

    def initial_mps(self):
        """
        A fresh MPS in the product state ``initial_state`` prepares, capped at
        ``max_bond_dimension``.

        Unlike the cached ``mps`` and ``original_circuit_mps``, the returned
        state belongs to the caller: evolving it leaves this instance untouched.

        Raises:
            ValueError: if ``initial_state`` entangles any pair of qubits.
        """
        return self._product_state_mps(self.max_bond_dimension)

    def _product_state_mps(self, max_bond_dimension: int | None):
        """
        Build the product-state MPS ``initial_state`` prepares.

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

        return self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_amplitudes=np.array(amplitudes),
            initial_state_circuit=self.initial_state,
            max_bond_dimension=max_bond_dimension,
        )

    @staticmethod
    def _conjugate_rotation(
        rotation: Tuple[str, float], tableau: stim.Tableau
    ) -> Tuple[str, float]:
        """
        Conjugate a Pauli rotation by a Clifford tableau:
        ``U exp(-i angle/2 G) U^dag = exp(-i angle/2 U G U^dag)``.

        ``U G U^dag = +-G'`` is again a Pauli string, so the sign moves into the
        angle and the result is still a ``(generator, angle)`` rotation.

        Args:
            rotation: ``(generator, angle)``, the generator a Pauli string.
            tableau: the Clifford ``U``. Pass ``tableau.inverse()`` to get
                ``U^dag R U``, as in :func:`~mpstab.pauli.conjugate`.

        Returns:
            ``(generator, angle)`` for the conjugated rotation.
        """
        generator, angle = rotation
        conjugated, sign = conjugate(generator, tableau)
        return conjugated, sign * angle

    def _dressed_rotations(self) -> List[Tuple[str, float]]:
        """
        Every magic gate's dressed ``(generator, signed_angle)``, in circuit order.

        Each gate's rotation is conjugated by the Clifford sub-circuit preceding
        it (:meth:`_conjugate_rotation`). The Clifford circuit is simulated once
        into a running ``stim.TableauSimulator`` and each rotation is conjugated
        at its own breakpoint, rather than re-simulating the prefix per magic
        gate.

        Requires a :class:`~mpstab.engines.StimEngine`, but only when there is
        magic to dress: a Clifford-only circuit has no dressed rotations whatever
        engine is in use.

        Raises:
            NotImplementedError: if there are magic gates and the stabilizers
                engine is not a ``StimEngine``.
        """
        if not self.magic_gates:
            return []

        if not isinstance(self.stab_engine, StimEngine):
            raise NotImplementedError(
                "Dressed-rotation extraction requires StimEngine (this circuit has "
                f"{len(self.magic_gates)} magic gate(s)). Call "
                "set_engines(stab_engine=StimEngine()) to enable it."
            )

        return self._dress(self.magic_gates, self.clifford_circuit)

    def _dress(self, magic_gates, clifford_circuit) -> List[Tuple[str, float]]:
        """
        Dressed ``(generator, signed_angle)`` of each ``(breakpoint, gate)`` in
        ``magic_gates``, conjugated by the first ``breakpoint`` gates of
        ``clifford_circuit``.

        With a :class:`~mpstab.engines.StimEngine` the Clifford circuit is
        simulated once, a segment at a time, so dressing every gate costs one
        pass over the circuit. Other engines backpropagate each generator
        through its own prefix, which is quadratic in the circuit length.
        """
        if not isinstance(self.stab_engine, StimEngine):
            rotations = []
            for breakpoint_index, gate in magic_gates:
                prefix = self._clifford_subcircuit(clifford_circuit, breakpoint_index)
                generator, sign = self.stab_engine.backpropagate(
                    gate_generator(gate, self.nqubits), prefix
                )
                rotations.append((generator, sign * gate_angle(gate)))
            return rotations

        instructions = self.stab_engine.instructions(clifford_circuit)
        simulator = stim.TableauSimulator()
        simulator.do(stim.Circuit(f"I {self.nqubits - 1}"))

        rotations: List[Tuple[str, float]] = []
        simulated = 0
        inverse = None
        for breakpoint_index, gate in magic_gates:
            if breakpoint_index < simulated:
                raise ValueError("Magic gates must be listed in circuit order.")
            if breakpoint_index > simulated or inverse is None:
                # Every Clifford gate up to this magic gate, in one stim call.
                segment = filter(None, instructions[simulated:breakpoint_index])
                simulator.do(stim.Circuit("\n".join(segment)))
                simulated = breakpoint_index
                # current_inverse_tableau() is U^dag for the Clifford prefix U.
                inverse = simulator.current_inverse_tableau()
            rotations.append(
                self._conjugate_rotation(
                    (gate_generator(gate, self.nqubits), gate_angle(gate)), inverse
                )
            )
        return rotations

    def _precompute_original_mps(self):
        """
        Evolve the MPS through every dressed rotation of the unmodified circuit.

        Also caches ``magic_gates`` and ``clifford_circuit``, so later expectation
        values do not re-partition the circuit.
        """
        self._init_tn(self.max_bond_dimension)
        evolved_mps = self.tn_engine.copy_state(self.mps)

        (self.magic_gates, self.clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=0.0, replacement_method="closest"
        )

        self._apply_rotations(
            evolved_mps, self._dressed_rotations(), description="building MPS"
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
        expval = constant + self._terms_expectation(
            terms, self.original_circuit_mps, self.clifford_circuit
        )

        if return_fidelity:
            return expval, self.original_circuit_mps.norm(squared=True)
        return expval

    def noisy_expectation(
        self,
        observable: Observable,
        noise,
        n_samples: int = 1000,
        seed: int | None = None,
        n_workers: int = 1,
    ):
        """
        Expectation value of ``observable`` on the circuit with ``noise``.

        Every set of Pauli faults only flips the signs of some dressed-rotation
        angles, so the noisy value is an average of noiseless HSMPO values:
        patterns with zero and one fault are summed exactly, those with two or
        more are sampled. Coherent errors join the rotation chain like magic
        gates. Only the rotations and faults in the observable's light cone are
        kept, which is exact. Readout errors multiply each term by ``1 - 2 p``
        per noisy qubit it acts on.

        Args:
            observable: a Pauli string or any other format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            noise: a :class:`~mpstab.noise.NoiseModel`.
            n_samples: fault configurations sampled for two or more faults. With
                ``0`` that stratum is dropped, biasing the mean by at most
                ``stratum_weights[2]`` times the sum of the absolute coefficients.
            seed: seed of the sampler.
            n_workers: processes evaluating the fault patterns in parallel. The
                result does not depend on it; each extra process costs about a
                second to start, so use it for large runs.

        Returns:
            :class:`~mpstab.noise.NoisyExpectation`, with the value in ``mean``
            and its standard error in ``stderr``.
        """
        constant, terms = noise.readout_observable(observable, self.nqubits)
        compiled = compile_noise(self, noise.events(self.ansatz.circuit))
        result = noisy_expectation(
            compiled, terms, n_samples=n_samples, seed=seed, n_workers=n_workers
        )
        return replace(result, mean=result.mean + constant)

    def _terms_expectation(self, terms: dict, state_mps, clifford_circuit) -> float:
        """
        ``sum_P c_P <P>`` for ``terms = {P: c_P}``, every ``P`` backpropagated
        through ``clifford_circuit`` before being evaluated on ``state_mps``.

        Both steps go through the engines' batched methods, so the Clifford
        tableau is built once and the MPS is swept once for all the terms.
        """
        backpropagated = self.stab_engine.backpropagate_many(terms, clifford_circuit)
        return self._evaluate_terms(terms, backpropagated, state_mps)

    def _evaluate_terms(self, terms: dict, backpropagated: list, state_mps) -> float:
        """
        ``sum_P c_P <P>`` for ``terms = {P: c_P}``, given each term
        backpropagated as ``(pauli, sign)``, in one sweep of ``state_mps``.
        """
        if not terms:
            return 0.0
        values = self.tn_engine.pauli_expvals(
            state_mps, [pauli for pauli, _ in backpropagated]
        )
        return sum(
            coefficient * np.real(value) * sign
            for coefficient, (_, sign), value in zip(
                terms.values(), backpropagated, values
            )
        )

    def expectation_from_partition(
        self,
        observable: Observable,
        replacement_probability: float,
        replacement_method: str = "closest",
        return_partitions: bool = False,
        light_cone: bool = True,
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
            light_cone: skip the dressed rotations that cannot affect the
                observable (:func:`~mpstab.pauli.light_cone`). This is exact,
                and with a bond-dimension cap it also spares those rotations'
                truncation error.

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
        constant, terms = pauli_terms(observable, self.nqubits)
        backpropagated = self.stab_engine.backpropagate_many(terms, clifford_circuit)
        rotations = self._dress(magic_gates, clifford_circuit)
        if light_cone:
            kept = pauli_light_cone(
                [generator for generator, _ in rotations],
                [pauli for pauli, _ in backpropagated],
                self.nqubits,
            )
            rotations = [rotations[k] for k in kept]
        self._apply_rotations(self.mps, rotations)
        expval = constant + self._evaluate_terms(terms, backpropagated, self.mps)

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
        self._apply_rotations(state_mps, self._dress(magic_gates, clifford_circuit))

    def _apply_rotations(self, state_mps, rotations, description="evolving MPS"):
        """
        Apply ``(generator, angle)`` rotations to ``state_mps``, in order.

        With :attr:`progress`, a progress bar shows the rotations applied and
        the current bond dimension, once the evolution has taken longer than
        :data:`_PROGRESS_DELAY` seconds.
        """
        for generator, angle in _progress_bar(
            rotations, state_mps, description, enabled=self.progress
        ):
            self.tn_engine.pauli_rot(
                state_circuit=state_mps,
                generator=generator,
                angle=angle,
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


#: Seconds an evolution must run before its progress bar appears.
_PROGRESS_DELAY = 1.0


def _progress_bar(rotations, state_mps, description: str, enabled: bool = True):
    """
    Iterate over ``rotations``, drawing a progress bar with the bond dimension
    of ``state_mps`` once the loop has run for :data:`_PROGRESS_DELAY` seconds.
    """
    rotations = list(rotations)
    if not enabled or not rotations:
        yield from rotations
        return
    bond = getattr(state_mps, "max_bond", None)
    every = max(1, len(rotations) // 200)  # refresh chi ~200 times at most
    with tqdm(
        total=len(rotations),
        desc=description,
        unit="rot",
        delay=_PROGRESS_DELAY,
        ascii=" ━",
        colour="green",
        bar_format="{desc} {bar} {n_fmt}/{total_fmt} rotations "
        "{rate_fmt} eta {remaining}{postfix}",
        leave=True,
    ) as bar:
        for step, rotation in enumerate(rotations, 1):
            yield rotation
            if bond is not None and (step % every == 0 or step == len(rotations)):
                bar.set_postfix_str(f"chi={bond()}", refresh=False)
            bar.update()
