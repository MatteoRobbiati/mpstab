"""The hybrid stabilizer-MPO surrogate of a quantum circuit."""

from dataclasses import MISSING, dataclass, fields, replace
from typing import List, Tuple, Union

import numpy as np
import stim
from qibo import Circuit
from tqdm.auto import tqdm

from mpstab.engines import (
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.evolutors.gradient import analytic_gradient
from mpstab.evolutors.optimization import minimize_expectation_dmrg
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.models.ansatze import Ansatz, CircuitAnsatz
from mpstab.models.noise.compile import compile_noise
from mpstab.models.noise.mc_noise_estimator import noisy_expectation
from mpstab.pauli import conjugate, pauli_light_cone


@dataclass
class HSMPO:
    """
    Hybrid stabilizer-MPO surrogate of a quantum circuit.

    The circuit is split into a Clifford part, handled by a stabilizers engine,
    and the magic gates it cannot absorb, applied to an MPS as dressed Pauli
    rotations by a tensor-network engine. Both engines are pluggable, at
    construction or later through :meth:`set_engines`.

    Args:
        ansatz: the circuit to surrogate, as an :class:`~mpstab.models.ansatze.Ansatz`
            or a plain qibo ``Circuit``.
        max_bond_dimension: MPS truncation cap, or ``None`` for no cap.
        initial_state: single-qubit-gates-only circuit preparing the initial
            state. Defaults to ``|0...0>``.
        progress: show a progress bar while the MPS is evolved through the
            dressed rotations. It only appears once the evolution has taken
            longer than about a second, so quick builds stay silent.
        stab_engine: stabilizers engine, ``StimEngine`` if ``None``.
        tn_engine: tensor-network engine, ``QuimbEngine`` if ``None``.
    """

    ansatz: Union[Ansatz, Circuit]
    max_bond_dimension: int = None
    initial_state: Circuit = None
    progress: bool = True
    stab_engine: StabilizersEngine = None
    tn_engine: TensorNetworkEngine = None

    def __post_init__(self):
        self._setup()
        self.original_circuit_mps = self._evolve_original()

    def _setup(self):
        """Normalise the inputs and install the engines; no MPS is built."""
        if isinstance(self.ansatz, Circuit):
            self.ansatz = CircuitAnsatz(qibo_circuit=self.ansatz)
        if self.initial_state is None:
            self.initial_state = Circuit(self.ansatz.nqubits)
        self.stab_engine = _engine(self.stab_engine, StabilizersEngine, StimEngine)
        self.tn_engine = _engine(self.tn_engine, TensorNetworkEngine, QuimbEngine)

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
        for f in fields(cls):
            setattr(self, f.name, None if f.default is MISSING else f.default)
        self.ansatz, self.stab_engine, self.tn_engine = ansatz, stab_engine, tn_engine
        self._setup()
        self._partition()
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
            tn_engine: tensor-network engine, ``QuimbEngine`` if ``None``.
        """
        self.stab_engine = _engine(stab_engine, StabilizersEngine, StimEngine)
        self.tn_engine = _engine(tn_engine, TensorNetworkEngine, QuimbEngine)
        self.original_circuit_mps = self._evolve_original()

    def initial_mps(self):
        """
        A fresh MPS in the product state ``initial_state`` prepares, capped at
        ``max_bond_dimension``. It belongs to the caller: evolving it leaves this
        instance untouched.

        Raises:
            ValueError: if ``initial_state`` entangles any pair of qubits.
        """
        for gate in self.initial_state.queue:
            if len(gate.qubits) > 1:
                raise ValueError(
                    "The initial-state circuit must be made of one-qubit gates "
                    f"only, but it has {gate.name} on qubits {gate.qubits}."
                )
        return self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_circuit=self.initial_state,
            max_bond_dimension=self.max_bond_dimension,
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

    def _partition(self):
        """
        Cache the circuit's magic gates and Clifford part, nothing replaced, and
        drop the dressed rotations derived from the previous ones.
        """
        (self.magic_gates, self.clifford_circuit), _ = (
            self.ansatz.partitionate_circuit()
        )
        self._dressed = None

    def _dressed_rotations(self) -> Tuple[Tuple[str, float], ...]:
        """
        Every magic gate's dressed ``(generator, signed_angle)``, in circuit order.

        Each gate's rotation is conjugated by the Clifford sub-circuit preceding
        it (:meth:`_dress`). Computed once per partition of the circuit, i.e.
        once per :meth:`set_parameters`.

        """
        if self._dressed is None:
            self._dressed = tuple(self._dress(self.magic_gates, self.clifford_circuit))
        return self._dressed

    def _dress(self, magic_gates, clifford_circuit) -> List[Tuple[str, float]]:
        """
        Dressed ``(generator, signed_angle)`` of each ``(breakpoint, gate)`` in
        ``magic_gates``, conjugated by the first ``breakpoint`` gates of
        ``clifford_circuit``.
        """
        return [
            (generator, sign * gate_angle(gate))
            for (generator, sign), (_, gate) in zip(
                self._dress_generators(magic_gates, clifford_circuit), magic_gates
            )
        ]

    def _dress_generators(self, magic_gates, clifford_circuit) -> List[Tuple[str, int]]:
        """
        Dressed ``(generator, sign)`` of each ``(breakpoint, gate)`` in
        ``magic_gates``: the gate's generator conjugated by the first
        ``breakpoint`` gates of ``clifford_circuit`` is ``sign * generator``.

        The stabilizers engine does the dressing
        (:meth:`~mpstab.engines.StabilizersEngine.backpropagate_at`).
        """
        return self.stab_engine.backpropagate_at(
            [gate_generator(gate, self.nqubits) for _, gate in magic_gates],
            [breakpoint_index for breakpoint_index, _ in magic_gates],
            clifford_circuit,
        )

    def _evolve_original(self):
        """
        Partition the unmodified circuit and evolve a fresh MPS through all its
        dressed rotations.
        """
        self._partition()
        mps = self.initial_mps()
        self._apply_rotations(mps, self._dressed_rotations(), "building MPS")
        return mps

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
        backpropagated = self.stab_engine.backpropagate_many(
            terms, self.clifford_circuit
        )
        expval = constant + self._evaluate_terms(
            terms, backpropagated, self.original_circuit_mps
        )

        if return_fidelity:
            return expval, self.tn_engine.squared_norm(self.original_circuit_mps)
        return expval

    def gradient(
        self,
        observable: Observable,
        parameters: List[int] | None = None,
        light_cone: bool = True,
        method: str = "adjoint",
        costate_bond: int | None = None,
        groups: int = 1,
        n_workers: int | None = None,
        return_value: bool = False,
    ):
        """
        Gradient of :meth:`expectation` with respect to the circuit parameters,
        analytically, instead of the two shifted circuits per parameter of the
        parameter-shift rule: its cost does not grow with the number of
        parameters. See
        :func:`~mpstab.evolutors.gradient.analytic_gradient`, which also
        reports the rotations applied and the truncation fidelities.

        Args:
            observable: a Pauli string or any other format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            parameters: indices into :meth:`get_parameters` to differentiate,
                all of them by default.
            light_cone: drop the rotations that cannot affect the observable
                first, which is exact.
            method: ``"adjoint"`` (default), one forward pass and one
                backward pass of a costate of twice the bond dimension; or
                ``"flips"``, one shifted tail per parameter, with no costate to
                truncate.
            costate_bond: bond cap of the adjoint costate, twice
                ``max_bond_dimension`` by default; ``max_bond_dimension``
                itself is about twice as fast, for twice the error.
            groups: differentiate the observable's terms in this many blocks of
                neighbouring terms, each on its own, narrower light cone. Faster
                and more accurate once the union of the cones saturates
                ``max_bond_dimension``; one block (default) uses the union.
            n_workers: processes the blocks run on in parallel,
                ``min(groups, os.cpu_count())`` by default. The pool starts on
                the first call (about a second per process) and is reused.
            return_value: also return the expectation value, which the
                gradient's forward pass computes anyway: what a training step
                needs. It is exact like :meth:`expectation`, but evaluated on
                the light-cone chain at the parameters of the last partition
                (:meth:`set_parameters`), not on the cached MPS.

        Returns:
            One derivative per requested parameter, or ``(value, gradient)``
            with ``return_value``.

        Raises:
            ValueError: if a requested parameter's gate sits at a Clifford
                angle, so that it was absorbed into the tableau and has no
                rotation to differentiate.
        """
        result = analytic_gradient(
            self,
            observable,
            parameters,
            light_cone,
            method,
            costate_bond,
            groups,
            n_workers,
        )
        if return_value:
            return result.value, result.values
        return result.values

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
        more are sampled against models built from the single-fault values,
        whose sums are known in closed form. Coherent errors join the rotation chain like magic
        gates. A general channel, such as amplitude damping, adds transition
        amplitudes between two sign patterns, one for the ket and one for the
        bra. Only the rotations and faults in the light cone are kept, which is
        exact. Readout errors are folded into the observable's terms.

        Args:
            observable: a Pauli string or any other format
                :func:`~mpstab.hamiltonians.pauli_terms` accepts.
            noise: a :class:`~mpstab.models.noise.NoiseModel`.
            n_samples: fault configurations sampled for two or more faults. With
                ``0`` that stratum is dropped, biasing the mean by at most
                ``stratum_weights[2]`` times the sum of the absolute coefficients.
            seed: seed of the sampler.
            n_workers: processes evaluating the fault patterns and transition
                amplitudes in parallel. The result does not depend on it; each
                extra process costs about a second to start, so use it for
                large runs.

        Returns:
            :class:`~mpstab.models.noise.NoisyExpectation`, with the value in ``mean``
            and its standard error in ``stderr``.
        """
        constant, terms = noise.readout_observable(observable, self.nqubits)
        compiled = compile_noise(self, noise.events(self.ansatz.circuit))
        result = noisy_expectation(
            compiled,
            terms,
            n_samples=n_samples,
            seed=seed,
            n_workers=n_workers,
        )
        return replace(result, mean=result.mean + constant)

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
                circuit, the sampled full circuit and the evolved MPS.
            light_cone: skip the dressed rotations that cannot affect the
                observable (:func:`~mpstab.pauli.pauli_light_cone`). This is exact,
                and with a bond-dimension cap it also spares those rotations'
                truncation error.

        Returns:
            ``(expval, partitions)``, with ``partitions`` ``None`` unless
            ``return_partitions``.
        """
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
        mps = self.initial_mps()
        self._apply_rotations(mps, rotations)
        expval = constant + self._evaluate_terms(terms, backpropagated, mps)

        partitions = (
            {
                "magic_gates": magic_gates,
                "only_cliffords": clifford_circuit,
                "full_circuit": full_circuit,
                "mps": mps,
            }
            if return_partitions
            else None
        )
        return expval, partitions

    def _apply_rotations(self, state_mps, rotations, description="evolving MPS"):
        """
        Apply ``(generator, angle)`` rotations to ``state_mps``, in order.

        With :attr:`progress`, a progress bar shows the rotations applied and
        the current bond dimension, once the evolution has taken longer than
        :data:`_PROGRESS_DELAY` seconds.
        """
        for generator, angle in _progress_bar(
            rotations,
            lambda: self.tn_engine.bond_dimension(state_mps),
            description,
            enabled=self.progress,
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
        return self.tn_engine.circuit_truncation_fidelity(
            self.nqubits,
            self.initial_state + self.ansatz.circuit,
            self.max_bond_dimension,
        )

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
        (magic_gates, clifford_circuit), _ = self.ansatz.partitionate_circuit(
            replacement_probability=replacement_probability,
            replacement_method=replacement_method,
        )
        mps = self.initial_mps()
        self._apply_rotations(mps, self._dress(magic_gates, clifford_circuit))
        return self.tn_engine.squared_norm(mps)

    def get_parameters(self) -> np.ndarray:
        """The circuit's current parameters."""
        return self.ansatz.circuit.get_parameters()

    def set_parameters(self, parameters: np.ndarray):
        """
        Set the circuit's parameters and re-partition the circuit. The cached
        MPS is rebuilt on its next use, so a training loop that only calls
        :meth:`gradient` never pays for it.
        """
        self.ansatz.circuit.set_parameters(parameters)
        self._partition()
        self.__dict__.pop("original_circuit_mps", None)
        self._stale_mps = True

    def __getattr__(self, name):
        # Only reached for missing attributes: rebuild the MPS that
        # set_parameters dropped, on first use.
        if name == "original_circuit_mps" and self.__dict__.get("_stale_mps"):
            mps = self.initial_mps()
            self._apply_rotations(mps, self._dressed_rotations(), "building MPS")
            self.original_circuit_mps, self._stale_mps = mps, False
            return mps
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

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
            NotImplementedError: with a tensor-network engine that does not
                implement :meth:`~mpstab.engines.TensorNetworkEngine.minimize_energy`.
        """
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


def _engine(engine, base: type, default: type):
    """``engine``, or a ``default()`` one if ``None``; it must be a ``base``."""
    if engine is None:
        return default()
    if not isinstance(engine, base):
        raise ValueError(f"{engine} is not a {base.__name__}.")
    return engine


#: Seconds an evolution must run before its progress bar appears.
_PROGRESS_DELAY = 1.0


def _progress_bar(rotations, bond, description: str, enabled: bool = True):
    """
    Iterate over ``rotations``, drawing a progress bar once the loop has run
    for :data:`_PROGRESS_DELAY` seconds, with the bond dimension ``bond()``
    returns when it is not ``None``.
    """
    rotations = list(rotations)
    if not enabled or not rotations:
        yield from rotations
        return
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
            if step % every == 0 or step == len(rotations):
                chi = bond()
                if chi is not None:
                    bar.set_postfix_str(f"chi={chi}", refresh=False)
            bar.update()
