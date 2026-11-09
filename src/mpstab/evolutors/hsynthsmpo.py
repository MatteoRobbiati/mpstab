"""
The head/tail split of an HSMPO's dressed rotations, measured with finite shots.

The dressed-rotation chain is cut at ``cut_index``. The head is resynthesised
into a circuit a device can run; the tail is folded into the observable, leaving
an MPO. :meth:`HSynthSMPO.expectation_at_cut` then measures the head against that
observable with a shot budget, by one of the routes below:

- ``"pauli"`` samples Pauli strings from the tail MPO and groups them into
  qubit-wise-commuting measurement settings.
- ``"shadows"`` measures random single-qubit Pauli bases and contracts each
  snapshot against the tail MPO directly; ``"tnice"`` runs on the same
  measurements and only replaces this estimator with the MPS-optimized one.

Resynthesising the head leaves a Clifford residual, so a device prepares
``U(residual) U(head)|psi_0>`` rather than the exact head state the tail
coefficients were derived against. The ``"pauli"`` route corrects for this by
conjugating every sampled string through the residual. The ``"shadows"`` and
``"tnice"`` routes take a ``tail_handling``:

- ``"append_to_head_circuit"`` (default) runs the residual as extra gates after
  the head: exact, with the shadow variance of the unfolded tail;
- ``"fold"`` conjugates the observable *and every tail generator* through the
  residual before folding the tail, so the device runs the head alone. A Pauli
  conjugated by a Clifford is a Pauli, so the tail stays a Pauli-rotation fold
  with the same MPO structure, and the result is exactly ``R^dag O' R``. It
  saves the residual's two-qubit gates, but the residual spreads Pauli weight,
  so the shadow variance ``sum_P c_P^2 3^w_P`` of the folded MPO can grow by
  orders of magnitude: whether it pays off depends on the device's two-qubit
  error.

:meth:`HSynthSMPO.expectation_at_cut` chains three stages that can also be run
separately, e.g. to execute the circuits on a real device and post-process the
results later: :meth:`HSynthSMPO.plan` builds the circuits and can save them as
OpenQASM, a device backend runs them, and
:func:`~mpstab.quantum_hardware.estimate.estimate` recombines the frequencies
using the plan.

``expectation_at_cut(method="simulation")`` is the classical MPO-MPS
contraction of the same split, with the head as a truncated MPS. It is a
reference for validating the hardware methods, not a measurement route: this
class is meant for hardware, and the hardware methods are the recommended way
to use it.
"""

import dataclasses
import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import stim

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.pauli import validate_pauli_string
from mpstab.quantum_hardware import (
    ExpectationResult,
    MeasurementPlan,
    QiboSimulator,
    build_head_and_residual,
    build_cnot_ladder_head_and_residual,
    build_pauli_plan,
    build_shadow_plan,
    count_two_qubit_gates,
    estimate,
    execute_plan,
    fold_pool_through_tableau,
    head_to_qibo_circuit,
    pool_pauli_terms,
    sample_pauli_strings,
    save_circuits,
    truncation_error_estimate,
)

__all__ = ["HSynthSMPO", "ExpectationResult", "ResynthesizedHead", "TailTruncation"]


@dataclass(frozen=True)
class ResynthesizedHead:
    """
    A head circuit resynthesised for hardware, plus its Clifford residual.

    Returned by :meth:`HSynthSMPO.resynthesize_head`.

    Attributes:
        circuit: a runnable qibo ``Circuit``, without measurement gates.
        residual_tableau: the Clifford residual as a ``stim.Tableau``.
        residual_gates: the same residual as a gate list, for
            ``tail_handling="append_to_head_circuit"``.
        n_gates: gates in the head.
        n_two_qubit_gates: two-qubit gates in the head.
        cut_index: the split point the head came from.
        method: ``"rustiq"`` or ``"cnot_ladder"``.
    """

    circuit: object
    residual_tableau: object
    residual_gates: tuple
    n_gates: int
    n_two_qubit_gates: int
    cut_index: int
    method: str


@dataclass(frozen=True)
class TailTruncation:
    """
    How closely the truncated tail fold approximates the exact one, cheapest view
    first. Returned by :meth:`HSynthSMPO.tail_truncation`.

    The first two fields compare operators only. They depend on the observable
    the tail is folded onto, not on the head state.

    Attributes:
        fidelity_estimate: ``||O_approx||_F**2 / 2**n``, a norm-ratio estimate
            needing only the working truncation, with no exact reference built.
        relative_frobenius_error: ``||O_approx - O_exact||_F / ||O_exact||_F``;
            ``None`` unless ``exact=True`` was requested.
        expval_abs_error: ``|<O_approx> - <O_exact>|`` on the exact head state;
            ``None`` unless ``with_expval=True`` was requested.
    """

    fidelity_estimate: float
    relative_frobenius_error: Union[float, None]
    expval_abs_error: Union[float, None]


@dataclass
class HSynthSMPO(HSMPO):
    """
    An :class:`~mpstab.evolutors.hsmpo.HSMPO` whose dressed rotations are split
    into a resynthesised head and an MPO tail, with the head measured from a
    finite shot budget.

    Requires :class:`~mpstab.engines.QuimbEngine` for the tail, the only engine
    that can conjugate an operator.
    """

    def build_post_head_mpo(
        self,
        observable: str,
        cut_index: int,
        max_bond_dimension: int = -1,
        residual: Union[stim.Tableau, None] = None,
    ):
        """
        The observable with everything after the head folded in, as an MPO.

        Heisenberg picture: the Clifford part, the tail rotations ``[cut_index:]``
        and, optionally, a Clifford residual are moved into the observable, so
        that ``<O> = sign * <head| operator |head>``. The steps are:

        1. backpropagate ``observable`` through the Clifford part with ``stim``,
           ``O -> sign * O'``, still a single Pauli string;
        2. if ``residual`` ``R`` is given, conjugate ``O'`` and every tail
           generator through it. Each one stays a Pauli string, and a
           generator's sign goes into its angle. This gives exactly
           ``R^dag T^dag O' T R`` without ever building ``R`` as an operator;
        3. turn ``O'`` into an MPO and conjugate it by the tail rotations,
           innermost first, truncating each step to ``max_bond_dimension``.

        Without ``residual`` the MPO pairs with the exact head state. With it,
        the MPO pairs with :attr:`ResynthesizedHead.circuit` run alone, so the
        device skips the residual's gates.

        Callers:

        - ``expectation_at_cut(method="simulation")`` and :meth:`tail_truncation`
          use it as the classical reference;
        - the ``"pauli"`` route passes no ``residual``. It samples Pauli strings
          from the MPO and folds each sampled string through ``R`` afterwards
          (:func:`~mpstab.quantum_hardware.pauli_expansion.fold_pool_through_tableau`),
          which is cheaper and unaffected by truncation;
        - the ``"shadows"`` and ``"tnice"`` routes pass ``residual`` under
          ``tail_handling="fold"``, because their estimators contract the whole
          MPO, so ``R`` has to be inside it. The price is usually more Pauli
          weight, and therefore a higher shadow variance.

        Args:
            observable: a Pauli string (qubit-0-leftmost).
            cut_index: the split point; only rotations ``[cut_index:]`` are
                folded.
            max_bond_dimension: bond cap for each conjugation. ``-1`` means
                ``self.max_bond_dimension``, ``None`` means untruncated.
            residual: the Clifford residual ``R`` to fold in, typically
                :attr:`ResynthesizedHead.residual_tableau`. ``None`` leaves it
                out.

        Returns:
            ``(operator, sign)``: the folded MPO and the +-1 sign picked up by
            the Clifford backpropagation and, if given, the residual fold.
        """
        validate_pauli_string(observable, self.nqubits)
        if max_bond_dimension == -1:
            max_bond_dimension = self.max_bond_dimension

        tail = self._dressed_rotations()[cut_index:]
        backpropagated, sign = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=self.clifford_circuit
        )
        if residual is not None:
            backpropagated, residual_sign = self.stab_engine.fold_pauli_through_tableau(
                backpropagated, residual, 1.0
            )
            sign = sign * residual_sign
            residual_inverse = residual.inverse()
            tail = [
                self._conjugate_rotation(rotation, residual_inverse)
                for rotation in tail
            ]

        operator = self.tn_engine.pauli_mpo(backpropagated)
        for generator, angle in reversed(tail):  # outermost rotation folded last
            operator = self.tn_engine.conjugate_operator(
                operator, generator, angle, max_bond_dimension
            )
        return operator, sign

    def _head_state_mps(self, head: List[Tuple[str, float]]):
        """The exact MPS after applying the head's dressed rotations."""
        state_mps = self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_amplitudes=None,
            initial_state_circuit=self.initial_state,
            max_bond_dimension=self.max_bond_dimension,
        )
        for generator, angle in head:
            self.tn_engine.pauli_rot(
                state_circuit=state_mps,
                generator=generator,
                angle=angle,
                max_bond_dimension=self.max_bond_dimension,
            )
        return state_mps

    def _simulate_at_cut(
        self, terms: dict, constant: float, cut_index: int, max_bond_dimension
    ) -> ExpectationResult:
        """
        ``expectation_at_cut(method="simulation")``: the classical split, with
        the head applied to the state MPS and the tail folded into each term.

        The head MPS is built once and shared by every term. ``truncation_l2``
        estimates the tail truncation only, with the same norm-ratio proxy as the
        ``"shadows"`` route; the head truncation is reported separately as
        ``state_fidelity``, the head MPS's squared norm, and is not part of
        ``total_error``.
        """
        state_mps = self._head_state_mps(self._dressed_rotations()[:cut_index])

        value = constant
        truncation_l2 = 0.0
        for pauli, coefficient in terms.items():
            operator, sign = self.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension
            )
            value += (
                coefficient
                * sign
                * np.real(
                    self.tn_engine.expval(state_circuit=state_mps, operator=operator)
                )
            )
            if max_bond_dimension is not None:
                truncation_l2 += abs(coefficient) * np.sqrt(
                    max(0.0, 1.0 - self._fold_fidelity(operator))
                )

        return ExpectationResult(
            value=float(np.real(value)),
            stderr=0.0,
            truncation_l1=None,
            truncation_l2=float(truncation_l2),
            n_settings=0,
            n_shots=0,
            state_fidelity=float(np.real(state_mps.norm(squared=True))),
        )

    def tail_truncation(
        self,
        observable: str,
        cut_index: int,
        reference_max_bond: Union[int, None] = None,
        exact: bool = True,
        residual: Union[stim.Tableau, None] = None,
        with_expval: bool = False,
        max_bond_dimension: Union[int, None] = -1,
    ) -> TailTruncation:
        """
        Quantify how closely the truncated tail fold approximates the exact one.

        By default only operators are compared: the tail fold truncated to
        ``max_bond_dimension`` against a reference fold truncated to
        ``reference_max_bond``, with no head state built. The truncation is
        strongly observable-dependent, because the tail is folded onto the
        target observable: the same circuit, cut and bond cap can be lossless
        for one observable (e.g. one outside the tail's backward light cone,
        which stays a single Pauli string) and heavily truncated for another.
        Re-assess it for every observable, or every Hamiltonian term.

        Args:
            observable: a Pauli string (qubit-0-leftmost).
            cut_index: the split point.
            reference_max_bond: bond cap for the reference fold when
                ``exact=True``; ``None`` means untruncated.
            exact: build the reference fold and report
                ``relative_frobenius_error``. When ``False``, report only the
                reference-free ``fidelity_estimate``.
            residual: forwarded to :meth:`build_post_head_mpo` for both folds, to
                assess the fold ``tail_handling="fold"`` actually measures.
            with_expval: also build the exact head state and report
                ``expval_abs_error``. Needs ``exact=True``.
            max_bond_dimension: bond cap for the fold under test. ``-1`` means
                ``self.max_bond_dimension``, ``None`` means untruncated.

        Returns:
            A :class:`TailTruncation`. ``fidelity_estimate`` is always set,
            ``relative_frobenius_error`` only with ``exact=True``, and
            ``expval_abs_error`` only with ``with_expval=True``; the others are
            ``None``.

        Raises:
            ValueError: for ``with_expval`` without ``exact``.
            NotImplementedError: for ``with_expval`` with ``residual``: the
                ``expval_abs_error`` of a folded tail needs the device head
                state ``R^dag |head>``, which is not built here.
        """
        validate_pauli_string(observable, self.nqubits)
        if with_expval and not exact:
            raise ValueError("tail_truncation(with_expval=True) needs exact=True.")
        if with_expval and residual is not None:
            raise NotImplementedError(
                "tail_truncation(residual=...) does not support with_expval=True."
            )

        operator_approx, sign = self.build_post_head_mpo(
            observable, cut_index, max_bond_dimension, residual=residual
        )
        fidelity_estimate = self._fold_fidelity(operator_approx)
        if not exact:
            return TailTruncation(fidelity_estimate, None, None)

        operator_exact, _ = self.build_post_head_mpo(
            observable, cut_index, reference_max_bond, residual=residual
        )
        exact_norm = float(np.real(operator_exact.norm()))
        difference_norm = float(np.real((operator_approx - operator_exact).norm()))
        relative_frobenius_error = (
            difference_norm / exact_norm if exact_norm != 0 else 0.0
        )
        if not with_expval:
            return TailTruncation(fidelity_estimate, relative_frobenius_error, None)

        state_mps = self._head_state_mps(self._dressed_rotations()[:cut_index])

        def expval(operator):
            return (
                np.real(
                    self.tn_engine.expval(state_circuit=state_mps, operator=operator)
                )
                * sign
            )

        return TailTruncation(
            fidelity_estimate=fidelity_estimate,
            relative_frobenius_error=relative_frobenius_error,
            expval_abs_error=abs(expval(operator_approx) - expval(operator_exact)),
        )

    def _fold_fidelity(self, operator) -> float:
        """
        ``||O||_F**2 / 2**n`` of a folded tail MPO: 1 for an exact fold of a
        Pauli string, since conjugation by a unitary preserves the norm, and
        lower by the weight the truncation discarded.
        """
        return float(np.real(operator.norm())) ** 2 / 2**self.nqubits

    def resynthesize_head(
        self, cut_index: int, metric_name: str = "count", preserve_order: bool = True
    ) -> ResynthesizedHead:
        """
        Resynthesise the head rotations ``[0:cut_index)`` into a runnable circuit
        plus a pure-Clifford residual.

        Prefers rustiq's Pauli-network synthesis. If ``rustiq`` is not installed it
        warns and falls back to the dependency-free CNOT-ladder decomposition,
        which gives no gate-count saving but keeps the hardware path working
        without the optional dependency.

        Cached on the head's dressed rotations (see :func:`_resynthesize`), so
        repeated calls at the same cut, e.g. from :meth:`expectation_at_cut` for
        several observables or methods, resynthesise once, while new parameters
        give new rotations and hence a fresh resynthesis. The returned head is
        shared between those calls: don't modify its ``circuit`` or
        ``residual_tableau`` in place.

        Args:
            cut_index: the split point.
            metric_name: rustiq metric, ``"count"`` or ``"depth"``.
            preserve_order: keep the (non-commuting) rotation order.
        """
        # float() so the key hashes by value whatever the backend returns as an
        # angle (a framework tensor would hash by identity, or not at all).
        rotations = tuple(
            (generator, float(angle))
            for generator, angle in self._dressed_rotations()[:cut_index]
        )
        return _resynthesize(
            rotations, self.nqubits, cut_index, metric_name, preserve_order
        )

    def expectation_at_cut(
        self,
        observable: Observable,
        cut_index: int,
        method: str = "pauli",
        n_shots: Union[int, None] = None,
        epsilon: Union[float, None] = None,
        backend=None,
        max_bond_dimension: Union[int, None] = -1,
        tail_handling: str = "append_to_head_circuit",
        seed: Union[int, None] = None,
        **method_kwargs,
    ) -> ExpectationResult:
        """
        Resynthesise the head, build the measurement circuits, run them and
        recombine the results, in one call: :meth:`plan`, then
        :func:`~mpstab.quantum_hardware.backend.execute_plan` on ``backend``,
        then :func:`~mpstab.quantum_hardware.estimate.estimate`. To run the
        circuits on a device and post-process later, call those three stages
        separately.

        This class is meant for hardware: use ``"pauli"``, ``"shadows"`` or
        ``"tnice"``, which run the head exactly on a device (or simulator) and
        keep only the tail classical. ``"simulation"`` instead contracts the
        head, as a truncated MPS, against the tail MPO, entirely classically. It
        is a reference for validating the hardware methods, or for comparing a
        device run against the classical split; it is not a measurement route,
        and at large scale it is exactly the computation the hardware methods
        exist to avoid.

        Args:
            observable: a Pauli string (qubit-0-leftmost) or a
                ``SymbolicHamiltonian``. A Hamiltonian folds one tail MPO per term
                but emits a single set of circuits, since the head does not depend
                on the observable, and pools Pauli strings shared between terms so
                each is measured once with the summed coefficient.
            cut_index: the split point.
            method: ``"pauli"``, ``"shadows"``, ``"tnice"`` or
                ``"simulation"``. ``"tnice"`` runs on exactly the same
                circuits, bases and shots as ``"shadows"`` -- see
                :meth:`_shadow_plan` -- and only replaces the fixed
                canonical-dual post-processing with the MPS-optimized
                estimator of :mod:`~mpstab.quantum_hardware.tnice`, so the two
                can be compared on identical measurement data.
                ``"simulation"`` is classical, see above and
                :meth:`_simulate_at_cut`: it needs no resynthesis, backend or
                shots, and ignores ``tail_handling`` and ``seed``.
            n_shots: a fixed shot budget. Exactly one of this and ``epsilon``
                for the hardware methods; neither for ``"simulation"``.
            epsilon: a target standard error, which sizes the shot budget from the
                per-route variance predictor. For ``"tnice"`` this reuses
                ``"shadows"``'s canonical-dual predictor, a conservative upper
                bound on the (unknown ahead of fitting) optimized variance.
            backend: anything with ``execute_circuits(circuits, nshots)``.
                Defaults to :class:`~mpstab.quantum_hardware.QiboSimulator`.
            max_bond_dimension: bond cap for every tail fold; ``-1`` means
                ``self.max_bond_dimension``. With ``"simulation"`` the head MPS
                is capped at ``self.max_bond_dimension``.
            tail_handling: ``"shadows"`` and ``"tnice"`` only, what to do with a
                non-trivial Clifford residual left by the resynthesis.
                ``"append_to_head_circuit"`` (default) runs it as extra gates
                after the head; ``"fold"`` reabsorbs it into the tail MPO (see
                :meth:`build_post_head_mpo`), so the device runs the head alone at a
                possibly much higher shadow variance.
            seed: RNG seed for Pauli sampling, random shadow bases, and (for
                ``"tnice"``) the train/test shot split, forwarded to
                :func:`~mpstab.quantum_hardware.estimate.estimate_tnice`.
            method_kwargs: ``"pauli"`` takes ``n_string_samples`` (default
                200); ``"shadows"`` and ``"tnice"`` take ``shots_per_setting``
                (default 1); ``"tnice"`` additionally takes
                ``lam`` (default 0.999), ``n_sweeps`` (default 4),
                ``bond_dimension`` (default ``None``, meaning the tail MPO's
                own) and ``test_fraction`` (default 0.5), forwarded to
                :func:`~mpstab.quantum_hardware.estimate.estimate_tnice`.

        Raises:
            ValueError: on an unknown ``method``, if neither or both of
                ``n_shots`` and ``epsilon`` are given for a hardware method, or
                if either is given with ``method="simulation"``.
        """
        if method == "simulation":
            if n_shots is not None or epsilon is not None:
                raise ValueError(
                    "method='simulation' spends no shots; drop n_shots and epsilon."
                )
            if max_bond_dimension == -1:
                max_bond_dimension = self.max_bond_dimension
            constant, terms = pauli_terms(observable, self.nqubits)
            return self._simulate_at_cut(terms, constant, cut_index, max_bond_dimension)

        estimator_kwargs = {}
        if method == "tnice":
            for key in ("lam", "n_sweeps", "bond_dimension", "test_fraction"):
                if key in method_kwargs:
                    estimator_kwargs[key] = method_kwargs.pop(key)
            estimator_kwargs["seed"] = seed

        plan = self.plan(
            observable,
            cut_index,
            method=method,
            n_shots=n_shots,
            epsilon=epsilon,
            max_bond_dimension=max_bond_dimension,
            tail_handling=tail_handling,
            seed=seed,
            out_folder=False,
            **method_kwargs,
        )
        frequencies = execute_plan(backend or QiboSimulator(), plan)
        return estimate(plan, frequencies, **estimator_kwargs)

    def plan(
        self,
        observable: Observable,
        cut_index: int,
        method: str = "pauli",
        n_shots: Union[int, None] = None,
        epsilon: Union[float, None] = None,
        max_bond_dimension: Union[int, None] = -1,
        tail_handling: str = "append_to_head_circuit",
        seed: Union[int, None] = None,
        out_folder=None,
        **plan_kwargs,
    ) -> MeasurementPlan:
        """
        Build the measurement circuits for ``observable`` at ``cut_index``,
        without running them, and save them as OpenQASM unless ``out_folder``
        is ``False``.

        This is the first of the three stages :meth:`expectation_at_cut` chains,
        exposed so the circuits can run anywhere, e.g. on a real device: run the
        saved circuits, then pass the counts and the returned plan to
        :func:`~mpstab.quantum_hardware.estimate.estimate`.

        Only the circuits and shots are saved, not the returned plan, which
        :func:`~mpstab.quantum_hardware.estimate.estimate` needs. Building it
        again takes the same model, the same arguments and a fixed ``seed``.
        Even then the residual's gates can differ, so the saved circuits are
        the ones to run: never replace them with those of a rebuilt plan.

        Args:
            observable: as in :meth:`expectation_at_cut`.
            cut_index: as in :meth:`expectation_at_cut`.
            method: ``"pauli"``, ``"shadows"`` or ``"tnice"``.
            n_shots: as in :meth:`expectation_at_cut`.
            epsilon: as in :meth:`expectation_at_cut`.
            max_bond_dimension: as in :meth:`expectation_at_cut`.
            tail_handling: as in :meth:`expectation_at_cut`.
            seed: as in :meth:`expectation_at_cut`.
            out_folder: where to save the circuits and shots, see
                :func:`~mpstab.quantum_hardware.storage.save_circuits`.
                ``None`` (the default) saves them to a new folder
                ``planned_runs/plan_<method>_cut<cut_index>_<timestamp>``
                under the current directory and prints its path; ``False``
                saves nothing.
            plan_kwargs: ``"pauli"`` takes ``n_string_samples`` (default 200);
                ``"shadows"`` and ``"tnice"`` take ``shots_per_setting``
                (default 1).

        Returns:
            A :class:`~mpstab.quantum_hardware.plan.MeasurementPlan`.

        Raises:
            ValueError: on an unknown ``method``, if neither or both of
                ``n_shots`` and ``epsilon`` are given, or if the circuits are
                saved (``out_folder`` is not ``False``) without a ``seed``.

        """
        if method not in ("pauli", "shadows", "tnice"):
            raise ValueError(
                f"Unknown method {method!r}, expected 'pauli', 'shadows', "
                "'tnice' or, in expectation_at_cut only, 'simulation'."
            )
        if (n_shots is None) == (epsilon is None):
            raise ValueError("Exactly one of n_shots or epsilon must be given.")
        if out_folder is not False and seed is None:
            raise ValueError(
                "plan() saves the circuits, so it needs a seed: the plan itself "
                "is not saved, and rebuilding it for estimate takes the same "
                "seed. Pass out_folder=False to build the plan without saving."
            )
        if max_bond_dimension == -1:
            max_bond_dimension = self.max_bond_dimension

        constant, terms = pauli_terms(observable, self.nqubits)
        resynthesis = self.resynthesize_head(cut_index)

        build_plan = self._pauli_plan if method == "pauli" else self._shadow_plan
        plan = build_plan(
            terms=terms,
            constant=constant,
            cut_index=cut_index,
            resynthesis=resynthesis,
            max_bond_dimension=max_bond_dimension,
            n_shots=n_shots,
            epsilon=epsilon,
            seed=seed,
            tail_handling=tail_handling,
            **plan_kwargs,
        )
        if method == "tnice":
            plan = dataclasses.replace(plan, method="tnice")
        if out_folder is None:
            # Local time with its UTC offset; microseconds, so plans built in
            # a loop do not collide.
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f%z")
            out_folder = f"planned_runs/plan_{method}_cut{cut_index}_{timestamp}"
            save_circuits(plan, out_folder)
            print(f"Circuits saved to {out_folder}")
        elif out_folder is not False:
            save_circuits(plan, out_folder)
        return plan

    def _pauli_plan(
        self,
        terms,
        constant,
        cut_index,
        resynthesis,
        max_bond_dimension,
        n_shots,
        epsilon,
        seed,
        tail_handling=None,
        n_string_samples=200,
    ):
        """Sample each term's tail MPO, pool the strings, fold the residual, group."""
        rng = np.random.default_rng(seed)
        ensembles, diagnostics = [], []
        for pauli, coefficient in terms.items():
            operator, sign = self.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension
            )
            ensemble = sample_pauli_strings(
                operator,
                n_samples=n_string_samples,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            ensembles.append((coefficient * sign, ensemble))
            l1, l2 = truncation_error_estimate(ensemble)
            diagnostics.append((abs(coefficient), l1, l2, ensemble.retained_weight))

        pooled = pool_pauli_terms(ensembles)
        if resynthesis.residual_tableau != stim.Tableau(self.nqubits):
            pooled = fold_pool_through_tableau(
                pooled, resynthesis.residual_tableau, self.stab_engine
            )

        truncation_l1, truncation_l2, retained_weight = _aggregate_truncation(
            diagnostics
        )
        return build_pauli_plan(
            resynthesis.circuit,
            self.nqubits,
            _assert_real_coefficients(pooled),
            n_shots,
            epsilon,
            constant=constant,
            truncation_l1=truncation_l1,
            truncation_l2=truncation_l2,
            retained_weight=retained_weight,
        )

    def _shadow_plan(
        self,
        terms,
        constant,
        cut_index,
        resynthesis,
        max_bond_dimension,
        n_shots,
        epsilon,
        seed,
        tail_handling,
        shots_per_setting=1,
    ):
        """
        Fold each term's tail MPO and pick random measurement bases.

        ``truncation_l2`` is estimated from each folded MPO's norm (see
        :meth:`_fold_fidelity`), so no reference fold and no head state are
        built.
        """
        circuit = self._shadow_circuit(resynthesis, tail_handling)
        residual = None
        if tail_handling == "fold" and resynthesis.residual_tableau != stim.Tableau(
            self.nqubits
        ):
            residual = resynthesis.residual_tableau

        mpo_terms = []
        truncation_l2 = 0.0
        for pauli, coefficient in terms.items():
            operator, sign = self.build_post_head_mpo(
                pauli, cut_index, max_bond_dimension, residual=residual
            )
            mpo_terms.append((pauli, coefficient, sign, operator))
            if max_bond_dimension is not None:
                truncation_l2 += abs(coefficient) * np.sqrt(
                    max(0.0, 1.0 - self._fold_fidelity(operator))
                )

        return build_shadow_plan(
            circuit,
            self.nqubits,
            mpo_terms,
            n_shots,
            epsilon,
            shots_per_setting=shots_per_setting,
            seed=seed,
            constant=constant,
            truncation_l2=float(truncation_l2),
        )

    def _shadow_circuit(self, resynthesis, tail_handling):
        """
        The circuit the shadows route measures, handling a non-trivial residual.

        Raises:
            ValueError: on an unknown ``tail_handling``.
        """
        if tail_handling not in ("append_to_head_circuit", "fold"):
            raise ValueError(
                f"Unknown tail_handling {tail_handling!r}, expected "
                "'append_to_head_circuit' or 'fold'."
            )
        if (
            resynthesis.residual_tableau == stim.Tableau(self.nqubits)
            or tail_handling == "fold"
        ):
            return resynthesis.circuit
        return resynthesis.circuit + head_to_qibo_circuit(
            list(resynthesis.residual_gates), self.nqubits
        )


@lru_cache(maxsize=32)
def _resynthesize(
    rotations: Tuple[Tuple[str, float], ...],
    nqubits: int,
    cut_index: int,
    metric_name: str,
    preserve_order: bool,
) -> ResynthesizedHead:
    """
    The body of :meth:`HSynthSMPO.resynthesize_head`, cached on its arguments.

    The key holds the head's dressed rotations rather than the evolutor, so a
    repeated call with the same rotations returns the same
    :class:`ResynthesizedHead` object without running rustiq again, and changing
    the circuit's parameters changes the angles, hence the key: the cache never
    serves a stale head and needs no invalidation.
    """
    paulis = [pauli for pauli, _ in rotations]
    angles = [angle for _, angle in rotations]
    try:
        head, residual_tableau, residual_gates = build_head_and_residual(
            paulis, angles, metric_name=metric_name, preserve_order=preserve_order
        )
        method = "rustiq"
    except ImportError:
        warnings.warn(
            "rustiq is not installed; falling back to the CNOT-ladder "
            "resynthesis (no gate-count saving). Install it with pip install "
            '"mpstab[rustiq]" for the optimized head.',
            stacklevel=3,
        )
        head, residual_tableau, residual_gates = build_cnot_ladder_head_and_residual(
            paulis, angles
        )
        method = "cnot_ladder"

    if len(residual_tableau) != nqubits:  # degenerate cut_index == 0 case
        residual_tableau = stim.Tableau(nqubits)

    return ResynthesizedHead(
        circuit=head_to_qibo_circuit(head, nqubits),
        residual_tableau=residual_tableau,
        residual_gates=tuple(residual_gates),
        n_gates=len(head),
        n_two_qubit_gates=count_two_qubit_gates(head),
        cut_index=cut_index,
        method=method,
    )


def _aggregate_truncation(diagnostics: list) -> Tuple[float, float, float]:
    """
    Combine per-term ``(|coeff|, l1, l2, retained_weight)`` truncation
    diagnostics.

    L1 adds by the triangle inequality, rigorous for any combination of terms;
    L2 adds in quadrature, assuming the terms' errors are independent;
    retained_weight is the ``|coeff|``-weighted average of each term's own
    :attr:`~mpstab.quantum_hardware.pauli_expansion.PauliEnsemble.retained_weight`
    -- a diagnostic for whether ``n_string_samples`` covered enough of the
    observable, not a rigorous combination of the per-term fractions.
    """
    if not diagnostics:
        return 0.0, 0.0, 1.0
    total_weight = sum(weight for weight, _, _, _ in diagnostics)
    l1 = sum(weight * term_l1 for weight, term_l1, _, _ in diagnostics)
    l2 = float(
        np.sqrt(sum((weight * term_l2) ** 2 for weight, _, term_l2, _ in diagnostics))
    )
    retained_weight = (
        sum(weight * retained for weight, _, _, retained in diagnostics) / total_weight
        if total_weight > 0
        else 1.0
    )
    return l1, l2, retained_weight


def _assert_real_coefficients(pooled: dict, tolerance: float = 1e-8) -> dict:
    """
    Coerce a pooled ``{pauli: complex coefficient}`` map to real, raising
    instead of silently truncating if any imaginary part isn't negligible.

    The tail-folded, Clifford-backpropagated observable is Hermitian, so
    every pooled coefficient should be real up to floating-point noise. A
    non-negligible imaginary part means backpropagation, tableau folding, or
    the input observable itself went wrong somewhere upstream -- silently
    dropping it (``float(np.real(c))``) would hide that bug behind a
    plausible-looking float instead of surfacing it.

    Raises:
        ValueError: if any coefficient's imaginary part exceeds ``tolerance``
            relative to its magnitude.
    """
    real_coefficients = {}
    for pauli, coefficient in pooled.items():
        magnitude = max(abs(coefficient), 1.0)
        if abs(np.imag(coefficient)) > tolerance * magnitude:
            raise ValueError(
                f"Pooled coefficient for {pauli!r} is not real within "
                f"tolerance ({coefficient!r}); the tail-folded observable "
                "should be Hermitian, so this points to a bug upstream "
                "(backpropagation, tableau folding, or a non-Hermitian "
                "input observable) rather than ordinary floating-point noise."
            )
        real_coefficients[pauli] = float(np.real(coefficient))
    return real_coefficients
