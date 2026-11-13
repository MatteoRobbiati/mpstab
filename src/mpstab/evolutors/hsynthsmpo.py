"""
The head/tail split of an HSMPO's dressed rotations, measured with finite shots.

The dressed-rotation chain is cut at ``cut_index``. The head is resynthesised
into a circuit a device can run; the tail is folded into the observable, leaving
an MPO. :meth:`HSynthSMPO.expectation_at_cut` then measures the head against that
observable with a shot budget, by one of the measurement methods of
:mod:`mpstab.qpu.measurement`, passed as an object or by name:

- :class:`~mpstab.qpu.measurement.pauli.PauliSampling` (``"pauli"``) samples
  Pauli strings from the tail MPO and groups them into qubit-wise-commuting
  measurement settings.
- :class:`~mpstab.qpu.measurement.shadows.ClassicalShadows` (``"shadows"``)
  measures random single-qubit Pauli bases and contracts each snapshot against
  the tail MPO directly; :class:`~mpstab.qpu.measurement.tnice.TNICE`
  (``"tnice"``) runs on the same measurements and only replaces this estimator
  with the MPS-optimized one.

The head is resynthesised by one of the synthesisers of
:mod:`mpstab.qpu.synthesis`, set as ``head_synthesis``.

Resynthesising the head leaves a Clifford residual, so a device prepares
``U(residual) U(head)|psi_0>`` rather than the exact head state the tail
coefficients were derived against. The ``"pauli"`` route corrects for this by
conjugating every sampled string through the residual. The ``"shadows"`` and
``"tnice"`` routes take a ``tail_handling`` option:

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
OpenQASM, a device backend runs them, and the method's ``estimate`` (or
:func:`~mpstab.qpu.measurement.estimate`) recombines the frequencies using the
plan.

``expectation_at_cut(method="simulation")`` is the classical MPO-MPS
contraction of the same split, with the head as a truncated MPS. It is a
reference for validating the hardware methods, not a measurement route: this
class is meant for hardware, and the hardware methods are the recommended way
to use it.
"""

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
from mpstab.qpu.execution import QiboSimulator, execute_plan, save_circuits
from mpstab.qpu.measurement import (
    ExpectationResult,
    MeasurementMethod,
    MeasurementPlan,
    resolve_method,
)
from mpstab.qpu.measurement.pauli_expansion import fold_fidelity
from mpstab.qpu.synthesis import (
    CnotLadderSynthesis,
    HeadSynthesis,
    RustiqSynthesis,
    count_two_qubit_gates,
    head_to_qibo_circuit,
    resolve_synthesis,
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
        method: the name of the synthesis that produced it, e.g.
            ``"extraction"``; ``"cnot_ladder"`` when rustiq was asked for but
            is not installed.
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

    Args:
        head_synthesis: how the head is resynthesised, a
            :class:`~mpstab.qpu.synthesis.abstract.HeadSynthesis` such as
            ``ExtractionSynthesis(connectivity="line")``, or the name of one
            with its default options: ``"rustiq"`` (default), ``"extraction"``
            or ``"cnot_ladder"``.
        light_cone: drop the rotations the observable cannot see, from the
            head the device runs and from the tail that is folded, see
            :meth:`relevant_rotations`. Exact, and off by default.
    """

    head_synthesis: Union[str, HeadSynthesis] = "rustiq"
    light_cone: bool = False

    def relevant_rotations(self, observable: Observable) -> Tuple[int, ...]:
        """
        Indices of the dressed rotations inside the light cone of ``observable``.

        Going backwards from the observable (a Pauli string in the dressed
        frame, per term), a rotation matters only if its string anticommutes
        with the observable or with a later rotation that matters. Any other
        rotation commutes with everything between itself and its own inverse
        in ``<psi| U^dag O U |psi>``, and cancels: removing it changes nothing.
        For an observable with several terms the union is returned, which is
        what a head shared by all the terms has to keep.

        Cheap (commutation of Pauli strings only) and exact. With a truncated
        tail it stays consistent with the truncation: the rotations removed
        are those that commute with the exact folded observable.
        """
        _, terms = pauli_terms(observable, self.nqubits)
        relevant = set()
        for pauli in terms:
            relevant |= self._relevant_for_pauli(pauli)
        return tuple(sorted(relevant))

    def _relevant_for_pauli(self, pauli: str) -> frozenset:
        """:meth:`relevant_rotations` of one Pauli string, cached on the rotations."""
        rotations = self._dressed_rotations()
        generators = tuple(generator for generator, _ in rotations)
        backpropagated, _ = self.stab_engine.backpropagate(
            observable=pauli, clifford_circuit=self.clifford_circuit
        )
        return _light_cone(generators, backpropagated)

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
          (:func:`~mpstab.qpu.measurement.pauli_expansion.fold_pool_through_tableau`),
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
        if self.light_cone:  # the rotations this string cannot see fold to nothing
            relevant = self._relevant_for_pauli(observable)
            tail = [
                rotation
                for index, rotation in enumerate(tail, start=cut_index)
                if index in relevant
            ]
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
        return fold_fidelity(operator, self.nqubits)

    def resynthesize_head(
        self,
        cut_index: int,
        synthesis: Union[str, HeadSynthesis, None] = None,
        observable: Union[Observable, None] = None,
    ) -> ResynthesizedHead:
        """
        Resynthesise the head rotations ``[0:cut_index)`` into a runnable circuit
        plus a pure-Clifford residual.

        If rustiq is asked for but not installed, this warns and falls back to
        :class:`~mpstab.qpu.synthesis.cnot_ladder.CnotLadderSynthesis`.

        Cached on the head's dressed rotations and the synthesis (see
        :func:`_resynthesize`), so repeated calls at the same cut, e.g. from
        :meth:`expectation_at_cut` for several observables or methods,
        resynthesise once, while new parameters give new rotations and hence a
        fresh resynthesis. The returned head is shared between those calls:
        don't modify its ``circuit`` or ``residual_tableau`` in place.

        Args:
            cut_index: the split point.
            synthesis: a :class:`~mpstab.qpu.synthesis.abstract.HeadSynthesis`
                or the name of one (default: the instance's
                ``head_synthesis``).
            observable: keep only the head rotations inside the light cone of
                this observable (:meth:`relevant_rotations`). The head is then
                valid for this observable only. ``None`` keeps them all.

        Raises:
            ValueError: on an unknown synthesis name.
        """
        synthesis = resolve_synthesis(
            self.head_synthesis if synthesis is None else synthesis
        )
        # float() so the key hashes by value whatever the backend returns as an
        # angle (a framework tensor would hash by identity, or not at all).
        dressed = self._dressed_rotations()[:cut_index]
        if observable is not None:
            relevant = self.relevant_rotations(observable)
            dressed = [dressed[index] for index in relevant if index < cut_index]
        rotations = tuple((generator, float(angle)) for generator, angle in dressed)
        return _resynthesize(rotations, self.nqubits, cut_index, synthesis)

    def expectation_at_cut(
        self,
        observable: Observable,
        cut_index: int,
        method: Union[str, MeasurementMethod] = "pauli",
        n_shots: Union[int, None] = None,
        epsilon: Union[float, None] = None,
        backend=None,
        max_bond_dimension: Union[int, None] = -1,
        seed: Union[int, None] = None,
        **method_options,
    ) -> ExpectationResult:
        """
        Resynthesise the head, build the measurement circuits, run them and
        recombine the results, in one call: :meth:`plan`, then
        :func:`~mpstab.qpu.execution.simulator.execute_plan` on ``backend``,
        then the method's ``estimate``. To run the circuits on a device and
        post-process later, call those three stages separately.

        This class is meant for hardware: use a measurement method, which runs
        the head exactly on a device (or simulator) and keeps only the tail
        classical. ``"simulation"`` instead contracts the
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
            method: a :class:`~mpstab.qpu.measurement.abstract.MeasurementMethod`,
                e.g. ``ClassicalShadows(tail_handling="fold")``; or the name of
                one, ``"pauli"``, ``"shadows"`` or ``"tnice"``, built with
                ``method_options``; or ``"simulation"``. ``"tnice"`` runs on
                exactly the same circuits, bases and shots as ``"shadows"`` and
                only replaces the fixed canonical-dual post-processing with the
                MPS-optimized estimator of :mod:`~mpstab.qpu.measurement.tnice`,
                so the two can be compared on identical measurement data.
                ``"simulation"`` is classical, see above and
                :meth:`_simulate_at_cut`: it needs no resynthesis, backend or
                shots, and ignores ``seed``.
            n_shots: a fixed shot budget. Exactly one of this and ``epsilon``
                for the hardware methods; neither for ``"simulation"``.
            epsilon: a target standard error, which sizes the shot budget from the
                per-route variance predictor. For ``"tnice"`` this reuses
                ``"shadows"``'s canonical-dual predictor, a conservative upper
                bound on the (unknown ahead of fitting) optimized variance.
            backend: anything with ``execute_circuits(circuits, nshots)``.
                Defaults to :class:`~mpstab.qpu.QiboSimulator`.
            max_bond_dimension: bond cap for every tail fold; ``-1`` means
                ``self.max_bond_dimension``. With ``"simulation"`` the head MPS
                is capped at ``self.max_bond_dimension``.
            seed: RNG seed for Pauli sampling, random shadow bases, and (for
                ``"tnice"``) the train/test shot split, forwarded to
                :func:`~mpstab.qpu.measurement.tnice.estimate_tnice`.
            method_options: options of a method given by name, e.g.
                ``n_string_samples`` for ``"pauli"`` or ``tail_handling`` for
                ``"shadows"``; see each method's class.

        Raises:
            ValueError: on an unknown ``method``, if neither or both of
                ``n_shots`` and ``epsilon`` are given for a hardware method, or
                if either is given with ``method="simulation"``.
            TypeError: on ``method_options`` the method does not take.
        """
        if method == "simulation":
            if n_shots is not None or epsilon is not None:
                raise ValueError(
                    "method='simulation' spends no shots; drop n_shots and epsilon."
                )
            if method_options:
                raise TypeError(
                    "method='simulation' takes no options; got "
                    f"{sorted(method_options)}."
                )
            if max_bond_dimension == -1:
                max_bond_dimension = self.max_bond_dimension
            constant, terms = pauli_terms(observable, self.nqubits)
            return self._simulate_at_cut(terms, constant, cut_index, max_bond_dimension)

        method = resolve_method(method, **method_options)
        plan = self.plan(
            observable,
            cut_index,
            method=method,
            n_shots=n_shots,
            epsilon=epsilon,
            max_bond_dimension=max_bond_dimension,
            seed=seed,
            out_folder=False,
        )
        frequencies = execute_plan(backend or QiboSimulator(), plan)
        return method.estimate(plan, frequencies, seed=seed)

    def plan(
        self,
        observable: Observable,
        cut_index: int,
        method: Union[str, MeasurementMethod] = "pauli",
        n_shots: Union[int, None] = None,
        epsilon: Union[float, None] = None,
        max_bond_dimension: Union[int, None] = -1,
        seed: Union[int, None] = None,
        out_folder=None,
        **method_options,
    ) -> MeasurementPlan:
        """
        Build the measurement circuits for ``observable`` at ``cut_index``,
        without running them, and save them as OpenQASM unless ``out_folder``
        is ``False``.

        This is the first of the three stages :meth:`expectation_at_cut` chains,
        exposed so the circuits can run anywhere, e.g. on a real device: run the
        saved circuits, then pass the counts and the returned plan to the
        method's ``estimate`` (or :func:`~mpstab.qpu.measurement.estimate`).

        Only the circuits and shots are saved, not the returned plan, which
        estimation needs. Building it
        again takes the same model, the same arguments and a fixed ``seed``.
        Even then the residual's gates can differ, so the saved circuits are
        the ones to run: never replace them with those of a rebuilt plan.

        Args:
            observable: as in :meth:`expectation_at_cut`.
            cut_index: as in :meth:`expectation_at_cut`.
            method: a :class:`~mpstab.qpu.measurement.abstract.MeasurementMethod`
                or the name of one, as in :meth:`expectation_at_cut` (but not
                ``"simulation"``).
            n_shots: as in :meth:`expectation_at_cut`.
            epsilon: as in :meth:`expectation_at_cut`.
            max_bond_dimension: as in :meth:`expectation_at_cut`.
            seed: as in :meth:`expectation_at_cut`.
            out_folder: where to save the circuits and shots, see
                :func:`~mpstab.qpu.execution.storage.save_circuits`.
                ``None`` (the default) saves them to a new folder
                ``planned_runs/plan_<method>_cut<cut_index>_<timestamp>``
                under the current directory and prints its path; ``False``
                saves nothing.
            method_options: as in :meth:`expectation_at_cut`.

        Returns:
            A :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan`.

        Raises:
            ValueError: on an unknown ``method``, if neither or both of
                ``n_shots`` and ``epsilon`` are given, or if the circuits are
                saved (``out_folder`` is not ``False``) without a ``seed``.

        """
        if method == "simulation":
            raise ValueError(
                "method='simulation' builds no circuits; use expectation_at_cut."
            )
        method = resolve_method(method, **method_options)
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
        if self.light_cone:
            resynthesis = self.resynthesize_head(cut_index, observable=observable)
        else:
            resynthesis = self.resynthesize_head(cut_index)

        plan = method.build_plan(
            self,
            terms=terms,
            constant=constant,
            cut_index=cut_index,
            resynthesis=resynthesis,
            max_bond_dimension=max_bond_dimension,
            n_shots=n_shots,
            epsilon=epsilon,
            seed=seed,
        )
        if out_folder is None:
            # Local time with its UTC offset; microseconds, so plans built in
            # a loop do not collide.
            timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f%z")
            out_folder = f"planned_runs/plan_{method.name}_cut{cut_index}_{timestamp}"
            save_circuits(plan, out_folder)
            print(f"Circuits saved to {out_folder}")
        elif out_folder is not False:
            save_circuits(plan, out_folder)
        return plan


@lru_cache(maxsize=256)
def _light_cone(generators: Tuple[str, ...], observable: str) -> frozenset:
    """
    Indices of the Pauli rotations ``generators`` (in circuit order) that a
    Pauli ``observable`` measured after them depends on: those anticommuting
    with the observable or with a later rotation that is itself kept.
    """

    def masks(pauli):
        x = sum(1 << q for q, label in enumerate(pauli) if label in "XY")
        z = sum(1 << q for q, label in enumerate(pauli) if label in "ZY")
        return x, z

    kept = [masks(observable)]
    relevant = set()
    for index in range(len(generators) - 1, -1, -1):
        x, z = masks(generators[index])
        if any(
            ((x & other_z).bit_count() + (z & other_x).bit_count()) & 1
            for other_x, other_z in kept
        ):
            kept.append((x, z))
            relevant.add(index)
    return frozenset(relevant)


@lru_cache(maxsize=32)
def _resynthesize(
    rotations: Tuple[Tuple[str, float], ...],
    nqubits: int,
    cut_index: int,
    synthesis: HeadSynthesis,
) -> ResynthesizedHead:
    """
    The body of :meth:`HSynthSMPO.resynthesize_head`, cached on its arguments.

    The key holds the head's dressed rotations and the (hashable) synthesis
    rather than the evolutor, so a repeated call with the same rotations returns
    the same :class:`ResynthesizedHead` object without synthesising again, and
    changing the circuit's parameters changes the angles, hence the key: the
    cache never serves a stale head and needs no invalidation.
    """
    paulis = [pauli for pauli, _ in rotations]
    angles = [angle for _, angle in rotations]
    try:
        head, residual_tableau, residual_gates = synthesis.synthesize(paulis, angles)
    except ImportError:
        if not isinstance(synthesis, RustiqSynthesis):
            raise
        warnings.warn(
            "rustiq is not installed; falling back to the CNOT-ladder "
            "resynthesis (no gate-count saving). Install it with pip install "
            '"mpstab[rustiq]" for the optimized head, or use '
            'head_synthesis="extraction", which needs no extra dependency.',
            stacklevel=3,
        )
        synthesis = CnotLadderSynthesis()
        head, residual_tableau, residual_gates = synthesis.synthesize(paulis, angles)

    if len(residual_tableau) != nqubits:  # degenerate cut_index == 0 case
        residual_tableau = stim.Tableau(nqubits)

    return ResynthesizedHead(
        circuit=head_to_qibo_circuit(head, nqubits),
        residual_tableau=residual_tableau,
        residual_gates=tuple(residual_gates),
        n_gates=len(head),
        n_two_qubit_gates=count_two_qubit_gates(head),
        cut_index=cut_index,
        method=synthesis.name,
    )
