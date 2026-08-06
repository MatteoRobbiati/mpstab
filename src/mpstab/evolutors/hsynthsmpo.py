"""
The head/tail split of an HSMPO's dressed rotations, measured with finite shots.

The dressed-rotation chain is cut at ``cut_index``. The head is resynthesised
into a circuit a device can run; the tail is folded into the observable, leaving
an MPO. :meth:`HSynthSMPO.expectation_at_cut` then measures the head against that
observable, always spending a shot budget, by one of two routes:

- ``"pauli"`` samples Pauli strings from the tail MPO and groups them into
  qubit-wise-commuting measurement settings.
- ``"shadows"`` measures random single-qubit Pauli bases and contracts each
  snapshot against the tail MPO directly.

Resynthesising the head leaves a Clifford residual, so a device prepares
``U(residual) U(head)|psi_0>`` rather than the exact head state the tail
coefficients were derived against. The ``"pauli"`` route corrects for this by
conjugating every sampled string through the residual. The ``"shadows"`` route
cannot: folding the residual into the tail MPO would destroy the product
structure its ``O(n chi^2)`` contraction relies on, so it raises unless the caller
passes ``tail_handling="append"`` to run the residual as extra gates.

:meth:`HSynthSMPO.expectation_from_split` is the exact MPO-MPS contraction of the
same split. It is a reference for validating the sampled routes, not a
measurement route, and :meth:`HSynthSMPO.expectation_at_cut` never calls it.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import stim

from mpstab.engines import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.pauli import validate_pauli_string
from mpstab.quantum_hardware import (
    ExpectationResult,
    QiboSimulator,
    build_head_and_residual,
    build_naive_head_and_residual,
    build_pauli_plan,
    build_shadow_plan,
    count_two_qubit_gates,
    estimate,
    fold_pool_through_tableau,
    head_to_qibo_circuit,
    pool_pauli_terms,
    sample_pauli_strings,
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
        tail_tableau: the Clifford residual as a ``stim.Tableau``.
        tail_gates: the same residual as a gate list, for ``tail_handling="append"``.
        n_gates: gates in the head.
        n_two_qubit_gates: two-qubit gates in the head.
        cut_index: the split point the head came from.
        method: ``"rustiq"`` or ``"naive"``.
    """

    circuit: object
    tail_tableau: object
    tail_gates: tuple
    n_gates: int
    n_two_qubit_gates: int
    cut_index: int
    method: str


@dataclass(frozen=True)
class TailTruncation:
    """
    How closely the truncated tail fold approximates the exact one, cheapest view
    first. Returned by :meth:`HSynthSMPO.tail_truncation`.

    Attributes:
        fidelity_estimate: ``||O_approx||_F**2 / 2**n``, a norm-ratio estimate
            needing only the working truncation, with no exact reference built.
        relative_frobenius_error: ``||O_approx - O_exact||_F / ||O_exact||_F``;
            ``None`` unless ``exact=True`` was requested.
        expval_abs_error: ``|<O_approx> - <O_exact>|`` on the same head state;
            ``None`` unless ``exact=True``.
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

    def tail_operator(
        self, observable: str, cut_index: int, max_bond_dimension: int = -1
    ):
        """
        The tail-folded observable MPO and its sign.

        Backpropagates ``observable`` through the circuit's Clifford part, then
        folds the tail rotations ``[cut_index:]`` into it.

        Args:
            observable: a Pauli string (qubit-0-leftmost).
            cut_index: the split point; only the tail is folded.
            max_bond_dimension: bond cap for each conjugation. ``-1`` means
                ``self.max_bond_dimension``, ``None`` means untruncated.

        Returns:
            ``(operator, sign)``.
        """
        validate_pauli_string(observable, self.nqubits)
        if max_bond_dimension == -1:
            max_bond_dimension = self.max_bond_dimension

        tail = self._dressed_rotations()[cut_index:]
        backpropagated, sign = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=self.clifford_circuit
        )
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

    def expectation_from_split(
        self, observable: str, cut_index: int, return_fidelity: bool = False
    ):
        """
        Exact expectation value from the split: the head applied to the state MPS,
        the tail folded into the observable.

        A reference for validating :meth:`expectation_at_cut`, not a measurement
        route.

        Args:
            observable: a Pauli string (qubit-0-leftmost).
            cut_index: the split point.
            return_fidelity: also return the state MPS's squared norm.

        Returns:
            The (real) expectation value, or ``(expval, fidelity)``.
        """
        self._require_quimb("expectation_from_split")
        validate_pauli_string(observable, self.nqubits)

        state_mps = self._head_state_mps(self._dressed_rotations()[:cut_index])
        operator, sign = self.tail_operator(observable, cut_index)
        expval = (
            np.real(self.tn_engine.expval(state_circuit=state_mps, operator=operator))
            * sign
        )

        if return_fidelity:
            return expval, state_mps.norm(squared=True)
        return expval

    def tail_truncation(
        self,
        observable: str,
        cut_index: int,
        reference_max_bond: Union[int, None] = None,
        exact: bool = True,
    ) -> TailTruncation:
        """
        Quantify how closely the truncated tail fold approximates the exact one.

        Args:
            observable: a Pauli string (qubit-0-leftmost).
            cut_index: the split point.
            reference_max_bond: bond cap for the reference fold when
                ``exact=True``; ``None`` means untruncated.
            exact: build the reference fold and report all three fields. When
                ``False``, report only the reference-free ``fidelity_estimate``.
        """
        self._require_quimb("tail_truncation")
        validate_pauli_string(observable, self.nqubits)

        operator_approx, sign = self.tail_operator(
            observable, cut_index, self.max_bond_dimension
        )
        fidelity_estimate = (
            float(np.real(operator_approx.norm())) ** 2 / 2**self.nqubits
        )
        if not exact:
            return TailTruncation(fidelity_estimate, None, None)

        state_mps = self._head_state_mps(self._dressed_rotations()[:cut_index])
        operator_exact, _ = self.tail_operator(
            observable, cut_index, reference_max_bond
        )
        exact_norm = float(np.real(operator_exact.norm()))
        difference_norm = float(np.real((operator_approx - operator_exact).norm()))

        def expval(operator):
            return (
                np.real(
                    self.tn_engine.expval(state_circuit=state_mps, operator=operator)
                )
                * sign
            )

        return TailTruncation(
            fidelity_estimate=fidelity_estimate,
            relative_frobenius_error=(
                difference_norm / exact_norm if exact_norm != 0 else 0.0
            ),
            expval_abs_error=abs(expval(operator_approx) - expval(operator_exact)),
        )

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

        Args:
            cut_index: the split point.
            metric_name: rustiq metric, ``"count"`` or ``"depth"``.
            preserve_order: keep the (non-commuting) rotation order.
        """
        dressed = self._dressed_rotations()[:cut_index]
        paulis = [pauli for pauli, _ in dressed]
        angles = [angle for _, angle in dressed]
        try:
            head, tail_tableau, tail_gates = build_head_and_residual(
                paulis, angles, metric_name=metric_name, preserve_order=preserve_order
            )
            method = "rustiq"
        except ImportError:
            import warnings

            warnings.warn(
                "rustiq is not installed; falling back to the naive CNOT-ladder "
                "resynthesis (no gate-count saving). Install it with pip install "
                '"mpstab[rustiq]" for the optimized head.',
                stacklevel=2,
            )
            head, tail_tableau, tail_gates = build_naive_head_and_residual(
                paulis, angles
            )
            method = "naive"

        if len(tail_tableau) != self.nqubits:  # degenerate cut_index == 0 case
            tail_tableau = stim.Tableau(self.nqubits)

        return ResynthesizedHead(
            circuit=head_to_qibo_circuit(head, self.nqubits),
            tail_tableau=tail_tableau,
            tail_gates=tuple(tail_gates),
            n_gates=len(head),
            n_two_qubit_gates=count_two_qubit_gates(head),
            cut_index=cut_index,
            method=method,
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
        tail_handling: str = "forbid",
        seed: Union[int, None] = None,
        **method_kwargs,
    ) -> ExpectationResult:
        """
        Resynthesise the head, build the measurement circuits, run them and
        recombine the results, in one call.

        Args:
            observable: a Pauli string (qubit-0-leftmost) or a
                ``SymbolicHamiltonian``. A Hamiltonian folds one tail MPO per term
                but emits a single set of circuits, since the head does not depend
                on the observable, and pools Pauli strings shared between terms so
                each is measured once with the summed coefficient.
            cut_index: the split point.
            method: ``"pauli"`` or ``"shadows"``.
            n_shots: a fixed shot budget. Exactly one of this and ``epsilon``.
            epsilon: a target standard error, which sizes the shot budget from the
                per-route variance predictor.
            backend: anything with ``execute_circuits(circuits, nshots)``.
                Defaults to :class:`~mpstab.quantum_hardware.QiboSimulator`.
            max_bond_dimension: bond cap for every tail fold; ``-1`` means
                ``self.max_bond_dimension``.
            tail_handling: ``"shadows"`` only. ``"forbid"`` raises if resynthesis
                left a non-trivial Clifford residual; ``"append"`` runs it as extra
                gates after the head.
            seed: RNG seed for Pauli sampling and random shadow bases.
            method_kwargs: ``"pauli"`` takes ``n_string_samples`` (default 200),
                ``"shadows"`` takes ``shots_per_setting`` (default 1).

        Raises:
            ValueError: on an unknown ``method``, or if neither or both of
                ``n_shots`` and ``epsilon`` are given.
        """
        self._require_quimb("expectation_at_cut")
        if (n_shots is None) == (epsilon is None):
            raise ValueError("Exactly one of n_shots or epsilon must be given.")
        if method not in ("pauli", "shadows"):
            raise ValueError(
                f"Unknown method {method!r}, expected 'pauli' or 'shadows'."
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
            **method_kwargs,
        )

        backend = backend or QiboSimulator()
        frequencies = [
            backend.execute_circuits([circuit], shots)[0]
            for circuit, shots in zip(plan.circuits, plan.shots)
        ]
        return estimate(plan, frequencies)

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
            operator, sign = self.tail_operator(pauli, cut_index, max_bond_dimension)
            ensemble = sample_pauli_strings(
                operator,
                n_samples=n_string_samples,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            ensembles.append((coefficient * sign, ensemble))
            l1, l2 = truncation_error_estimate(ensemble)
            diagnostics.append((abs(coefficient), l1, l2))

        pooled = pool_pauli_terms(ensembles)
        if resynthesis.tail_tableau != stim.Tableau(self.nqubits):
            pooled = fold_pool_through_tableau(
                pooled, resynthesis.tail_tableau, self.stab_engine
            )

        truncation_l1, truncation_l2 = _aggregate_truncation(diagnostics)
        return build_pauli_plan(
            resynthesis.circuit,
            self.nqubits,
            {pauli: float(np.real(c)) for pauli, c in pooled.items()},
            n_shots,
            epsilon,
            constant=constant,
            truncation_l1=truncation_l1,
            truncation_l2=truncation_l2,
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
        """Fold each term's tail MPO and pick random measurement bases."""
        circuit = self._shadow_circuit(resynthesis, cut_index, tail_handling)

        mpo_terms = []
        for pauli, coefficient in terms.items():
            operator, sign = self.tail_operator(pauli, cut_index, max_bond_dimension)
            mpo_terms.append((pauli, coefficient, sign, operator))

        truncation_l2 = 0.0
        if max_bond_dimension is not None:
            truncation_l2 = sum(
                abs(coefficient)
                * self.tail_truncation(
                    pauli, cut_index, reference_max_bond=None, exact=True
                ).expval_abs_error
                for pauli, coefficient in terms.items()
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

    def _shadow_circuit(self, resynthesis, cut_index, tail_handling):
        """
        The circuit the shadows route measures, handling a non-trivial residual.

        Raises:
            ValueError: on an unknown ``tail_handling``, or on a non-trivial
                residual under ``"forbid"``.
        """
        if tail_handling not in ("forbid", "append"):
            raise ValueError(
                f"Unknown tail_handling {tail_handling!r}, expected 'forbid' or "
                "'append'."
            )
        if resynthesis.tail_tableau == stim.Tableau(self.nqubits):
            return resynthesis.circuit
        if tail_handling == "forbid":
            raise ValueError(
                "resynthesize_head left a non-trivial Clifford residual "
                f"({resynthesis.method} resynthesis at cut_index={cut_index}). The "
                "shadows route cannot fold it into the tail MPO without destroying "
                "the product structure its O(n chi^2) contraction relies on. Pass "
                "tail_handling='append' to run the residual as extra gates instead."
            )
        return resynthesis.circuit + head_to_qibo_circuit(
            list(resynthesis.tail_gates), self.nqubits
        )

    def _require_quimb(self, method_name: str):
        """
        Raises:
            NotImplementedError: unless the tensor-network engine is a
                :class:`~mpstab.engines.QuimbEngine`.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                f"{method_name} requires QuimbEngine, the only engine that can "
                "conjugate an operator. Call set_engines(tn_engine=QuimbEngine())."
            )


def _aggregate_truncation(diagnostics: list) -> Tuple[float, float]:
    """
    Combine per-term ``(|coeff|, l1, l2)`` truncation estimates.

    L1 adds by the triangle inequality, rigorous for any combination of terms;
    L2 adds in quadrature, assuming the terms' errors are independent.
    """
    if not diagnostics:
        return 0.0, 0.0
    l1 = sum(weight * term_l1 for weight, term_l1, _ in diagnostics)
    l2 = float(
        np.sqrt(sum((weight * term_l2) ** 2 for weight, _, term_l2 in diagnostics))
    )
    return l1, l2
