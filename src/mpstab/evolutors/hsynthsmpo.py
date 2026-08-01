"""
Head/tail split of :class:`HSMPO`'s dressed rotations, with a sampled (never
exact) measurement of the head.

**Design invariant**: :class:`HSynthSMPO` never simulates the head exactly for
a *measurement* -- that is :class:`HSMPO`'s job already. The only two
measurement routes are ``"pauli"`` and ``"shadows"`` (see
:meth:`HSynthSMPO.expectation_at_cut`), and both always consume a finite shot
budget: there is no ``method="exact"``. :meth:`HSynthSMPO.expectation_from_split`
remains as a reference-only exact MPO-MPS contraction, for validating the
sampled routes and for tests, and is not reachable from
:meth:`HSynthSMPO.expectation_at_cut`.

Circuits are decided classically before anything touches a backend
(:mod:`~mpstab.evolutors.quantum_hardware.plan`); a backend (a real device, or
the default :class:`~mpstab.evolutors.quantum_hardware.plan.QiboSimulator`) is
one duck-typed ``execute_circuits(circuits, nshots)`` method returning
frequency dictionaries, exactly what qibo's own ``Circuit.__call__(...).frequencies()``
produces; :mod:`~mpstab.evolutors.quantum_hardware.estimate` turns those into
an :class:`~mpstab.evolutors.quantum_hardware.estimate.ExpectationResult`.

Resynthesizing the head for hardware (:meth:`HSynthSMPO.resynthesize_head`)
introduces a subtlety the ``"pauli"`` route corrects for: a device running the
resynthesized head prepares ``U(head)|psi_0>``, not the exact head state the
tail-folded coefficients were derived against. By Eq. (8) the two differ by
the resynthesis Clifford residual ``U(tail)``, so every sampled Pauli string
is conjugated through it
(:func:`~mpstab.evolutors.quantum_hardware.tail.fold_pool_through_tableau`)
before QWC grouping (conjugation does not preserve qubit-wise commutativity,
so folding after grouping would silently break the settings). The ``"shadows"``
route cannot fold that residual the same way (it would grow the tail MPO's
bond dimension and turn the single-qubit basis Paulis into non-local strings,
destroying the product structure its ``O(n chi^2)`` contraction relies on), so
it instead raises unless the caller passes ``tail_handling="append"``, which
runs the residual as extra gates after the head.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import stim
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.engines import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.quantum_hardware import (
    ExpectationResult,
    build_naive_head_and_residual,
    build_pauli_plan,
    build_shadow_plan,
    estimate,
    fold_pool_through_tableau,
    head_to_qibo_circuit,
    pool_pauli_terms,
    sample_pauli_strings,
    truncation_error_estimate,
)
from mpstab.evolutors.quantum_hardware.plan import QiboSimulator
from mpstab.evolutors.quantum_hardware.rustiq_synthesis import build_head_and_residual
from mpstab.evolutors.utils import dressed_rotations, validate_pauli_observable

__all__ = ["HSynthSMPO", "ExpectationResult", "ResynthesizedHead", "TailTruncation"]


def _hamiltonian_terms(
    observable: Union[str, SymbolicHamiltonian], nqubits: int
) -> Tuple[float, dict]:
    """Normalize a Pauli-string or qibo ``SymbolicHamiltonian`` observable into
    a constant shift plus ``{full_pauli_string: coefficient}``."""
    if isinstance(observable, str):
        validate_pauli_observable(observable, nqubits)
        return 0.0, {observable: 1.0}
    if isinstance(observable, SymbolicHamiltonian):
        coeffs, pauli_names, target_qubits = observable.simple_terms
        terms: dict = {}
        for coeff, names, qubits in zip(coeffs, pauli_names, target_qubits):
            labels = ["I"] * nqubits
            for name, qubit in zip(names, qubits):
                labels[qubit] = name
            string = "".join(labels)
            terms[string] = terms.get(string, 0.0) + coeff.real
        return observable.constant.real, terms
    raise ValueError(
        f"Given observable of type {type(observable)}, expected a Pauli string "
        "or a qibo SymbolicHamiltonian."
    )


def _aggregate_truncation(term_diagnostics: list) -> Tuple[float, float]:
    """Combine per-term ``(|coeff|, l1, l2)`` truncation estimates: triangle
    inequality for L1 (rigorous for any combination of terms), quadrature for
    L2 (assume-independent-errors, as :func:`estimate_shadows` also does)."""
    if not term_diagnostics:
        return 0.0, 0.0
    l1 = sum(weight * l1_term for weight, l1_term, _ in term_diagnostics)
    l2 = float(
        np.sqrt(sum((weight * l2_term) ** 2 for weight, _, l2_term in term_diagnostics))
    )
    return l1, l2


@dataclass(frozen=True)
class ResynthesizedHead:
    """A head circuit resynthesized for hardware, plus its Clifford residual.
    Returned by :meth:`HSynthSMPO.resynthesize_head`."""

    circuit: object  #: runnable qibo ``Circuit``, no measurement gates yet
    tail_tableau: object  #: ``stim.Tableau``, the Clifford residual of Eq. (8)
    tail_gates: (
        tuple  #: the same residual as a gate list, for ``tail_handling="append"``
    )
    n_gates: int
    n_two_qubit_gates: int
    cut_index: int
    method: str  #: ``"rustiq"`` or ``"naive"``


@dataclass(frozen=True)
class TailTruncation:
    """
    How much the tail-MPO bond-dimension truncation approximates the exact
    fold, cheapest view first. Returned by :meth:`HSynthSMPO.tail_truncation`.

    Attributes:
        fidelity_estimate: ``||O_approx||_F**2 / 2**n``, a norm-ratio estimate
            needing only the working truncation: no exact reference is built.
        relative_frobenius_error: ``||O_approx - O_exact||_F / ||O_exact||_F``,
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
    HSMPO variant that splits the dressed-rotation chain at a cut index and
    measures the head with a finite shot budget. See the module docstring for
    the design invariant.

    Only :class:`~mpstab.engines.QuimbEngine` is supported for the MPO tail.
    """

    def _dressed_rotations(self) -> List[Tuple[str, float]]:
        """Every magic gate's ``(generator, angle)`` in circuit order; see
        :func:`mpstab.evolutors.utils.dressed_rotations`."""
        return dressed_rotations(
            self.nqubits,
            self.stab_engine,
            self.magic_gates,
            self.clifford_circuit,
            self._gate_angle,
        )

    def _build_state_mps(self, head: List[Tuple[str, float]]):
        """Build the state MPS by applying the head dressed rotations directly
        (exactly), as :meth:`HSMPO._precompute_original_mps` does for the full circuit.
        """
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

    def tail_operator(
        self, observable: str, cut_index: int, max_bond_dimension: int = -1
    ):
        """
        The tail-folded observable MPO and its sign.

        Folds the tail rotations ``[cut_index:]`` into ``observable`` (after
        backpropagating it through the base circuit's Clifford part) via
        :meth:`~mpstab.engines.QuimbEngine.conjugate_operator`.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: split point; only the tail ``[cut_index:]`` is folded.
            max_bond_dimension: bond cap for each conjugation. ``-1`` (default)
                means ``self.max_bond_dimension``; ``None`` is untruncated.

        Returns:
            ``(operator, sign)``.
        """
        validate_pauli_observable(observable, self.nqubits)
        if max_bond_dimension == -1:
            max_bond_dimension = self.max_bond_dimension

        tail = self._dressed_rotations()[cut_index:]
        backprop_observable, sign = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=self.clifford_circuit
        )
        operator = self.tn_engine.pauli_mpo(backprop_observable)
        for generator, angle in reversed(tail):  # applied outermost-last
            operator = self.tn_engine.conjugate_operator(
                operator, generator, angle, max_bond_dimension
            )
        return operator, sign

    def expectation_from_split(
        self, observable: str, cut_index: int, return_fidelity: bool = False
    ):
        """
        Exact MPO-MPS contraction from splitting the dressed rotations at
        ``cut_index``: the head is applied exactly to the state MPS, the tail
        is folded into the observable via :meth:`tail_operator`.

        **Reference-only diagnostic, not a measurement route**: use this to
        validate :meth:`expectation_at_cut`'s sampled routes, or in tests where
        an exact ground truth is needed; it is not called from
        :meth:`expectation_at_cut`.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: number of leading dressed rotations applied to the
                state MPS; the rest are folded into the observable.
            return_fidelity: if ``True``, also return the state MPS's squared norm.

        Returns:
            The (real) expectation value, or ``(expval, fidelity)``.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "expectation_from_split requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )
        validate_pauli_observable(observable, self.nqubits)

        state_mps = self._build_state_mps(self._dressed_rotations()[:cut_index])
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
        Quantify how much the bond-dimension truncation approximates the exact
        tail fold; see :class:`TailTruncation` for the three fields.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: the split point (see :meth:`expectation_from_split`).
            reference_max_bond: bond cap for the reference operator when
                ``exact=True``; ``None`` (default) is untruncated.
            exact: if ``False``, skip the reference fold and report only the
                cheap, reference-free ``fidelity_estimate``; ``True`` (default)
                also builds the reference and reports the other two fields.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "tail_truncation requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )
        validate_pauli_observable(observable, self.nqubits)

        operator_approx, sign = self.tail_operator(
            observable, cut_index, self.max_bond_dimension
        )
        fidelity_estimate = (
            float(np.real(operator_approx.norm())) ** 2 / 2**self.nqubits
        )
        if not exact:
            return TailTruncation(fidelity_estimate, None, None)

        state_mps = self._build_state_mps(self._dressed_rotations()[:cut_index])
        operator_exact, _ = self.tail_operator(
            observable, cut_index, reference_max_bond
        )
        exact_norm = float(np.real(operator_exact.norm()))
        difference_norm = float(np.real((operator_approx - operator_exact).norm()))

        expval_approx = (
            np.real(
                self.tn_engine.expval(state_circuit=state_mps, operator=operator_approx)
            )
            * sign
        )
        expval_exact = (
            np.real(
                self.tn_engine.expval(state_circuit=state_mps, operator=operator_exact)
            )
            * sign
        )

        return TailTruncation(
            fidelity_estimate=fidelity_estimate,
            relative_frobenius_error=(
                difference_norm / exact_norm if exact_norm != 0 else 0.0
            ),
            expval_abs_error=abs(expval_approx - expval_exact),
        )

    def resynthesize_head(
        self, cut_index: int, metric_name: str = "count", preserve_order: bool = True
    ) -> ResynthesizedHead:
        """
        Resynthesize the head rotations ``[0:cut_index)`` into a runnable,
        hardware-native circuit, splitting off a pure-Clifford residual tail.

        Tries the low-level rustiq Pauli-network API first (see
        :mod:`mpstab.evolutors.quantum_hardware.rustiq_synthesis`); if
        ``rustiq`` is not installed (optional, not on PyPI -- ``pip install
        "mpstab[rustiq]"``), falls back to the dependency-free CNOT-ladder
        decomposition of
        :func:`~mpstab.evolutors.quantum_hardware.naive_synthesis.build_naive_head_and_residual`
        (identity residual, no gate-count saving) and warns once rather than
        raising, since the hardware path must keep working without the
        optional dependency.
        """
        dressed = self._dressed_rotations()[:cut_index]
        paulis = [p for p, _ in dressed]
        angles = [a for _, a in dressed]
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
        circuit = head_to_qibo_circuit(head, self.nqubits)
        n_two_qubit = sum(1 for entry in head if len(entry) == 2 and len(entry[1]) == 2)
        return ResynthesizedHead(
            circuit=circuit,
            tail_tableau=tail_tableau,
            tail_gates=tuple(tail_gates),
            n_gates=len(head),
            n_two_qubit_gates=n_two_qubit,
            cut_index=cut_index,
            method=method,
        )

    def _plan_pauli(
        self,
        terms,
        constant,
        cut_index,
        resynth,
        max_bond_dimension,
        n_shots,
        epsilon,
        seed,
        n_string_samples=200,
    ):
        rng = np.random.default_rng(seed)
        pool_input, term_diagnostics = [], []
        for label, coeff in terms.items():
            operator, sign = self.tail_operator(label, cut_index, max_bond_dimension)
            ensemble = sample_pauli_strings(
                operator,
                n_samples=n_string_samples,
                seed=int(rng.integers(0, 2**32 - 1)),
            )
            pool_input.append((coeff * sign, ensemble))
            l1, l2 = truncation_error_estimate(ensemble)
            term_diagnostics.append((abs(coeff), l1, l2))
        pooled = pool_pauli_terms(pool_input)

        if resynth.tail_tableau != stim.Tableau(self.nqubits):
            pooled = fold_pool_through_tableau(
                pooled, resynth.tail_tableau, self.stab_engine
            )

        coefficients = {label: float(np.real(c)) for label, c in pooled.items()}
        truncation_l1, truncation_l2 = _aggregate_truncation(term_diagnostics)
        return build_pauli_plan(
            resynth.circuit,
            self.nqubits,
            coefficients,
            n_shots,
            epsilon,
            constant=constant,
            truncation_l1=truncation_l1,
            truncation_l2=truncation_l2,
        )

    def _plan_shadows(
        self,
        terms,
        constant,
        cut_index,
        resynth,
        max_bond_dimension,
        n_shots,
        epsilon,
        seed,
        tail_handling,
        shots_per_setting=1,
    ):
        tableau_trivial = resynth.tail_tableau == stim.Tableau(self.nqubits)
        circuit = resynth.circuit
        if not tableau_trivial:
            if tail_handling == "append":
                circuit = circuit + head_to_qibo_circuit(
                    list(resynth.tail_gates), self.nqubits
                )
            elif tail_handling == "forbid":
                raise ValueError(
                    f"resynthesize_head left a non-trivial Clifford tail ({resynth.method} "
                    f"resynthesis at cut_index={cut_index}); the shadows route cannot fold it "
                    "into the tail MPO without destroying the product structure its O(n chi^2) "
                    "contraction relies on. Pass tail_handling='append' to run the residual as "
                    "extra gates instead."
                )
            else:
                raise ValueError(
                    f"Unknown tail_handling {tail_handling!r}, expected 'forbid' or 'append'."
                )

        mpo_terms = []
        for label, coeff in terms.items():
            operator, sign = self.tail_operator(label, cut_index, max_bond_dimension)
            mpo_terms.append((label, coeff, sign, operator))

        truncation_l2 = 0.0
        if max_bond_dimension is not None:
            truncation_l2 = sum(
                abs(coeff)
                * self.tail_truncation(
                    label, cut_index, reference_max_bond=None, exact=True
                ).expval_abs_error
                for label, coeff in terms.items()
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

    def expectation_at_cut(
        self,
        observable: Union[str, SymbolicHamiltonian],
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
        Resynthesize the head, build the measurement circuits, run them on
        ``backend`` and turn the result into an :class:`ExpectationResult`,
        all in one call. There is no ``method="exact"``: exactly one of
        ``n_shots``/``epsilon`` is required.

        Args:
            observable: Pauli string (qubit-0-leftmost) or ``SymbolicHamiltonian``.
                For a Hamiltonian, one tail MPO is folded per term but a
                single set of circuits is emitted (the head does not depend
                on the observable), and Pauli strings shared between terms
                are pooled before grouping so each is measured once with the
                summed coefficient.
            cut_index: the head/tail split point.
            method: ``"pauli"`` or ``"shadows"``.
            n_shots: fixed shot budget.
            epsilon: target standard error; sizes the shot budget via the
                corrected per-route predictor (Neyman-optimal for ``"pauli"``,
                see :meth:`~mpstab.evolutors.quantum_hardware.plan.PauliMeasurementPlan.shots_for_precision`;
                exact :func:`~mpstab.evolutors.quantum_hardware.shadow_variance_from_mpo`
                for ``"shadows"``), never the ``||c||_1**2`` worst case, which
                can rank the two routes backwards.
            backend: anything with an ``execute_circuits(circuits, nshots)``
                method; defaults to :class:`~mpstab.evolutors.quantum_hardware.plan.QiboSimulator`.
            max_bond_dimension: bond cap for every tail fold; ``-1`` means
                ``self.max_bond_dimension``.
            tail_handling: ``"shadows"``-only. ``"forbid"`` (default) raises if
                resynthesis left a non-trivial Clifford tail; ``"append"``
                instead runs the residual as extra gates after the head.
            seed: RNG seed for Pauli-string sampling / random shadow bases.
            **method_kwargs: ``"pauli"`` accepts ``n_string_samples`` (default
                200); ``"shadows"`` accepts ``shots_per_setting`` (default 1).
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "expectation_at_cut requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )
        if (n_shots is None) == (epsilon is None):
            raise ValueError("Exactly one of n_shots or epsilon must be given.")
        if method not in ("pauli", "shadows"):
            raise ValueError(
                f"Unknown method {method!r}, expected 'pauli' or 'shadows'."
            )
        if max_bond_dimension == -1:
            max_bond_dimension = self.max_bond_dimension

        constant, terms = _hamiltonian_terms(observable, self.nqubits)
        resynth = self.resynthesize_head(cut_index)

        if method == "pauli":
            plan = self._plan_pauli(
                terms,
                constant,
                cut_index,
                resynth,
                max_bond_dimension,
                n_shots,
                epsilon,
                seed,
                **method_kwargs,
            )
        else:
            plan = self._plan_shadows(
                terms,
                constant,
                cut_index,
                resynth,
                max_bond_dimension,
                n_shots,
                epsilon,
                seed,
                tail_handling,
                **method_kwargs,
            )

        backend = backend or QiboSimulator()
        frequencies = [
            backend.execute_circuits([circuit], shots)[0]
            for circuit, shots in zip(plan.circuits, plan.shots)
        ]
        return estimate(plan, frequencies)
