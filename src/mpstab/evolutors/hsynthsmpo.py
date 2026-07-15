from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from mpstab.engines import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.quantum_hardware import (
    build_head_and_residual,
    head_counts_only,
    head_to_qibo_circuit,
)
from mpstab.evolutors.utils import dressed_rotations, validate_pauli_observable


def _pauli_expectation_from_state(pauli_str: str, state) -> complex:
    """Dense expectation of a Pauli string on a statevector, via qibo's SymbolicHamiltonian."""
    from qibo import symbols
    from qibo.hamiltonians import SymbolicHamiltonian

    form = 1
    for i, p in enumerate(pauli_str):
        form *= getattr(symbols, p)(i)
    ham = SymbolicHamiltonian(form=form, nqubits=len(pauli_str))
    return ham.expectation_from_state(state)


@dataclass
class HSynthSMPO(HSMPO):
    """
    HSMPO variant that can split the dressed-rotation chain at a cut index.

    The dressed Pauli rotations extracted by the base :class:`HSMPO` are split in
    two at ``cut_index``:

    - the *head* rotations ``[0:cut_index)`` (closer to the initial state) are
      applied directly to the state MPS, exactly as the base :class:`HSMPO` does
      (via :meth:`TensorNetworkEngine.pauli_rot`);
    - the *tail* rotations ``[cut_index:]`` (closer to the observable) are folded
      into the observable as an MPO by conjugating it through each rotation
      (Heisenberg picture), instead of touching the state.

    This trades two-qubit gate applications on the (truncation-sensitive) state
    MPS for MPO-MPO compression on the observable side.

    Separately -- and only for running on real hardware -- the head rotations
    can be *resynthesized* into an efficient native-gate circuit via the
    low-level rustiq Pauli-network API (:mod:`mpstab.evolutors.quantum_hardware`).
    That circuit is never used for the classical expectation (which applies the
    exact rotations to the MPS); it is exposed via :meth:`foldable_head_circuit`
    and measured by :meth:`foldable_head_gate_counts`. The pure-Clifford residual
    left over by the resynthesis is reabsorbed *exactly* into an observable via
    :meth:`fold_observable_through_tail` (see :meth:`expectation_from_rustiq_fold`).
    This resynthesis path is the only one that needs the optional ``rustiq``
    dependency (``pip install "mpstab[rustiq]"``).

    Only :class:`~mpstab.engines.QuimbEngine` is supported for the MPO tail.
    """

    def _dressed_rotations(self) -> List[Tuple[str, float]]:
        """
        Return every magic gate's ``(generator, angle)`` in circuit order.

        See :func:`mpstab.evolutors.utils.dressed_rotations` for the algorithm
        (a single-pass Clifford simulation, requires ``StimEngine``).
        """
        return dressed_rotations(
            self.nqubits,
            self.stab_engine,
            self.magic_gates,
            self.clifford_circuit,
            self._gate_angle,
        )

    def _build_state_mps(self, head: List[Tuple[str, float]]):
        """
        Build the state MPS by applying the head dressed rotations directly.

        Starts from the initial state and applies each ``(generator, angle)`` via
        :meth:`TensorNetworkEngine.pauli_rot`, exactly as
        :meth:`HSMPO._precompute_original_mps` does for the full circuit. No
        resynthesis is involved, so this needs no ``rustiq``.
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

    def _build_tail_operator(
        self,
        observable: str,
        tail: List[Tuple[str, float]],
        max_bond_dimension: int | None,
    ):
        """
        Build the observable MPO with the tail rotations folded in.

        Returns ``(operator, sign)`` with ``O`` the backpropagated observable and
        each tail rotation folded via :meth:`TensorNetworkEngine.conjugate_operator`
        (see that method for the fold convention). Each conjugation is compressed
        to ``max_bond_dimension`` (``None`` means no truncation / the exact
        operator).
        """
        backprop_observable, sign = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=self.clifford_circuit
        )
        operator = self.tn_engine.pauli_mpo(backprop_observable)

        # Applied outermost-last, so iterate the tail in reverse.
        for generator, angle in reversed(tail):
            operator = self.tn_engine.conjugate_operator(
                operator, generator, angle, max_bond_dimension
            )
        return operator, sign

    def expectation_from_split(
        self,
        observable: str,
        cut_index: int,
        return_fidelity: bool = False,
    ):
        """
        Compute the expectation value by splitting the dressed rotations at
        ``cut_index``: the head rotations are applied (exactly) to the state MPS,
        the tail rotations are folded into the observable as an MPO.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: Number of leading dressed rotations to apply to the state
                MPS. The remaining rotations are folded into the observable.
                ``0`` folds everything into the observable;
                ``len(magic_gates)`` applies everything to the state (equivalent
                to :meth:`HSMPO.expectation`).
            return_fidelity: If ``True``, also return the (squared) norm of the
                state MPS, as in :meth:`HSMPO.expectation`.

        Returns:
            The (real) expectation value, or ``(expval, fidelity)`` if
            ``return_fidelity`` is ``True``.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "expectation_from_split requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )

        validate_pauli_observable(observable, self.nqubits)

        dressed = self._dressed_rotations()
        head, tail = dressed[:cut_index], dressed[cut_index:]

        state_mps = self._build_state_mps(head)
        operator, sign = self._build_tail_operator(
            observable, tail, self.max_bond_dimension
        )

        expval = (
            np.real(self.tn_engine.expval(state_circuit=state_mps, operator=operator))
            * sign
        )

        if return_fidelity:
            return expval, state_mps.norm(squared=True)
        return expval

    def mpo_tail_approximation(
        self,
        observable: str,
        cut_index: int,
        reference_max_bond: int | None = None,
    ) -> dict:
        """
        Quantify how much the bond-dimension truncation approximates the MPO tail.

        The tail rotations ``[cut_index:]`` are folded into the observable twice:
        once with the working truncation (``self.max_bond_dimension``) and once
        with a reference cap (``reference_max_bond``, ``None`` = untruncated /
        exact). Both an operator-level and an expectation-level error are
        reported. The (shared) state MPS is built exactly from the head rotations,
        so the reported errors isolate the *MPO-tail* truncation only.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: The split point (see :meth:`expectation_from_split`).
            reference_max_bond: Bond dimension for the reference operator. ``None``
                (default) builds the exact, untruncated tail MPO. Note the exact
                MPO bond dimension can grow quickly with the tail length.

        Returns:
            dict with:
              - ``relative_frobenius_error``:
                ``||O_approx - O_exact||_F / ||O_exact||_F`` for the folded tail
                operator (0 means the truncation was lossless).
              - ``absolute_frobenius_error``: ``||O_approx - O_exact||_F``.
              - ``expval_approx`` / ``expval_reference``: expectation values on
                the (same) state MPS, using each operator.
              - ``expval_abs_error``: ``|expval_approx - expval_reference|``.
              - ``approx_max_bond`` / ``reference_max_bond``: reached MPO bond
                dimensions.
              - ``max_bond_dimension``: the working cap used for ``O_approx``.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "mpo_tail_approximation requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )

        validate_pauli_observable(observable, self.nqubits)

        dressed = self._dressed_rotations()
        head, tail = dressed[:cut_index], dressed[cut_index:]

        state_mps = self._build_state_mps(head)

        operator_approx, sign = self._build_tail_operator(
            observable, tail, self.max_bond_dimension
        )
        operator_exact, _ = self._build_tail_operator(
            observable, tail, reference_max_bond
        )

        exact_norm = float(np.real(operator_exact.norm()))
        difference_norm = float(np.real((operator_approx - operator_exact).norm()))

        expval_approx = (
            np.real(
                self.tn_engine.expval(state_circuit=state_mps, operator=operator_approx)
            )
            * sign
        )
        expval_reference = (
            np.real(
                self.tn_engine.expval(state_circuit=state_mps, operator=operator_exact)
            )
            * sign
        )

        return {
            "relative_frobenius_error": (
                difference_norm / exact_norm if exact_norm != 0 else 0.0
            ),
            "absolute_frobenius_error": difference_norm,
            "expval_approx": expval_approx,
            "expval_reference": expval_reference,
            "expval_abs_error": abs(expval_approx - expval_reference),
            "approx_max_bond": operator_approx.max_bond(),
            "reference_max_bond": operator_exact.max_bond(),
            "max_bond_dimension": self.max_bond_dimension,
        }

    def foldable_head_and_tail(
        self,
        cut_index: int,
        metric_name: str = "count",
        preserve_order: bool = True,
    ):
        """
        Resynthesize the head rotations ``[0:cut_index)`` with the low-level
        rustiq Pauli-network API, splitting off a pure-Clifford residual tail.

        Uses ``rustiq.pauli_network_synthesis`` directly (see the module
        docstring of :mod:`mpstab.evolutors.quantum_hardware.rustiq_synthesis`
        for why): the residual is provably Clifford and can be reabsorbed
        *exactly* into an observable via :meth:`fold_observable_through_tail`
        -- no tensor-network truncation involved. Requires the optional
        ``rustiq`` package (``pip install "mpstab[rustiq]"``).

        Args:
            cut_index: Number of leading dressed rotations to resynthesize.
            metric_name: rustiq metric, ``"count"`` or ``"depth"``.
            preserve_order: Whether rustiq must preserve the rotation order.

        Returns:
            ``(head, tail_tableau, tail_gates)``, see
            :func:`mpstab.evolutors.quantum_hardware.build_head_and_residual`.
        """
        dressed = self._dressed_rotations()[:cut_index]
        paulis = [p for p, _ in dressed]
        angles = [a for _, a in dressed]
        return build_head_and_residual(
            paulis, angles, metric_name=metric_name, preserve_order=preserve_order
        )

    def foldable_head_gate_counts(
        self,
        cut_index: int,
        metric_name: str = "count",
        preserve_order: bool = True,
    ) -> dict:
        """
        Report a two-qubit-gate resource comparison for a given ``cut_index``.

        Only the head rotations ``[0:cut_index)`` are resynthesized, and only
        the cheap rustiq skeleton is computed (no rotation placement -- a
        single rustiq call; see
        :func:`mpstab.evolutors.quantum_hardware.head_counts_only`). The
        reported ``original_circuit_2q_gates`` counts the *full* original
        ansatz circuit, so it is directly comparable to
        ``synthesized_head_2q_gates`` only at ``cut_index == len(magic_gates)``
        (fully resynthesized vs. fully original); for partial cuts it is
        provided as context.
        """
        dressed = self._dressed_rotations()
        paulis = [p for p, _ in dressed[:cut_index]]
        total, two_q = head_counts_only(
            paulis, metric_name=metric_name, preserve_order=preserve_order
        )
        original_2q = sum(
            1
            for gate in self.ansatz.circuit.queue
            if len(set(gate.target_qubits) | set(gate.control_qubits)) == 2
        )

        return {
            "n_head_rotations": cut_index,
            "n_tail_rotations": len(dressed) - cut_index,
            "synthesized_head_total_gates": total,
            "synthesized_head_2q_gates": two_q,
            "original_circuit_2q_gates": original_2q,
        }

    def foldable_head_circuit(
        self,
        cut_index: int,
        metric_name: str = "count",
        preserve_order: bool = True,
    ):
        """Resynthesized head ``[0:cut_index)`` as a runnable qibo Circuit."""
        head, _, _ = self.foldable_head_and_tail(
            cut_index, metric_name=metric_name, preserve_order=preserve_order
        )
        return head_to_qibo_circuit(head, self.nqubits)

    def fold_observable_through_tail(
        self,
        observable: str,
        tail_tableau,
        sign: float = 1.0,
    ):
        """
        Reabsorb a foldable-resynthesis tail tableau into ``observable``, via
        :meth:`~mpstab.engines.StimEngine.fold_pauli_through_tableau`.
        """
        return self.stab_engine.fold_pauli_through_tableau(
            observable, tail_tableau, sign
        )

    def expectation_from_rustiq_fold(
        self,
        observable: str,
        metric_name: str = "count",
        preserve_order: bool = True,
    ) -> float:
        """
        Exact expectation value using the foldable rustiq resynthesis of *all*
        dressed rotations.

        All dressed rotations are resynthesized into hardware-native gates
        (see :meth:`foldable_head_and_tail`, called here with
        ``cut_index == len(magic_gates)``); the resulting pure-Clifford tail --
        and the base circuit's Clifford part -- are reabsorbed *exactly* into
        ``observable`` via :class:`~mpstab.engines.StimEngine` (no
        MPO/tensor-network truncation is involved, unlike
        :meth:`mpo_tail_approximation`). This only works exactly because *all*
        rotations are resynthesized together, leaving no non-Clifford
        remainder: an intermediate cut (as in :meth:`expectation_from_split`)
        would leave a tail rotation whose Heisenberg conjugation is generally
        *not* a single Pauli, which is exactly why :meth:`mpo_tail_approximation`
        needs a truncated tensor-network operator instead of a stim fold.

        The head circuit is run with qibo's statevector simulator, so this is
        intended for verification at small/moderate qubit counts, not the
        exponentially large systems the MPS-based methods target. Requires the
        optional ``rustiq`` package.
        """
        validate_pauli_observable(observable, self.nqubits)

        backprop_observable, sign = self.stab_engine.backpropagate(
            observable=observable, clifford_circuit=self.clifford_circuit
        )
        n_dressed = len(self.magic_gates)
        head, tail_tableau, _ = self.foldable_head_and_tail(
            n_dressed, metric_name=metric_name, preserve_order=preserve_order
        )
        folded_observable, fold_sign = self.fold_observable_through_tail(
            backprop_observable, tail_tableau
        )

        head_circuit = head_to_qibo_circuit(head, self.nqubits)
        state = head_circuit().state()

        expval = _pauli_expectation_from_state(folded_observable, state)
        return float(np.real(expval)) * sign * fold_sign

    def _mpo_tail_fidelity(self, observable: str, cut_index: int) -> float:
        """
        Cheap quimb-``fidelity_estimate``-style norm-ratio estimate of the
        MPO-tail truncation, WITHOUT an exact reference (mirrors quimb's
        ``CircuitMPS.fidelity_estimate``, which just reads off the retained norm
        of the un-renormalized truncated state).

        Heisenberg conjugation R^dag O R is Frobenius-norm preserving, so with a
        bond-``self.max_bond_dimension`` truncation the retained operator norm
        drops; the ratio ``||O_tail||_F^2 / ||O||_F^2`` estimates the truncation
        fidelity. Much cheaper than :meth:`mpo_tail_approximation` (one
        truncated fold, no exact/untruncated reference is computed).

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: The split point (see :meth:`expectation_from_split`);
                only the tail ``[cut_index:]`` is folded.

        Returns:
            The estimated fidelity, in ``[0, 1]``.
        """
        if not isinstance(self.tn_engine, QuimbEngine):
            raise NotImplementedError(
                "_mpo_tail_fidelity requires QuimbEngine. Call "
                "set_engines(tn_engine=QuimbEngine()) to enable it."
            )

        validate_pauli_observable(observable, self.nqubits)

        tail = self._dressed_rotations()[cut_index:]
        operator, _ = self._build_tail_operator(
            observable, tail, self.max_bond_dimension
        )
        norm_init_sq = 2 ** self.nqubits
        return float(np.real(operator.norm())) ** 2 / norm_init_sq
