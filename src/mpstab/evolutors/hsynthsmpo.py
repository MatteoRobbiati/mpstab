from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from mpstab.engines import QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.evolutors.transpilation import (
    count_two_qubit_gates_qibo,
    count_two_qubit_gates_qiskit,
    qiskit_circuit_to_qibo,
    synthesize_pauli_rotations,
)
from mpstab.evolutors.utils import validate_pauli_observable


@dataclass
class HSynthSMPO(HSMPO):
    """
    HSMPO variant that can split the dressed-rotation chain at a cut index.

    The dressed Pauli rotations extracted by the base :class:`HSMPO` are split in
    two at ``cut_index``:

    - the *head* rotations ``[0:cut_index)`` (closer to the initial state) are
      resynthesized into a native-gate circuit via Qiskit's Rustiq high-level
      synthesis and applied exactly to build the state MPS;
    - the *tail* rotations ``[cut_index:]`` (closer to the observable) are folded
      into the observable as an MPO by conjugating it through each rotation
      (Heisenberg picture), instead of touching the state.

    This trades two-qubit gate applications on the (truncation-sensitive) state
    MPS for MPO-MPO compression on the observable side. It also exposes a
    :meth:`count_two_qubit_gates` utility to compare the resynthesized head with
    the original circuit.

    Only :class:`~mpstab.engines.QuimbEngine` is supported.
    """

    def _dressed_rotations(self) -> List[Tuple[str, float]]:
        """
        Return every magic gate's ``(generator, angle)`` in circuit order.

        This mirrors the loop body of :meth:`HSMPO._precompute_original_mps`, but
        collects the dressed rotations instead of applying them to the MPS.
        """
        rotations = []
        for k, magic_gate in self.magic_gates:
            clifford_subcircuit = self._clifford_subcircuit(self.clifford_circuit, k)
            generator, sign = self._conjugate_generator(magic_gate, clifford_subcircuit)
            rotations.append((generator, self._gate_angle(magic_gate) * sign))
        return rotations

    def _synthesize_head(
        self,
        cut_index: int,
        upto_clifford: bool = False,
        preserve_order: bool = True,
    ):
        """Resynthesize the head rotations ``[0:cut_index)`` into a Qiskit circuit."""
        head = self._dressed_rotations()[:cut_index]
        return synthesize_pauli_rotations(
            head,
            nqubits=self.nqubits,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
        )

    def _build_state_mps(
        self,
        head: List[Tuple[str, float]],
        upto_clifford: bool = False,
        preserve_order: bool = True,
    ):
        """Resynthesize the head rotations and build the corresponding state MPS."""
        qiskit_head = synthesize_pauli_rotations(
            head,
            nqubits=self.nqubits,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
        )
        head_circuit = qiskit_circuit_to_qibo(qiskit_head, self.nqubits)

        return self.tn_engine.build_circuit_mps(
            n=self.nqubits,
            initial_state_amplitudes=None,
            initial_state_circuit=self.initial_state + head_circuit,
            max_bond_dimension=self.max_bond_dimension,
        )

    def _build_tail_operator(
        self,
        observable: str,
        tail: List[Tuple[str, float]],
        max_bond_dimension: int | None,
    ):
        """
        Build the observable MPO with the tail rotations folded in.

        Returns ``(operator, sign)`` where ``operator`` is
        ``O' = R_k^dag ... R_1^dag O R_1 ... R_k`` with ``O`` the backpropagated
        observable, and each conjugation compressed to ``max_bond_dimension``
        (``None`` means no truncation, i.e. the exact operator).
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
        upto_clifford: bool = False,
        preserve_order: bool = True,
        return_fidelity: bool = False,
    ):
        """
        Compute the expectation value by splitting the dressed rotations at
        ``cut_index``: the head is resynthesized and applied to the state, the
        tail is folded into the observable as an MPO.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: Number of leading dressed rotations to resynthesize into
                the state circuit. The remaining rotations are folded into the
                observable. ``0`` folds everything into the observable;
                ``len(magic_gates)`` resynthesizes everything into the state.
            upto_clifford: Passed to the Rustiq synthesis; must stay ``False`` for
                exact expectation values.
            preserve_order: Passed to the Rustiq synthesis.
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

        state_mps = self._build_state_mps(
            head, upto_clifford=upto_clifford, preserve_order=preserve_order
        )
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
        upto_clifford: bool = False,
        preserve_order: bool = True,
    ) -> dict:
        """
        Quantify how much the bond-dimension truncation approximates the MPO tail.

        The tail rotations ``[cut_index:]`` are folded into the observable twice:
        once with the working truncation (``self.max_bond_dimension``) and once
        with a reference cap (``reference_max_bond``, ``None`` = untruncated /
        exact). Both an operator-level and an expectation-level error are
        reported.

        Args:
            observable: Pauli string observable (qubit-0-leftmost).
            cut_index: The split point (see :meth:`expectation_from_split`).
            reference_max_bond: Bond dimension for the reference operator. ``None``
                (default) builds the exact, untruncated tail MPO. Note the exact
                MPO bond dimension can grow quickly with the tail length.
            upto_clifford: Passed to the Rustiq synthesis of the head.
            preserve_order: Passed to the Rustiq synthesis of the head.

        Returns:
            dict with:
              - ``relative_frobenius_error``:
                ``||O_approx - O_exact||_F / ||O_exact||_F`` for the folded tail
                operator (0 means the truncation was lossless).
              - ``absolute_frobenius_error``: ``||O_approx - O_exact||_F``.
              - ``expval_approx`` / ``expval_reference``: expectation values on
                the (same) resynthesized-head state, using each operator.
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

        state_mps = self._build_state_mps(
            head, upto_clifford=upto_clifford, preserve_order=preserve_order
        )

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

    def count_two_qubit_gates(
        self,
        cut_index: int,
        upto_clifford: bool = False,
        preserve_order: bool = True,
    ) -> dict:
        """
        Report a two-qubit-gate resource comparison for a given ``cut_index``.

        Only the head rotations ``[0:cut_index)`` are synthesized. The reported
        ``original_circuit_2q_gates`` counts the *full* original ansatz circuit,
        so it is directly comparable to ``synthesized_head_2q_gates`` only at
        ``cut_index == len(magic_gates)`` (fully resynthesized vs. fully
        original); for partial cuts it is provided as context.
        """
        dressed = self._dressed_rotations()
        qiskit_head = self._synthesize_head(
            cut_index, upto_clifford=upto_clifford, preserve_order=preserve_order
        )

        return {
            "n_head_rotations": cut_index,
            "n_tail_rotations": len(dressed) - cut_index,
            "synthesized_head_2q_gates": count_two_qubit_gates_qiskit(qiskit_head),
            "original_circuit_2q_gates": count_two_qubit_gates_qibo(
                self.ansatz.circuit
            ),
        }
