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
      applied directly to the state MPS, exactly as the base :class:`HSMPO` does
      (via :meth:`TensorNetworkEngine.pauli_rot`);
    - the *tail* rotations ``[cut_index:]`` (closer to the observable) are folded
      into the observable as an MPO by conjugating it through each rotation
      (Heisenberg picture), instead of touching the state.

    This trades two-qubit gate applications on the (truncation-sensitive) state
    MPS for MPO-MPO compression on the observable side.

    Separately -- and only for running on real hardware -- the head rotations can
    be *resynthesized* into an efficient native-gate circuit via Qiskit's Rustiq
    high-level synthesis. That circuit is never used for the classical
    expectation (which applies the exact rotations to the MPS); it is exposed via
    :meth:`synthesized_head_circuit` (qibo / qiskit / OpenQASM 2.0) and measured
    by :meth:`count_two_qubit_gates`. The resynthesis path is the only one that
    needs the optional ``qiskit`` dependency.

    Only :class:`~mpstab.engines.QuimbEngine` is supported for the MPO tail.
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

    def _build_state_mps(self, head: List[Tuple[str, float]]):
        """
        Build the state MPS by applying the head dressed rotations directly.

        Starts from the initial state and applies each ``(generator, angle)`` via
        :meth:`TensorNetworkEngine.pauli_rot`, exactly as
        :meth:`HSMPO._precompute_original_mps` does for the full circuit. No
        resynthesis is involved, so this needs no ``qiskit``.
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

    def synthesized_head_circuit(
        self,
        cut_index: int,
        output_format: str = "qibo",
        upto_clifford: bool = False,
        preserve_order: bool = True,
    ):
        """
        Resynthesize the head rotations ``[0:cut_index)`` into a hardware circuit.

        This is the circuit you would run on a real quantum computer: the head
        Pauli rotations synthesized into native gates by Qiskit's Rustiq backend.
        It is *not* used by :meth:`expectation_from_split`, which applies the exact
        rotations to the MPS instead.

        Args:
            cut_index: Number of leading dressed rotations to synthesize.
            output_format: Output format, one of ``"qibo"`` (a :class:`qibo.Circuit`),
                ``"qiskit"`` (the raw Rustiq :class:`~qiskit.QuantumCircuit`), or
                ``"qasm"`` (an OpenQASM 2.0 string).
            upto_clifford: If ``True``, Rustiq may synthesize up to a trailing
                Clifford (fewer gates; the trailing Clifford can be absorbed into
                the measurement/post-processing). ``False`` (default) reproduces
                the exact head unitary.
            preserve_order: Whether Rustiq must preserve the rotation order.

        Returns:
            The head circuit in the requested format.
        """
        head = self._dressed_rotations()[:cut_index]
        qiskit_circuit = synthesize_pauli_rotations(
            head,
            nqubits=self.nqubits,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
        )

        if output_format == "qiskit":
            return qiskit_circuit

        qibo_circuit = qiskit_circuit_to_qibo(qiskit_circuit, self.nqubits)
        if output_format == "qibo":
            return qibo_circuit
        if output_format == "qasm":
            return qibo_circuit.to_qasm()

        raise ValueError(
            f"Unknown format '{output_format}'. Use one of 'qibo', 'qiskit', 'qasm'."
        )

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
        qiskit_head = self.synthesized_head_circuit(
            cut_index,
            output_format="qiskit",
            upto_clifford=upto_clifford,
            preserve_order=preserve_order,
        )

        return {
            "n_head_rotations": cut_index,
            "n_tail_rotations": len(dressed) - cut_index,
            "synthesized_head_2q_gates": count_two_qubit_gates_qiskit(qiskit_head),
            "original_circuit_2q_gates": count_two_qubit_gates_qibo(
                self.ansatz.circuit
            ),
        }
