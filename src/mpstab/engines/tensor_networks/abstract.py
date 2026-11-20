"""The interface a tensor-network engine must implement."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Iterable

#: What each method takes and returns is engine-specific: an engine may
#: represent states and operators however it likes, as long as the objects it
#: hands back are the ones it is later given.


class TensorNetworkEngine(ABC):
    """MPS state evolution and MPO expectation values."""

    @abstractmethod
    def build_circuit_mps(
        self,
        n: int,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """
        Build an MPS on ``n`` qubits in the state a qibo circuit prepares.

        Args:
            n: number of qubits.
            initial_state_circuit: the qibo circuit preparing the state.
            max_bond_dimension: truncation cap, or ``None`` for no cap.
        """

    @abstractmethod
    def pauli_mpo(self, pauli_string: str | Any):
        """Build the MPO for a Pauli string."""

    @abstractmethod
    def expval(self, state_circuit: Any, operator: Any):
        """Expectation value of ``operator`` on ``state_circuit``."""

    def pauli_expvals(self, state_circuit: Any, paulis) -> list:
        """
        Expectation values of many Pauli strings on one state, in order.

        Engines that can share work across strings override this.
        """
        return [self.expval(state_circuit, self.pauli_mpo(p)) for p in paulis]

    def pauli_transitions(
        self, bra_state: Any, ket_state: Any, paulis, normalize: bool = True
    ) -> list:
        """
        Transition amplitudes ``<bra|P|ket>`` of many Pauli strings between
        two states, in order. With ``normalize`` each state is divided by its
        own norm.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not compute transition amplitudes, "
            "which analytic gradients and non-Pauli noise channels need; use "
            "QuimbEngine."
        )

    def apply_pauli_sum(
        self, state_circuit: Any, paulis, coefficients, max_bond_dimension
    ):
        """
        A new state ``sum_t c_t P_t |state>``, compressed to
        ``max_bond_dimension``; the input state is left untouched.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not apply Pauli sums to a state, which "
            "the adjoint gradient needs; use QuimbEngine or method='flips'."
        )

    @abstractmethod
    def pauli_rot(
        self, state_circuit: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """Apply ``exp(-i angle/2 generator)`` to the state, in place."""

    def rotation_operator(self, generator: str, angle: float):
        """
        Build ``exp(-i angle/2 generator)`` once, in a form
        :meth:`apply_rotation` accepts, so a rotation applied many times is
        not rebuilt on every call.

        The default just stores the arguments; engines with a costly operator
        construction override both methods.
        """
        return generator, angle

    def apply_rotation(self, state_circuit: Any, rotation: Any, max_bond_dimension):
        """
        Apply a rotation built by :meth:`rotation_operator` to the state, in
        place. Must give the same state as :meth:`pauli_rot` with the same
        generator and angle.
        """
        generator, angle = rotation
        self.pauli_rot(state_circuit, generator, angle, max_bond_dimension)

    def squared_norm(self, state_circuit: Any):
        """
        ``<psi|psi>`` of a state, which truncation leaves below 1, as a backend
        scalar (so it stays differentiable). The default asks the state's own
        ``norm(squared=True)``, ``nan`` if it has none.
        """
        norm = getattr(state_circuit, "norm", None)
        return float("nan") if norm is None else norm(squared=True)

    def copy_state(self, state_circuit: Any):
        """
        An independent copy of a state: evolving either one in place leaves
        the other untouched.

        The default is a deep copy; engines whose operations never write into
        existing arrays override it with a cheaper one that shares them.
        """
        return copy.deepcopy(state_circuit)

    @abstractmethod
    def conjugate_operator(
        self, operator: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.
        """

    def conjugate_operator_many(
        self, operator: Any, rotations: Iterable, max_bond_dimension: int
    ):
        """
        Conjugate ``operator`` by each ``(generator, angle)`` rotation in turn,
        as repeated :meth:`conjugate_operator` calls would: the first rotation
        is folded first, i.e. it ends up innermost.

        Engines override it to keep the operator in a working form across the
        whole sequence instead of converting it on every call.
        """
        for generator, angle in rotations:
            operator = self.conjugate_operator(
                operator, generator, angle, max_bond_dimension
            )
        return operator
