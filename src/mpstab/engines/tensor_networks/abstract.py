"""The interface a tensor-network engine must implement."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Iterable

#: What each method takes and returns is engine-specific: an engine may
#: represent states and operators however it likes, as long as the objects it
#: hands back are the ones it is later given. Nothing outside the engine looks
#: inside them: whatever mpstab needs from a state or an operator goes through a
#: method here, and operators leave the engine only as plain arrays
#: (:meth:`TensorNetworkEngine.operator_site_arrays`).


class TensorNetworkEngine(ABC):
    """
    MPS state evolution and MPO expectation values.

    The abstract methods are what :class:`~mpstab.evolutors.hsmpo.HSMPO` needs
    to build and evolve its MPS and evaluate expectation values. The others
    have a default: either a generic implementation an engine may override for
    speed, or a ``NotImplementedError`` for an optional feature, raised only
    when that feature is used:

    - :meth:`conjugate_operator` and :meth:`operator_site_arrays`: the
      head/tail split of :class:`~mpstab.evolutors.hqsmpo.HQSMPO`;
    - :meth:`pauli_sum_mpo` and :meth:`minimize_energy`: DMRG;
    - :meth:`circuit_truncation_fidelity`: the pure-tensor-network baseline of
      :attr:`~mpstab.evolutors.hsmpo.HSMPO.truncation_fidelity_pure_tn`.
    """

    def _unsupported(self, feature: str):
        """The error an optional method raises on an engine without it."""
        return NotImplementedError(
            f"{type(self).__name__} does not implement {feature}; " "QuimbEngine does."
        )

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

    def pauli_transitions(self, bra_state: Any, ket_state: Any, paulis) -> list:
        """
        Normalized transition amplitudes ``<bra|P|ket>`` of many Pauli
        strings between two states, in order: each state is divided by its
        own norm.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not compute transition amplitudes, "
            "which non-Pauli noise channels need."
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

    def copy_state(self, state_circuit: Any):
        """
        An independent copy of a state: evolving either one in place leaves
        the other untouched.

        The default is a deep copy; engines whose operations never write into
        existing arrays override it with a cheaper one that shares them.
        """
        return copy.deepcopy(state_circuit)

    @abstractmethod
    def squared_norm(self, state_circuit: Any):
        """
        ``<psi|psi>``: 1 for an exact state, lower by the weight truncation
        has discarded. A float, or a backend scalar on autodiff backends so it
        stays differentiable.
        """

    def bond_dimension(self, state_circuit: Any) -> int | None:
        """
        The largest bond dimension of the state, shown in progress bars.
        ``None`` (the default) when the engine does not report it.
        """
        return None

    def conjugate_operator(
        self, operator: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.
        """
        raise self._unsupported("operator conjugation (conjugate_operator)")

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

    def operator_site_arrays(self, operator: Any) -> list:
        """
        The operator's site tensors as numpy arrays ordered
        ``(left, right, out, in)``, boundary bonds as singleton axes.

        This is how an operator leaves the engine: the Pauli-basis expansion,
        the shadow estimator and the operator norms in
        :mod:`~mpstab.qpu.measurement.pauli_expansion` all work on these
        arrays, whatever engine folded the operator.
        """
        raise self._unsupported("operator export (operator_site_arrays)")

    def pauli_sum_mpo(self, terms: dict):
        """
        One MPO for ``sum_P c_P P`` from a ``{pauli_string: c_P}`` map, for
        :meth:`minimize_energy`.
        """
        raise self._unsupported("Pauli-sum MPOs (pauli_sum_mpo)")

    def minimize_energy(
        self,
        operator: Any,
        initial_state: Any,
        bond_dims: list,
        cutoff: float,
        tol: float,
        max_sweeps: int,
        verbosity: int,
    ) -> dict:
        """
        Minimize ``<psi|operator|psi>`` over MPS, starting from ``initial_state``
        (which must be left untouched), with bond dimensions growing per sweep
        along ``bond_dims``.

        Returns:
            A dict with the optimized ``state``, its ``energy``, whether it
            ``converged`` and the per-sweep ``energy_history``.
        """
        raise self._unsupported("DMRG (minimize_energy)")

    def circuit_truncation_fidelity(
        self, nqubits: int, circuit: Any, max_bond_dimension: int | None
    ) -> float:
        """
        Fidelity estimate of simulating the whole qibo ``circuit`` gate by gate
        as a plain MPS capped at ``max_bond_dimension``.
        """
        raise self._unsupported("plain-MPS circuit simulation")
