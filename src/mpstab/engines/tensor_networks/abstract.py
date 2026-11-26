"""The interface a tensor-network engine must implement."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Iterable

import numpy as np

from mpstab.engines.tensor_networks import _mps_kernels as kernels

#: What each method takes and returns is engine-specific: an engine may
#: represent states and operators however it likes, as long as the objects it
#: hands back are the ones it is later given. Nothing outside the engine looks
#: inside them: whatever mpstab needs from a state or an operator goes through a
#: method here, and operators leave the engine only as plain arrays
#: (:meth:`TensorNetworkEngine.operator_site_arrays`).


class TensorNetworkEngine(ABC):
    """
    MPS state evolution and MPO expectation values.

    The engine splits into two layers.

    **Representation** (engine-specific). How states and operators are stored,
    and how they are seen as site arrays:

    - :meth:`build_circuit_mps` and :meth:`pauli_mpo` (abstract);
    - :meth:`state_sites`: a state as ``(left, phys, right)`` site arrays in
      mixed canonical form, read and written in place;
    - :meth:`operator_sites` and :meth:`operator_from_sites`: an operator as
      ``(left, 4, right)`` site arrays and back;
    - :attr:`linalg`: the array module and factorizations those arrays live
      on, NumPy by default.

    **Algorithms** (shared, concrete). Expectation values, rotations, norms,
    transition amplitudes and operator conjugation are implemented once here,
    by the windowed kernels of :mod:`._mps_kernels`, on top of that site view.
    An engine that provides the representation hooks gets all of them, with
    the same truncation rule (:attr:`svd_cutoff`); it may still override any of
    them, e.g. to keep a differentiable code path, as
    :class:`~mpstab.engines.QuimbEngine` does on jax and torch, or to run its
    own algorithm, as :class:`~mpstab.engines.NativeTensorNetworkEngine` does.

    Optional features raise ``NotImplementedError`` only when used:

    - :meth:`pauli_sum_mpo` and :meth:`minimize_energy`: DMRG;
    - :meth:`circuit_truncation_fidelity`: the pure-tensor-network baseline of
      :attr:`~mpstab.evolutors.hsmpo.HSMPO.truncation_fidelity_pure_tn`.
    """

    #: Singular-value cutoff (``"rsum2"`` mode, relative) when compressing after
    #: a rotation or a conjugation. ``None`` keeps quimb's default
    #: (:data:`~._mps_kernels.DEFAULT_SVD_CUTOFF`), and ``0.0`` truncates only
    #: through ``max_bond_dimension``.
    svd_cutoff: float | None = None

    #: How worker processes are started when the engine is sent to them
    #: (``noisy_expectation`` with ``n_workers``): a ``multiprocessing`` start
    #: method, or ``None`` for the platform default. An engine holding a GPU
    #: context, which a forked child cannot use, asks for ``"spawn"``.
    process_start_method: str | None = None

    def _overrides(self, method: str) -> bool:
        """Whether the engine's class overrides a method of this base class."""
        return getattr(type(self), method) is not getattr(TensorNetworkEngine, method)

    @property
    def has_site_view(self) -> bool:
        """
        Whether the shared algorithms can run on this engine's states, through
        :meth:`state_sites`. When ``False``, :meth:`pauli_expvals` falls back
        to one :meth:`expval` per string and :meth:`pauli_transitions` is
        unavailable.

        The default is whether the engine implements :meth:`state_sites`; an
        engine whose site view depends on its configuration, such as
        :class:`~mpstab.engines.QuimbEngine` on an autodiff backend, says so
        explicitly.
        """
        return self._overrides("state_sites")

    @property
    def has_operator_sites(self) -> bool:
        """
        Whether operators go in and out of the engine as site arrays, so
        :meth:`conjugate_operator_many` can run the shared kernel. When
        ``False``, it falls back to the engine's own :meth:`conjugate_operator`.

        The default is whether the engine implements :meth:`operator_from_sites`
        and :meth:`operator_sites` (or :meth:`operator_site_arrays`).
        """
        return self._overrides("operator_from_sites") and (
            self._overrides("operator_sites") or self._overrides("operator_site_arrays")
        )

    def _unsupported(self, feature: str):
        """The error an optional method raises on an engine without it."""
        return NotImplementedError(
            f"{type(self).__name__} does not implement {feature}; "
            "QuimbEngine and CuTensorNetEngine do."
        )

    # ------------------------------------------------------------------
    # Representation: what every engine defines for itself.
    # ------------------------------------------------------------------

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

    @property
    def linalg(self) -> kernels.Linalg:
        """
        The array module and factorizations of the engine's site arrays: the
        kernels run on whatever arrays :meth:`state_sites` hands them, with
        these routines. NumPy unless the engine keeps its arrays elsewhere.
        """
        return kernels.NUMPY

    @property
    def _cutoff(self) -> float:
        """The cutoff the kernels truncate with."""
        cutoff = self.svd_cutoff
        return kernels.DEFAULT_SVD_CUTOFF if cutoff is None else cutoff

    def state_sites(self, state_circuit: Any):
        """
        The state as a mutable sequence of ``(left, phys, right)`` site arrays
        in mixed canonical form, the shared algorithms' view of it.

        The returned view has an integer ``center`` attribute, the
        orthogonality center, which the algorithms update; ``len``, item access
        and item assignment over the sites; and a ``commit()`` method, called
        once an algorithm is done, which writes the changed sites back into
        the state (a no-op for an engine whose state is the view itself).
        Assigned arrays replace the old ones, which are never written into.
        """
        raise self._unsupported("the canonical site view (state_sites)")

    def operator_sites(self, operator: Any) -> list:
        """
        The operator as ``(left, 4, right)`` site arrays on :attr:`linalg`, the
        middle index ``2 * out + in``, in any gauge.

        The default converts :meth:`operator_site_arrays`.
        """
        sites = []
        for array in self.operator_site_arrays(operator):
            left, right = array.shape[0], array.shape[1]
            sites.append(
                self.linalg.asarray(array.transpose(0, 2, 3, 1)).reshape(left, 4, right)
            )
        return sites

    def operator_from_sites(self, sites: list, like: Any):
        """
        The engine's operator for ``(left, 4, right)`` site arrays, inverse of
        :meth:`operator_sites`. ``like`` is the operator the sites came from,
        whose metadata (index names, tags) the result may copy.
        """
        raise self._unsupported("operator import (operator_from_sites)")

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

    def pauli_string_of(self, operator: Any) -> str | None:
        """
        The Pauli string an operator built by :meth:`pauli_mpo` stands for, or
        ``None`` for any other operator. A known string lets :meth:`expval`
        take the windowed path instead of a full contraction.
        """
        return None

    def copy_state(self, state_circuit: Any):
        """
        An independent copy of a state: evolving either one in place leaves
        the other untouched.

        The default is a deep copy; engines whose operations never write into
        existing arrays override it with a cheaper one that shares them.
        """
        return copy.deepcopy(state_circuit)

    def bond_dimension(self, state_circuit: Any) -> int | None:
        """
        The largest bond dimension of the state, shown in progress bars.
        ``None`` (the default) when the engine does not report it.
        """
        return None

    # ------------------------------------------------------------------
    # Algorithms: shared by every engine through the site view.
    # ------------------------------------------------------------------

    def squared_norm(self, state_circuit: Any):
        """
        ``<psi|psi>``: 1 for an exact state, lower by the weight truncation
        has discarded. Read off the orthogonality center of the canonical
        form, no full contraction needed.
        """
        view = self.state_sites(state_circuit)
        norm = kernels.squared_norm(view, view.center, self.linalg)
        view.commit()
        return norm

    def pauli_expvals(self, state_circuit: Any, paulis) -> np.ndarray:
        """
        Normalised expectation values of many Pauli strings on one state, in
        order, as a real array. An engine without a site view gets one
        :meth:`expval` per string.

        The strings are visited in order of their first non-identity site, with
        the orthogonality center moved to each in turn, so a whole Hamiltonian
        costs one sweep of the chain plus each term's own support rather than a
        full contraction per term. Moving the center is a pure gauge change
        that leaves the state itself untouched.
        """
        paulis = list(paulis)
        if not self.has_site_view:
            # An engine with its own algorithms: one expectation value each.
            return np.array(
                [np.real(self.expval(state_circuit, self.pauli_mpo(p))) for p in paulis]
            )
        linalg = self.linalg
        view = self.state_sites(state_circuit)
        norm = kernels.squared_norm(view, view.center, linalg)
        values = np.empty(len(paulis))

        windows = [kernels.bounds(pauli) for pauli in paulis]
        for k in sorted(
            range(len(paulis)),
            key=lambda k: view.center if windows[k] is None else windows[k][0],
        ):
            window = windows[k]
            if window is not None and not window[0] <= view.center <= window[1]:
                view.center = kernels.move_center(view, view.center, window[0], linalg)
            values[k] = kernels.pauli_expectation(
                view, view.center, paulis[k], linalg
            ).real
        view.commit()
        return values / norm

    def expval(self, state_circuit: Any, operator: Any):
        """
        Expectation value of ``operator`` on ``state_circuit``, normalised:
        truncation costs the state its unit norm, so the contraction is divided
        by it.

        A Pauli string (:meth:`pauli_string_of`) goes through
        :meth:`pauli_expvals`; any other operator is contracted site by site
        against the state (:meth:`_operator_expectation`).
        """
        view = self.state_sites(state_circuit)
        pauli = self.pauli_string_of(operator)
        if pauli is not None and len(pauli) == len(view):
            view.commit()
            return self.pauli_expvals(state_circuit, [pauli])[0]
        sites = self.operator_sites(operator)
        if len(sites) != len(view):
            view.commit()
            raise ValueError(
                f"The operator acts on {len(sites)} sites, the state has {len(view)}."
            )
        norm = kernels.squared_norm(view, view.center, self.linalg)
        value = self._operator_expectation(view, sites)
        view.commit()
        return np.real(value) / norm

    def _operator_expectation(self, state_sites, operator_sites) -> complex:
        """
        Unnormalized ``<psi|O|psi>`` from the two sets of site arrays. Engines
        with a tensor-network contractor may override it.
        """
        return kernels.operator_expectation(state_sites, operator_sites, self.linalg)

    def pauli_transitions(self, bra_state: Any, ket_state: Any, paulis):
        """
        Normalized transition amplitudes ``<bra|P|ket>`` of many Pauli
        strings between two states, in order, as a complex array: each state
        is divided by its own norm.
        """
        if not self.has_site_view:
            raise NotImplementedError(
                f"{type(self).__name__} does not compute transition amplitudes, "
                "which non-Pauli noise channels need."
            )
        linalg = self.linalg

        def sites(state):
            # One view at a time: the two may be the same state.
            view = self.state_sites(state)
            norm = kernels.squared_norm(view, view.center, linalg)
            arrays = [view[i] for i in range(len(view))]
            view.commit()
            return arrays, norm

        bra_sites, bra_norm = sites(bra_state)
        ket_sites, ket_norm = sites(ket_state)
        norm = np.sqrt(bra_norm * ket_norm)
        values = np.array(
            [
                kernels.pauli_transition(bra_sites, ket_sites, pauli, linalg)
                for pauli in paulis
            ],
            dtype=complex,
        )
        return values / norm

    def pauli_rot(
        self,
        state_circuit: Any,
        generator: str,
        angle: float,
        max_bond_dimension: int | None,
    ):
        """
        Apply ``exp(-i angle/2 generator)`` to the state in place, then
        compress.

        Only the sites between the first and last non-identity letter of the
        generator are touched, and that window alone is recompressed
        (:func:`~._mps_kernels.pauli_rotation`), with the ``"rsum2"`` cutoff
        :attr:`svd_cutoff` and the cap ``max_bond_dimension``.
        """
        view = self.state_sites(state_circuit)
        view.center = kernels.pauli_rotation(
            view,
            view.center,
            generator.upper(),
            float(angle),
            max_bond=max_bond_dimension,
            cutoff=self._cutoff,
            linalg=self.linalg,
        )
        view.commit()

    def rotation_operator(self, generator: str, angle: float):
        """
        Build ``exp(-i angle/2 generator)`` once, in a form
        :meth:`apply_rotation` accepts, so a rotation applied many times is
        not rebuilt on every call.

        The default just stores the arguments, which is all the windowed
        kernel needs; engines with a costly operator construction override
        both methods.
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

    def conjugate_operator(
        self, operator: Any, generator: str, angle: float, max_bond_dimension: int
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.

        Folds the rotation into the observable so that
        ``<R psi| O |R psi> = <psi| R^dag O R |psi>``, compressing to
        ``max_bond_dimension`` with the cutoff :attr:`svd_cutoff`.
        """
        return self.conjugate_operator_many(
            operator, [(generator, angle)], max_bond_dimension
        )

    def conjugate_operator_many(
        self, operator: Any, rotations: Iterable, max_bond_dimension: int | None
    ):
        """
        Conjugate ``operator`` by each ``(generator, angle)`` rotation in turn,
        as repeated :meth:`conjugate_operator` calls would: the first rotation
        is folded first, i.e. it ends up innermost.

        The operator is unpacked once into canonical site arrays
        (:meth:`operator_sites`), and each conjugation touches only its
        generator's window (:func:`~._mps_kernels.operator_conjugation`), so a
        whole tail costs about as much as evolving a state through it.

        An engine without operator site arrays but with its own
        :meth:`conjugate_operator` gets one call of it per rotation instead.

        Even at ``max_bond_dimension=None`` the cutoff applies, and the exact
        operator rank of a scrambled observable grows exponentially, so folding
        a long tail accrues truncation error; see
        :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.tail_truncation` to quantify it.
        """
        if not self.has_operator_sites:
            if not self._overrides("conjugate_operator"):
                raise self._unsupported("operator conjugation (conjugate_operator)")
            # An engine with its own conjugation: one call per rotation.
            for generator, angle in rotations:
                operator = self.conjugate_operator(
                    operator, generator, angle, max_bond_dimension
                )
            return operator
        linalg = self.linalg
        sites = list(self.operator_sites(operator))
        center = kernels.move_center(sites, len(sites) - 1, 0, linalg)
        for generator, angle in rotations:
            center = kernels.operator_conjugation(
                sites,
                center,
                generator.upper(),
                float(angle),
                max_bond=max_bond_dimension,
                cutoff=self._cutoff,
                linalg=linalg,
            )
        return self.operator_from_sites(sites, operator)

    # ------------------------------------------------------------------
    # Optional features.
    # ------------------------------------------------------------------

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
