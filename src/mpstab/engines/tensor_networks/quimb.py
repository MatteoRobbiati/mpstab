"""Tensor-network engine backed by quimb, with cotengra contraction paths."""

from __future__ import annotations

import copy
from typing import Any

import cotengra as ctg
import numpy as np
from qibo.gates.abstract import ParametrizedGate
from quimb.gates import I, X, Y, Z
from quimb.tensor import (
    CircuitMPS,
    MatrixProductOperator,
    MatrixProductState,
    MPO_identity,
    MPO_product_operator,
)

from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine

# jax and torch are optional quimb backends.
try:
    import jax.numpy as jnp

    from mpstab.engines.tensor_networks._jax_linalg import (
        install as _install_jax_linalg,
    )
except ImportError:
    jnp = None
try:
    import torch
except ImportError:
    torch = None

#: qibo gate names mapped to their quimb equivalents.
GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "rxx": "RXX",
    "ryy": "RYY",
    "rzz": "RZZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}

#: Pauli labels mapped to quimb's own single-qubit Pauli arrays.
PAULI_GATES = {"I": I, "X": X, "Y": Y, "Z": Z}

#: quimb's default singular-value cutoff (``"rsum2"`` mode) when compressing.
_QUIMB_DEFAULT_CUTOFF = 1e-10

#: Attribute marking an MPO built by :meth:`QuimbEngine.pauli_mpo` with the
#: Pauli string it represents. quimb's ``copy`` drops it, so any arithmetic on
#: the MPO (``2 * mpo``, a conjugation) falls back to a generic contraction.
_PAULI_ATTR = "_mpstab_pauli"

#: Attribute caching an MPS's gauge: ``(center, tids, arrays, inds)``.
_GAUGE_ATTR = "_mpstab_gauge"


class PauliRotation(tuple):
    """
    ``(generator, angle)`` for ``exp(-i angle/2 generator)``, as handed out by
    :meth:`QuimbEngine.rotation_operator` on the numpy backend, where rotations
    are applied by the windowed kernel rather than as an MPO.
    """

    __slots__ = ()

    def __new__(cls, generator: str, angle: float):
        return super().__new__(cls, (generator, angle))


class _CanonicalMPS:
    """
    A quimb ``MatrixProductState`` seen as a list of ``(left, phys, right)``
    arrays in mixed canonical form, for :mod:`._mps_kernels`.

    The orthogonality center is cached on the MPS itself together with the
    exact array object held by each site, and trusted only while the MPS holds
    exactly those arrays and no other tensor. The cached arrays are read-only,
    so anything else that changes the MPS has to replace them, and the next
    view then re-canonicalizes from scratch. ``deepcopy`` preserves the
    array identities, so copies keep their cached gauge, but numpy hands the
    copies back writable: write into a deep copy's arrays and the gauge goes
    stale.

    Arrays are loaded lazily, and only the ones a kernel changed are written
    back, so the cost of an operation scales with the sites it touches.
    """

    def __init__(self, mps: MatrixProductState):
        self.mps = mps
        self.L = mps.L
        tensor_map = mps.tensor_map
        gauge = getattr(mps, _GAUGE_ATTR, None)
        if gauge is not None:
            center, tids, arrays, inds = gauge
            # Exactly L tensors: an extra one (``gate_(..., contract=False)``)
            # changes the state while leaving the site arrays untouched.
            if len(tids) == self.L == len(tensor_map) and all(
                tid in tensor_map and tensor_map[tid].data is array
                for tid, array in zip(tids, arrays)
            ):
                self.center, self.tids, self.inds = center, tids, inds
                self.stored = list(arrays)
                self.loaded = {}
                return
        self._standardize()

    def _standardize(self):
        """Put every tensor in quimb's ``lrp`` order and canonicalize to site 0."""
        mps, L = self.mps, self.L
        self.tids, self.inds = [], []
        for i in range(L):
            (tid,) = mps._get_tids_from_tags(mps.site_tag(i), which="any")
            bonds = []
            if i > 0:
                bonds.append(mps.bond(i - 1, i))
            if i < L - 1:
                bonds.append(mps.bond(i, i + 1))
            inds = (*bonds, mps.site_ind(i))
            mps.tensor_map[tid].transpose_(*inds)
            self.tids.append(tid)
            self.inds.append(inds)
        self.stored = [mps.tensor_map[tid].data for tid in self.tids]
        self.loaded = {}
        self.center = kernels.move_center(self, L - 1, 0)

    def __getitem__(self, i: int) -> np.ndarray:
        if i not in self.loaded:
            data = np.asarray(self.stored[i])
            if self.L == 1:
                data = data[None, :, None]
            elif i == 0:
                data = data.T[None]  # (r, p) -> (1, p, r)
            elif i == self.L - 1:
                data = data[:, :, None]  # (l, p) -> (l, p, 1)
            else:
                data = data.transpose(0, 2, 1)  # (l, r, p) -> (l, p, r)
            self.loaded[i] = data
        return self.loaded[i]

    def __setitem__(self, i: int, array: np.ndarray):
        self.loaded[i] = array
        self.stored[i] = None  # dirty

    def commit(self):
        """Write the changed arrays back to the MPS and cache the gauge."""
        tensor_map, L = self.mps.tensor_map, self.L
        for i, array in self.loaded.items():
            if self.stored[i] is not None:
                continue
            if L == 1:
                data = array[0, :, 0]
            elif i == 0:
                data = array[0].T
            elif i == L - 1:
                data = array[:, :, 0]
            else:
                data = array.transpose(0, 2, 1)
            tensor = tensor_map[self.tids[i]]
            tensor.modify(data=data, inds=self.inds[i])
            self.stored[i] = tensor.data
        # Read-only, so an in-place write raises instead of silently
        # invalidating the cached gauge, here or on a ``copy_state`` copy
        # sharing these arrays.
        for array in self.stored:
            array.flags.writeable = False
        setattr(
            self.mps,
            _GAUGE_ATTR,
            (self.center, self.tids, tuple(self.stored), self.inds),
        )


def _qibo_circuit_to_quimb(nqubits: int, qibo_circ, **circuit_kwargs) -> CircuitMPS:
    """
    Convert a qibo circuit into a quimb ``CircuitMPS``. Measurement gates are
    skipped.

    Args:
        nqubits: circuit width.
        qibo_circ: the qibo circuit to convert.
        circuit_kwargs: forwarded to the ``CircuitMPS`` constructor, e.g.
            ``max_bond`` and ``to_backend``.

    Raises:
        ValueError: on a gate with no quimb equivalent in :data:`GATE_MAP`.
    """
    circuit = CircuitMPS(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        name = getattr(gate, "name", None)
        quimb_name = GATE_MAP.get(name)
        if quimb_name == "measure":
            continue
        if quimb_name is None:
            raise ValueError(f"Gate {name} is not supported by the quimb engine.")

        parameters = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())
        parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if parametrized:
            circuit.apply_gate(quimb_name, *parameters, *qubits, parametrized=True)
        else:
            circuit.apply_gate(quimb_name, *parameters, *qubits)

    return circuit


def _array_module(backend: str):
    """The array module of a quimb backend: numpy, ``jax.numpy`` or torch."""
    if backend == "numpy":
        return np
    if backend not in ("jax", "torch"):
        raise ValueError(
            f"Unsupported quimb backend {backend!r}, expected 'numpy', 'jax' or 'torch'."
        )
    module = jnp if backend == "jax" else torch
    if module is None:
        raise ImportError(
            f"The {backend!r} quimb backend needs {backend}: "
            f'pip install "mpstab[{"pytorch" if backend == "torch" else "jax"}]"'
        )
    if backend == "jax":
        # jax's SVD derivative is nan on repeated zero singular values, which
        # an MPS evolved from a product state hits at its first compressions.
        _install_jax_linalg()
    return module


class QuimbEngine(TensorNetworkEngine):
    """
    MPS evolution and MPO expectation values via quimb.

    The only engine supporting operator-side conjugation, and so the only one
    that can run :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`'s head/tail
    split or DMRG optimization.
    """

    def __init__(
        self,
        backend: str = "numpy",
        cache: bool = False,
        cache_directory: str | None = "contractions_cache",
        svd_cutoff: float | None = None,
    ):
        """
        Args:
            backend: array backend, ``"numpy"``, ``"jax"`` or ``"torch"``.
            cache: reuse contraction paths across calls via cotengra's
                hyper-optimizer instead of contracting greedily.
            cache_directory: where cotengra persists those paths. Created if
                missing.
            svd_cutoff: singular-value cutoff when compressing the MPS after a
                rotation. ``None`` keeps quimb's default (``1e-10``, relative),
                and ``0.0`` truncates only through ``max_bond_dimension``. Many
                small-angle rotations leave Schmidt values below the default,
                so exact comparisons need ``0.0``.
        """
        self.svd_cutoff = svd_cutoff
        self.np = _array_module(backend)
        self.backend = backend
        self.optimizer = (
            ctg.ReusableHyperOptimizer(
                directory=cache_directory,
                minimize="flops",
                max_repeats=128,
                progbar=False,
            )
            if cache
            else "greedy"
        )

    def __getstate__(self):
        # The array module does not pickle, which sending the engine to another
        # process (noisy_expectation with n_workers) needs; the backend name
        # is enough to restore it.
        state = self.__dict__.copy()
        del state["np"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.np = _array_module(self.backend)

    def _to_backend(self, mpo):
        """Move an MPO's arrays onto the engine's backend, when it needs it."""
        if self.backend == "torch":
            mpo.apply_to_arrays(self.np.as_tensor)
        return mpo

    def PauliExp(self, pauli_string: str, theta: float):
        """
        The MPO for ``exp(-i theta/2 P)``, from Euler's formula
        ``cos(theta/2) I - i sin(theta/2) P``.
        """
        identity = MPO_identity(len(pauli_string), phys_dim=2)
        pauli = MPO_product_operator(
            [PAULI_GATES[label.upper()] for label in pauli_string]
        )

        if self.backend == "torch":
            self._to_backend(pauli)
            self._to_backend(identity)
            theta = self.np.as_tensor(theta)

        return (self.np.cos(theta / 2) * identity).add_MPO(
            -1j * self.np.sin(theta / 2) * pauli
        )

    def build_circuit_mps(
        self,
        n: int,
        initial_state_amplitudes: Any,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """Build the MPS by running ``initial_state_circuit`` through quimb."""
        if initial_state_circuit is None:
            raise NotImplementedError(
                "QuimbEngine builds its MPS from a qibo circuit; pass "
                "initial_state_circuit rather than initial_state_amplitudes."
            )
        return _qibo_circuit_to_quimb(
            nqubits=n,
            qibo_circ=initial_state_circuit,
            max_bond=max_bond_dimension,
            to_backend=self.np.asarray,
        ).psi

    @property
    def _fast(self) -> bool:
        """
        Whether the windowed NumPy kernels apply. They are plain NumPy, so the
        jax and torch backends, which may need to differentiate through the
        state, keep quimb's own contractions.
        """
        return self.backend == "numpy"

    def pauli_mpo(self, pauli_string: str | object):
        """Build the MPO for a Pauli string."""
        mpo = MPO_product_operator(
            [PAULI_GATES[label.upper()] for label in pauli_string]
        )
        mpo.add_tag("MPO")
        setattr(mpo, _PAULI_ATTR, "".join(pauli_string).upper())
        return self._to_backend(mpo)

    def expval(
        self, state_circuit: MatrixProductState, operator: MatrixProductOperator
    ):
        """
        Expectation value of ``operator`` on ``state_circuit``, normalised.

        Truncation costs the state its unit norm, so the contraction is divided
        by it; the norm is also kept on ``self.norm`` for callers that want it.

        A Pauli string from :meth:`pauli_mpo` is evaluated directly on the
        state's canonical form, over the sites between the string's support and
        the orthogonality center only. This may move the center, a pure gauge
        change that leaves the state itself untouched.

        Any other operator MPO has upper (output) index ``k{i}`` and lower
        (input) index ``b{i}``. The ket must feed the operator's input, so it is
        reindexed ``k -> b`` while the conjugated bra keeps ``k{i}`` to meet the
        output. Reindexing the bra instead would evaluate ``<psi|O^T|psi>``,
        which differs for any operator with an odd number of Y's.
        """
        pauli = getattr(operator, _PAULI_ATTR, None)
        if self._fast and pauli is not None and len(pauli) == state_circuit.L:
            return self.pauli_expvals(state_circuit, [pauli])[0]

        self.norm = state_circuit.norm(squared=True).real
        ket = state_circuit.reindex({f"k{i}": f"b{i}" for i in range(state_circuit.L)})
        bra = state_circuit.H
        return (bra & operator & ket).contract(
            backend=self.backend, optimize=self.optimizer
        ).real / self.norm

    def pauli_expvals(self, state_circuit: MatrixProductState, paulis):
        """
        Normalised expectation values of many Pauli strings on one state.

        The strings are visited in order of their first non-identity site, with
        the orthogonality center moved to each in turn, so a whole Hamiltonian
        costs one sweep of the chain plus each term's own support rather than a
        full contraction per term.

        Returns:
            One value per string, in the order given: a real array on the
            numpy backend, a list of backend scalars otherwise.
        """
        paulis = list(paulis)
        if not self._fast:
            # A plain list: jax and torch values may be traced for autodiff.
            return [self.expval(state_circuit, self.pauli_mpo(p)) for p in paulis]

        view = _CanonicalMPS(state_circuit)
        self.norm = kernels.squared_norm(view, view.center)
        values = np.empty(len(paulis))

        windows = [kernels.bounds(pauli) for pauli in paulis]
        for k in sorted(
            range(len(paulis)),
            key=lambda k: view.center if windows[k] is None else windows[k][0],
        ):
            window = windows[k]
            if window is not None and not window[0] <= view.center <= window[1]:
                view.center = kernels.move_center(view, view.center, window[0])
            values[k] = kernels.pauli_expectation(view, view.center, paulis[k]).real
        view.commit()
        return values / self.norm

    def pauli_rot(
        self,
        state_circuit: MatrixProductState,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """Apply ``exp(-i angle/2 generator)`` to the state in place, then compress."""
        self.apply_rotation(
            state_circuit, self.rotation_operator(generator, angle), max_bond_dimension
        )

    def rotation_operator(self, generator: str, angle: float):
        """
        ``exp(-i angle/2 generator)`` in a form :meth:`apply_rotation` accepts:
        a :class:`PauliRotation` on the numpy backend, applied by the windowed
        kernel, and the backend MPO otherwise.
        """
        if self._fast:
            return PauliRotation(generator.upper(), float(angle))
        return self._to_backend(self.PauliExp(generator, angle))

    def apply_rotation(
        self,
        state_circuit: MatrixProductState,
        rotation,
        max_bond_dimension: int | None,
    ):
        """
        Apply a :meth:`rotation_operator` result to the state in place, then
        compress.

        A :class:`PauliRotation` only touches the sites between the first and
        last non-identity letter of its generator, recompressing that window
        alone; an MPO goes through quimb's ``gate_with_mpo``, which recompresses
        the whole chain. Both truncate with the same rule.
        """
        cutoff = _QUIMB_DEFAULT_CUTOFF if self.svd_cutoff is None else self.svd_cutoff
        if isinstance(rotation, PauliRotation):
            generator, angle = rotation
            view = _CanonicalMPS(state_circuit)
            view.center = kernels.pauli_rotation(
                view,
                view.center,
                generator,
                angle,
                max_bond=max_bond_dimension,
                cutoff=cutoff,
            )
            view.commit()
            return

        options = {} if self.svd_cutoff is None else {"cutoff": self.svd_cutoff}
        state_circuit.gate_with_mpo(
            rotation, inplace=True, max_bond=max_bond_dimension, **options
        )

    def copy_state(self, state_circuit: MatrixProductState):
        """
        An independent copy of the state.

        On the numpy backend the copy has its own tensors but shares their
        arrays: the windowed kernels, like quimb's own operations, replace a
        tensor's array rather than write into it, so neither state can change
        the other. The cached canonical form is carried over, so the copy
        needs no re-canonicalization. Other backends get a deep copy.
        """
        if not self._fast:
            return copy.deepcopy(state_circuit)
        state = state_circuit.copy()
        gauge = getattr(state_circuit, _GAUGE_ATTR, None)
        if gauge is not None:
            setattr(state, _GAUGE_ATTR, gauge)
        return state

    def conjugate_operator(
        self,
        operator: MatrixProductOperator,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.

        Folds the rotation into the observable so that
        ``<R psi| O |R psi> = <psi| R^dag O R |psi>``. The generators are Pauli
        strings, so ``R(angle)^dag == R(-angle)``, and each MPO-MPO product is
        compressed to ``max_bond_dimension``.

        Even at ``max_bond_dimension=None`` quimb applies its default SVD cutoff,
        and the exact operator rank of a scrambled observable grows
        exponentially, so folding a long tail accrues truncation error. Keeping
        this exact is only affordable for modest tail lengths; see
        :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.tail_truncation` to
        quantify it.
        """
        rotation = self._to_backend(self.PauliExp(generator, angle))
        rotation_dag = self._to_backend(self.PauliExp(generator, -angle))

        operator = rotation_dag.apply(
            operator, compress=True, max_bond=max_bond_dimension
        )
        return operator.apply(rotation, compress=True, max_bond=max_bond_dimension)
