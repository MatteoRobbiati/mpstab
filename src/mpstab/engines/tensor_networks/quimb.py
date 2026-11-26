"""Tensor-network engine backed by quimb."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from qibo.gates.abstract import ParametrizedGate
from quimb.tensor import (
    DMRG2,
    CircuitMPS,
    MatrixProductOperator,
    MatrixProductState,
    MPO_identity,
    MPO_product_operator,
    bonds,
)

from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine
from mpstab.pauli import PAULI_MATRICES

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


#: quimb's default singular-value cutoff (``"rsum2"`` mode) when compressing.
_QUIMB_DEFAULT_CUTOFF = 1e-10

#: Relative cutoff for compressions meant to be exact up to roundoff.
_EXACT_CUTOFF = 1e-14

#: Pauli strings stacked into one block-diagonal MPO before compressing it.
_PAULI_SUM_CHUNK = 64

#: Attribute marking an MPO built by :meth:`QuimbEngine.pauli_mpo` with the
#: Pauli string it represents. quimb's ``copy`` drops it, so any arithmetic on
#: the MPO (``2 * mpo``, a conjugation) falls back to a generic contraction.
_PAULI_ATTR = "_mpstab_pauli"

#: Attribute caching an MPS's gauge: ``(center, tids, arrays, inds)``.
_GAUGE_ATTR = "_mpstab_gauge"


class ArrayMPS:
    """
    An MPS as bare ``(left, phys, right)`` arrays in mixed canonical form
    around ``center``: the :meth:`QuimbEngine.working_state` of the numpy
    backend.

    It skips quimb's tensor objects, whose bookkeeping dominates the many
    small rotations of a noise estimate. The kernels never write into an
    array, they replace it, so a copy is a new list sharing the arrays.
    """

    def __init__(self, arrays: list[np.ndarray], center: int):
        self.arrays = arrays
        self.center = center
        self.L = len(arrays)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.arrays[i]

    def __setitem__(self, i: int, array: np.ndarray):
        self.arrays[i] = array

    def commit(self):
        """Nothing to write back: the arrays are the state."""

    def copy(self) -> ArrayMPS:
        return ArrayMPS(list(self.arrays), self.center)

    def max_bond(self) -> int:
        return max(array.shape[2] for array in self.arrays)


def _view(state):
    """The kernel view of a state: itself if bare arrays, else a canonical view."""
    return state if isinstance(state, ArrayMPS) else _CanonicalMPS(state)


def _pauli_product_mpo(pauli_string: str):
    """The bond-dimension-1 MPO of a Pauli string."""
    return MPO_product_operator(
        [PAULI_MATRICES[label.upper()] for label in pauli_string]
    )


class _CanonicalMPS:
    """
    A quimb ``MatrixProductState`` seen as a list of ``(left, phys, right)``
    arrays in mixed canonical form, for :mod:`._mps_kernels`.

    The orthogonality center is cached on the MPS itself together with the
    exact array object held by each site, and trusted only while the MPS holds
    exactly those arrays and no other tensor. The cached arrays are read-only,
    so anything else that changes the MPS has to replace them, and the next
    view then re-canonicalizes from scratch. ``deepcopy`` keeps the cached
    gauge pointing at the copy's arrays, but numpy hands those copies back
    writable, so a writable array also invalidates the cache: a deep copy is
    re-canonicalized once, whether or not anything wrote into it.

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
            # changes the state while leaving the site arrays untouched. And
            # still read-only: a deep copy's arrays are writable, so they may
            # have been written into.
            if len(tids) == self.L == len(tensor_map) and all(
                tid in tensor_map
                and tensor_map[tid].data is array
                and not array.flags.writeable
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


def _mpo_to_sites(mpo: MatrixProductOperator) -> list[np.ndarray]:
    """
    The MPO's site tensors as ``(left, 4, right)`` complex arrays, the middle
    index being ``2 * out + in`` (quimb's upper index is the output).
    """
    L = mpo.L
    sites = []
    for i in range(L):
        inds = [mpo.upper_ind(i), mpo.lower_ind(i)]
        if i > 0:
            inds.insert(0, mpo.bond(i - 1, i))
        if i < L - 1:
            inds.append(mpo.bond(i, i + 1))
        data = np.asarray(mpo[i].transpose(*inds).data, dtype=complex)
        if i == 0:
            data = data[None]
        if i == L - 1:
            data = data[..., None]
        left, right = data.shape[0], data.shape[-1]
        sites.append(data.reshape(left, 4, right))
    return sites


def _sites_to_mpo(sites: list[np.ndarray], like: MatrixProductOperator):
    """Inverse of :func:`_mpo_to_sites`, with ``like``'s index names and tags."""
    L = len(sites)
    arrays = []
    for i, site in enumerate(sites):
        tensor = site.reshape(site.shape[0], 2, 2, site.shape[-1])  # (l, k, b, r)
        if L == 1:
            arrays.append(tensor[0, :, :, 0])
        elif i == 0:
            arrays.append(tensor[0].transpose(2, 0, 1))  # (r, k, b)
        elif i == L - 1:
            arrays.append(tensor[..., 0])  # (l, k, b)
        else:
            arrays.append(tensor.transpose(0, 3, 1, 2))  # (l, r, k, b)
    return MatrixProductOperator(
        arrays,
        shape="lrud",
        upper_ind_id=like.upper_ind_id,
        lower_ind_id=like.lower_ind_id,
        site_tag_id=like.site_tag_id,
        tags=[tag for tag in like.tags if tag not in set(like.site_tags)],
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


def _pauli_sum_mpo(paulis: list[str], coefficients) -> MatrixProductOperator:
    """
    ``sum_t c_t P_t`` as an MPO: block-diagonal site tensors, one block per
    string, compressed once to its exact rank.
    """
    nqubits = len(paulis[0])
    coefficients = np.asarray(coefficients, dtype=complex)
    sites = [np.stack([PAULI_MATRICES[p[j]] for p in paulis]) for j in range(nqubits)]
    sites[0] = sites[0] * coefficients[:, None, None]  # (term, up, down)
    if nqubits == 1:
        return MatrixProductOperator([sites[0].sum(axis=0)])
    arrays = [sites[0]]
    for site in sites[1:-1]:
        block = np.zeros((len(paulis), *site.shape), dtype=complex)
        block[np.arange(len(paulis)), np.arange(len(paulis))] = site
        arrays.append(block)  # (left, right, up, down)
    arrays.append(sites[-1])
    operator = MatrixProductOperator(arrays)
    operator.compress(cutoff=_EXACT_CUTOFF)
    return operator


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

    Besides evolving states, it conjugates operators, which
    :class:`~mpstab.evolutors.hqsmpo.HQSMPO`'s head/tail split needs,
    and its MPS feeds quimb's DMRG.
    """

    def __init__(
        self,
        backend: str = "numpy",
        svd_cutoff: float | None = None,
    ):
        """
        Args:
            backend: array backend, ``"numpy"``, ``"jax"`` or ``"torch"``.
            svd_cutoff: singular-value cutoff when compressing the MPS after a
                rotation. ``None`` keeps quimb's default (``1e-10``, relative),
                and ``0.0`` truncates only through ``max_bond_dimension``. Many
                small-angle rotations leave Schmidt values below the default,
                so exact comparisons need ``0.0``.
        """
        self.svd_cutoff = svd_cutoff
        self.np = _array_module(backend)
        self.backend = backend

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

    def _pauli_exp(self, pauli_string: str, theta: float):
        """
        The backend MPO for ``exp(-i theta/2 P)``, from Euler's formula
        ``cos(theta/2) I - i sin(theta/2) P``.
        """
        identity = self._to_backend(MPO_identity(len(pauli_string), phys_dim=2))
        pauli = self._to_backend(_pauli_product_mpo(pauli_string))
        if self.backend == "torch":
            theta = self.np.as_tensor(theta)

        return (self.np.cos(theta / 2) * identity).add_MPO(
            -1j * self.np.sin(theta / 2) * pauli
        )

    def build_circuit_mps(
        self,
        n: int,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """Build the MPS by running ``initial_state_circuit`` through quimb."""
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
        mpo = _pauli_product_mpo(pauli_string)
        mpo.add_tag("MPO")
        setattr(mpo, _PAULI_ATTR, "".join(pauli_string).upper())
        return self._to_backend(mpo)

    def expval(
        self, state_circuit: MatrixProductState, operator: MatrixProductOperator
    ):
        """
        Expectation value of ``operator`` on ``state_circuit``, normalised.

        Truncation costs the state its unit norm, so the contraction is divided
        by it.

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

        norm = self.squared_norm(state_circuit).real
        ket = state_circuit.reindex({f"k{i}": f"b{i}" for i in range(state_circuit.L)})
        bra = state_circuit.H
        return (bra & operator & ket).contract(
            backend=self.backend, optimize="greedy"
        ).real / norm

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

        view = _view(state_circuit)
        norm = kernels.squared_norm(view, view.center)
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
        return values / norm

    def pauli_transitions(
        self,
        bra_state: MatrixProductState,
        ket_state: MatrixProductState,
        paulis,
        normalize: bool = True,
    ):
        """
        Transition amplitudes ``<bra|P|ket>`` of many Pauli strings between
        two states, with ``normalize`` divided by the norms of both states.

        Returns:
            A complex array on the numpy backend, a list of backend scalars
            otherwise.
        """
        paulis = list(paulis)
        if not self._fast:
            norm = (
                np.sqrt(
                    bra_state.norm(squared=True).real
                    * ket_state.norm(squared=True).real
                )
                if normalize
                else 1.0
            )
            ket = ket_state.reindex({f"k{i}": f"b{i}" for i in range(ket_state.L)})
            return [
                (bra_state.H & self.pauli_mpo(p) & ket).contract(
                    backend=self.backend, optimize="greedy"
                )
                / norm
                for p in paulis
            ]

        def sites(state):
            # One view at a time: the two may be the same state.
            view = _view(state)
            norm = kernels.squared_norm(view, view.center)
            arrays = [view[i] for i in range(view.L)]
            view.commit()
            return arrays, view.center, norm

        bra_sites, bra_center, bra_norm = sites(bra_state)
        ket_sites, ket_center, ket_norm = sites(ket_state)
        values = kernels.pauli_transitions(
            bra_sites, ket_sites, paulis, centers=(bra_center, ket_center)
        )
        return values / np.sqrt(bra_norm * ket_norm) if normalize else values

    def apply_pauli_sum(
        self,
        state_circuit: MatrixProductState,
        paulis,
        coefficients,
        max_bond_dimension: int | None,
    ) -> MatrixProductState:
        """
        A new state ``sum_t c_t P_t |state>``.

        The terms are stacked into block-diagonal MPOs of
        :data:`_PAULI_SUM_CHUNK` terms each, compressed exactly and summed (a
        sum of local strings keeps a small bond), and the result is applied to
        the state and compressed to ``max_bond_dimension``.
        """
        paulis, coefficients = list(paulis), list(coefficients)
        operator = None
        for start in range(0, len(paulis), _PAULI_SUM_CHUNK):
            chunk = slice(start, start + _PAULI_SUM_CHUNK)
            block = _pauli_sum_mpo(paulis[chunk], coefficients[chunk])
            if operator is None:
                operator = block
            else:
                operator = operator.add_MPO(block)
                operator.compress(cutoff=_EXACT_CUTOFF)
        operator = self._to_backend(operator)
        options = {} if self.svd_cutoff is None else {"cutoff": self.svd_cutoff}
        return operator.apply(
            state_circuit, compress=True, max_bond=max_bond_dimension, **options
        )

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
        a ``(generator, angle)`` tuple on the numpy backend, applied by the
        windowed kernel, and the backend MPO otherwise.
        """
        if self._fast:
            return generator.upper(), float(angle)
        return self._pauli_exp(generator, angle)

    def apply_rotation(
        self,
        state_circuit: MatrixProductState,
        rotation,
        max_bond_dimension: int | None,
    ):
        """
        Apply a :meth:`rotation_operator` result to the state in place, then
        compress.

        A ``(generator, angle)`` tuple only touches the sites between the first and
        last non-identity letter of its generator, recompressing that window
        alone; an MPO goes through quimb's ``gate_with_mpo``, which recompresses
        the whole chain. Both truncate with the same rule.
        """
        cutoff = _QUIMB_DEFAULT_CUTOFF if self.svd_cutoff is None else self.svd_cutoff
        if isinstance(rotation, tuple):
            generator, angle = rotation
            view = _view(state_circuit)
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

    def squared_norm(self, state_circuit: MatrixProductState):
        """
        ``<psi|psi>`` of the MPS. On the numpy backend it is read off the
        orthogonality center of the cached canonical form, no full contraction
        needed; other backends return quimb's backend scalar, so it stays
        differentiable.

        On the numpy backend, when the state has no valid cached canonical form
        (a fresh state, or one changed outside the engine), it is
        canonicalized in place first: its tensors are transposed and
        gauge-transformed, the state itself is unchanged.
        """
        if not self._fast:
            return state_circuit.norm(squared=True)
        view = _view(state_circuit)
        norm = kernels.squared_norm(view, view.center)
        view.commit()
        return float(norm)

    @property
    def truncation_cutoff(self) -> float:
        """The relative singular-value cutoff rotations are compressed with."""
        return _QUIMB_DEFAULT_CUTOFF if self.svd_cutoff is None else self.svd_cutoff

    def site_arrays(self, state_circuit: MatrixProductState):
        """
        The state as the form :mod:`._mps_kernels` works on: a list of
        ``(left, phys, right)`` arrays in mixed canonical form, and the
        orthogonality center. numpy backend only.

        The arrays are read-only and shared with the state; the kernels never
        write into an array, they replace it in the list.
        """
        view = _CanonicalMPS(state_circuit)
        arrays = [view[i] for i in range(view.L)]
        view.commit()
        return arrays, view.center

    @staticmethod
    def state_from_arrays(arrays) -> MatrixProductState:
        """An MPS from ``(left, phys, right)`` site arrays, two sites or more."""
        L = len(arrays)
        data = [arrays[0][0].T]  # (1, p, r) -> (r, p)
        data += [array.transpose(0, 2, 1) for array in arrays[1 : L - 1]]
        data.append(arrays[-1][:, :, 0])  # (l, p, 1) -> (l, p)
        return MatrixProductState(data, shape="lrp")

    def copy_state(self, state_circuit: MatrixProductState):
        """
        An independent copy of the state.

        On the numpy backend the copy has its own tensors but shares their
        arrays: the windowed kernels, like quimb's own operations, replace a
        tensor's array rather than write into it, so neither state can change
        the other. The cached canonical form is carried over, so the copy
        needs no re-canonicalization. Other backends get a deep copy.
        """
        if isinstance(state_circuit, ArrayMPS):
            return state_circuit.copy()
        if not self._fast:
            return copy.deepcopy(state_circuit)
        state = state_circuit.copy()
        gauge = getattr(state_circuit, _GAUGE_ATTR, None)
        if gauge is not None:
            setattr(state, _GAUGE_ATTR, gauge)
        return state

    def working_state(self, state_circuit: MatrixProductState):
        """
        On the numpy backend, the state as an :class:`ArrayMPS` sharing its
        arrays, so the MPS itself is left untouched; otherwise the state.
        """
        if not self._fast or isinstance(state_circuit, ArrayMPS):
            return state_circuit
        view = _CanonicalMPS(state_circuit)
        arrays = [view[i] for i in range(view.L)]
        view.commit()
        return ArrayMPS(arrays, view.center)

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
        :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.tail_truncation` to
        quantify it.
        """
        if self._fast:
            return self.conjugate_operator_many(
                operator, [(generator, angle)], max_bond_dimension
            )
        rotation = self._pauli_exp(generator, angle)
        rotation_dag = self._pauli_exp(generator, -angle)

        operator = rotation_dag.apply(
            operator, compress=True, max_bond=max_bond_dimension
        )
        return operator.apply(rotation, compress=True, max_bond=max_bond_dimension)

    def conjugate_operator_many(
        self,
        operator: MatrixProductOperator,
        rotations,
        max_bond_dimension: int | None,
    ):
        """
        Conjugate ``operator`` by each ``(generator, angle)`` rotation in turn,
        the first one innermost, as repeated :meth:`conjugate_operator` calls.

        On the numpy backend the MPO is unpacked once into canonical site
        arrays, and each conjugation touches only its generator's window
        (:func:`~._mps_kernels.operator_conjugation`), with the same truncation
        rule as quimb's ``compress``, so a whole tail costs about as much as
        evolving a state through it. Other backends go through quimb.
        """
        if not self._fast:
            return super().conjugate_operator_many(
                operator, rotations, max_bond_dimension
            )
        cutoff = _QUIMB_DEFAULT_CUTOFF if self.svd_cutoff is None else self.svd_cutoff
        sites = _mpo_to_sites(operator)
        center = kernels.move_center(sites, len(sites) - 1, 0)
        for generator, angle in rotations:
            center = kernels.operator_conjugation(
                sites,
                center,
                generator.upper(),
                float(angle),
                max_bond=max_bond_dimension,
                cutoff=cutoff,
            )
        return _sites_to_mpo(sites, operator)

    def bond_dimension(self, state_circuit: MatrixProductState) -> int:
        """The largest bond dimension of the MPS."""
        return state_circuit.max_bond()

    def operator_site_arrays(self, operator: MatrixProductOperator) -> list:
        """
        The MPO's site tensors as numpy arrays ordered ``(left, right, k, b)``,
        ``k`` the upper (output) and ``b`` the lower (input) index.
        """
        arrays = []
        for site in range(operator.L):
            tensor = operator[site]
            left = list(bonds(operator[site - 1], tensor)) if site > 0 else []
            right = (
                list(bonds(tensor, operator[site + 1])) if site < operator.L - 1 else []
            )
            order = left + right + [operator.upper_ind(site), operator.lower_ind(site)]
            array = np.asarray(tensor.transpose(*order).data, dtype=complex)
            if not left:
                array = array[None, ...]
            if not right:
                array = array[:, None, ...]
            arrays.append(array)
        return arrays

    def pauli_sum_mpo(self, terms: dict) -> MatrixProductOperator:
        """
        The MPO of ``sum_P c_P P``.

        Raises:
            ValueError: if ``terms`` is empty.
        """
        if not terms:
            raise ValueError("Cannot build the MPO of an empty Pauli sum.")
        mpos = [c * self.pauli_mpo(p) for p, c in terms.items()]
        total = mpos[0]
        for mpo in mpos[1:]:
            total = total + mpo
        return total

    def minimize_energy(
        self,
        operator: MatrixProductOperator,
        initial_state: MatrixProductState,
        bond_dims: list,
        cutoff: float,
        tol: float,
        max_sweeps: int,
        verbosity: int,
    ) -> dict:
        """Two-site DMRG with quimb's ``DMRG2``, on a copy of ``initial_state``."""
        dmrg = DMRG2(
            operator,
            bond_dims=bond_dims,
            cutoffs=cutoff,
            p0=copy.deepcopy(initial_state),
        )
        converged = dmrg.solve(tol=tol, max_sweeps=max_sweeps, verbosity=verbosity)
        return {
            "state": dmrg.state,
            "energy": float(np.real(dmrg.energy)),
            "converged": converged,
            "energy_history": [
                float(np.real(e)) for e in getattr(dmrg, "energies", None) or []
            ],
        }

    def circuit_truncation_fidelity(
        self, nqubits: int, circuit, max_bond_dimension: int | None
    ) -> float:
        """quimb's ``CircuitMPS.fidelity_estimate`` after running ``circuit``."""
        return _qibo_circuit_to_quimb(
            nqubits=nqubits, qibo_circ=circuit, max_bond=max_bond_dimension
        ).fidelity_estimate()
