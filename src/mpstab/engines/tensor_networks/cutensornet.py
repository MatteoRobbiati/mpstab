"""
Tensor-network engine on NVIDIA GPUs, backed by cuTensorNet.

States and operators live on the GPU as CuPy site arrays (:class:`SiteMPS`,
:class:`SiteMPO`). The engine supplies only the representation hooks of
:class:`~mpstab.engines.TensorNetworkEngine`; every algorithm (windowed Pauli
rotations, expectation values, operator conjugation, transition amplitudes) is
the shared one, run on the GPU through a :class:`~._mps_kernels.Linalg` whose
QR and SVD are cuTensorNet's tensor decompositions
(``cuquantum.tensornet.tensor.decompose``). Expectation values of general
operators are contracted as one network by cuTensorNet's ``contract``, with
its path optimizer.

Truncation uses the same ``"rsum2"`` rule as :class:`~mpstab.engines.QuimbEngine`
(the singular values of each split are read back to the host to pick the kept
rank), so both engines produce the same states up to floating-point roundoff.

Needs ``cupy`` and ``cuquantum``, both imported when the engine is first
built: ``pip install "mpstab[cutensornet]"`` (CUDA 12 wheels). The engine runs
on CuPy's current device; select another GPU with
``cupy.cuda.Device(i).use()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.engines.tensor_networks._sites import (
    SiteMPO,
    SiteMPS,
    pauli_mpo_sites,
    product_state_sites,
)
from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine

# cupy and cuquantum are optional dependencies. cuquantum moved its Python API
# from ``cuquantum.cutensornet`` (23.x-24.x) to ``cuquantum.tensornet`` (25.x).
try:
    import cupy
except ImportError:
    cupy = None
try:
    from cuquantum.tensornet import contract
    from cuquantum.tensornet.tensor import QRMethod, SVDMethod, decompose
except ImportError:
    try:
        from cuquantum import contract
        from cuquantum.cutensornet.tensor import QRMethod, SVDMethod, decompose
    except ImportError:
        contract = decompose = QRMethod = SVDMethod = None
try:
    from cuquantum.bindings import cutensornet as _cutensornet_bindings
except ImportError:
    try:
        from cuquantum import cutensornet as _cutensornet_bindings
    except ImportError:
        _cutensornet_bindings = None

_INSTALL_HINT = (
    "CuTensorNetEngine needs cupy and cuquantum (NVIDIA cuTensorNet): "
    'pip install "mpstab[cutensornet]"'
)


@dataclass(frozen=True)
class CuQuantumAPI:
    """
    The cupy and cuTensorNet entry points the engine uses: everything it
    calls on the GPU goes through these.

    :class:`CuTensorNetEngine` builds one from the installed packages; passing
    another (``cuquantum=``) runs the engine on any implementation with the
    same signatures, e.g. a NumPy stand-in to test the engine without a GPU.
    """

    cupy: Any
    contract: Callable
    decompose: Callable
    QRMethod: type
    SVDMethod: type
    create_handle: Callable | None = None
    destroy_handle: Callable | None = None


def _load_cuquantum() -> CuQuantumAPI:
    """
    cupy and cuTensorNet's Python API, as imported at module load.

    Raises:
        ImportError: if either package is missing.
    """
    if cupy is None or decompose is None:
        raise ImportError(_INSTALL_HINT)
    create_handle = getattr(_cutensornet_bindings, "create", None)
    destroy_handle = getattr(_cutensornet_bindings, "destroy", None)
    return CuQuantumAPI(
        cupy, contract, decompose, QRMethod, SVDMethod, create_handle, destroy_handle
    )


class CuTensorNetLinalg(kernels.Linalg):
    """
    The kernels' factorizations on the GPU: CuPy arrays, QR and SVD by
    cuTensorNet's tensor decompositions. The truncated splits are the
    :class:`~._mps_kernels.Linalg` defaults, so truncation follows the shared
    ``"rsum2"`` rule.
    """

    def __init__(self, cuquantum: CuQuantumAPI, reuse_handle: bool = True):
        self._cq = cuquantum
        self.xp = cuquantum.cupy
        self.reuse_handle = reuse_handle and cuquantum.create_handle is not None
        #: One cuTensorNet handle per device, created on first use there.
        self._handles: dict = {}

    @property
    def options(self) -> dict | None:
        """
        The options of every cuTensorNet call: the current device's handle,
        so switching devices with ``cupy.cuda.Device(i).use()`` is followed.
        """
        if not self.reuse_handle:
            return None
        device = self.xp.cuda.runtime.getDevice()
        handle = self._handles.get(device)
        if handle is None:
            handle = self._handles[device] = self._cq.create_handle()
        return {"handle": handle}

    def close(self):
        """Destroy the handles this backend created."""
        handles, self._handles = self._handles, {}
        if self._cq.destroy_handle is None:
            return
        for device, handle in handles.items():
            with self.xp.cuda.Device(device):
                self._cq.destroy_handle(handle)

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown: CUDA may already be gone
            pass

    def asarray(self, array):
        return self.xp.asarray(array)

    def to_numpy(self, array) -> np.ndarray:
        return self.xp.asnumpy(array)

    def qr(self, matrix):
        q, r = self._cq.decompose(
            "ij->ik,kj", matrix, method=self._cq.QRMethod(), options=self.options
        )
        return q, r

    def svd(self, matrix):
        u, s, vh = self._cq.decompose(
            "ij->ik,kj", matrix, method=self._cq.SVDMethod(), options=self.options
        )
        return u, s, vh


class CuTensorNetEngine(TensorNetworkEngine):
    """
    MPS evolution and MPO expectation values on an NVIDIA GPU via cuTensorNet.

    It covers what :class:`~mpstab.evolutors.hsmpo.HSMPO` and
    :class:`~mpstab.evolutors.hqsmpo.HQSMPO` need, including the head/tail
    split and the noisy estimators' transition amplitudes. DMRG and the
    plain-MPS baseline :attr:`~mpstab.evolutors.hsmpo.HSMPO.truncation_fidelity_pure_tn`
    stay with :class:`~mpstab.engines.QuimbEngine`.

    Initial states must be product states (one-qubit gates only), which is
    all :class:`~mpstab.evolutors.hsmpo.HSMPO` builds.
    """

    #: CUDA cannot be used in a child forked after the parent initialized it.
    process_start_method = "spawn"

    def __init__(
        self,
        svd_cutoff: float | None = None,
        dtype: str = "complex128",
        reuse_handle: bool = True,
        cuquantum: CuQuantumAPI | None = None,
    ):
        """
        Args:
            svd_cutoff: singular-value cutoff (``"rsum2"`` mode, relative) when
                compressing. ``None`` keeps quimb's default (``1e-10``), so
                results match :class:`~mpstab.engines.QuimbEngine`; ``0.0``
                truncates only through ``max_bond_dimension``.
            dtype: ``"complex128"`` or ``"complex64"``. Single precision halves
                memory and is markedly faster on consumer GPUs, at about
                ``1e-7`` relative accuracy.
            reuse_handle: create one cuTensorNet handle per device, on first
                use, and pass it to every call instead of letting each
                decomposition create its own. Handles are destroyed with the
                engine.
            cuquantum: the cupy and cuTensorNet functions to run on, or
                ``None`` to use the installed packages. Must pickle for
                ``noisy_expectation`` with ``n_workers``.

        Raises:
            ImportError: if cupy or cuquantum is not installed.
            ValueError: on an unsupported ``dtype``.
        """
        if dtype not in ("complex128", "complex64"):
            raise ValueError(
                f"Unsupported dtype {dtype!r}, expected 'complex128' or 'complex64'."
            )
        self.svd_cutoff = svd_cutoff
        self.dtype = dtype
        self.reuse_handle = reuse_handle
        self._api = cuquantum
        self._setup()

    def _setup(self):
        """Import the GPU stack and build the linear-algebra backend."""
        self._cq = self._api if self._api is not None else _load_cuquantum()
        self._linalg = CuTensorNetLinalg(self._cq, self.reuse_handle)

    def __getstate__(self):
        # Modules and handles do not pickle, which sending the engine to
        # another process (noisy_expectation with n_workers) needs: the child
        # imports the packages again, or unpickles an injected API.
        state = self.__dict__.copy()
        del state["_cq"], state["_linalg"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._setup()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(svd_cutoff={self.svd_cutoff!r}, "
            f"dtype={self.dtype!r})"
        )

    @property
    def linalg(self) -> CuTensorNetLinalg:
        return self._linalg

    def _device(self, array):
        """A host array on the GPU, in the engine's dtype."""
        return self._cq.cupy.asarray(array, dtype=self.dtype)

    # -- representation --------------------------------------------------

    def build_circuit_mps(
        self,
        n: int,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ) -> SiteMPS:
        """
        The product state a qibo circuit of one-qubit gates prepares, as a
        bond-1 :class:`SiteMPS` on the GPU (every site is normalized, so any
        center is canonical).

        Raises:
            ValueError: if the circuit entangles any pair of qubits.
        """
        sites = product_state_sites(n, initial_state_circuit)
        return SiteMPS([self._device(site) for site in sites], center=0)

    def pauli_mpo(self, pauli_string: str | object) -> SiteMPO:
        """The bond-1 :class:`SiteMPO` of a Pauli string."""
        pauli = "".join(pauli_string).upper()
        return SiteMPO([self._device(s) for s in pauli_mpo_sites(pauli)], pauli)

    def pauli_string_of(self, operator: SiteMPO) -> str | None:
        return operator.pauli if isinstance(operator, SiteMPO) else None

    def state_sites(self, state_circuit: SiteMPS) -> SiteMPS:
        """The state itself: a :class:`SiteMPS` is its own canonical view."""
        if not isinstance(state_circuit, SiteMPS):
            raise TypeError(
                f"{type(self).__name__} states are SiteMPS, got "
                f"{type(state_circuit).__name__}."
            )
        return state_circuit

    def state_from_sites(self, sites: list, center: int) -> SiteMPS:
        """A :class:`SiteMPS` on the given site arrays."""
        return SiteMPS(sites, center)

    def operator_sites(self, operator: SiteMPO) -> list:
        return list(operator.sites)

    def operator_from_sites(self, sites: list, like: Any) -> SiteMPO:
        return SiteMPO(sites)

    def operator_site_arrays(self, operator: SiteMPO) -> list:
        """The MPO's sites as numpy ``(left, right, out, in)`` arrays."""
        arrays = []
        for site in operator.sites:
            left, _, right = site.shape
            arrays.append(
                self._cq.cupy.asnumpy(
                    site.reshape(left, 2, 2, right).transpose(0, 3, 1, 2)
                ).astype(complex)
            )
        return arrays

    def copy_state(self, state_circuit: SiteMPS) -> SiteMPS:
        """A copy sharing the site arrays, which no operation writes into."""
        return state_circuit.copy()

    def bond_dimension(self, state_circuit: SiteMPS) -> int:
        return state_circuit.max_bond()

    # -- cuTensorNet contraction -------------------------------------------

    def _operator_expectation(self, state_sites, operator_sites) -> complex:
        """
        Unnormalized ``<psi|O|psi>`` as one network, ``bra - MPO - ket``,
        contracted by cuTensorNet with its own path optimizer.
        """
        L = len(state_sites)

        def mode(kind: int, j: int) -> int:
            # kinds: 0 bra bond, 1 operator bond, 2 ket bond, 3 out, 4 in
            return kind * (L + 1) + j

        operands = []
        for j in range(L):
            ket = state_sites[j]
            site = operator_sites[j]
            left, _, right = site.shape
            operands += [
                ket.conj(),
                [mode(0, j), mode(3, j), mode(0, j + 1)],
                site.reshape(left, 2, 2, right),
                [mode(1, j), mode(3, j), mode(4, j), mode(1, j + 1)],
                ket,
                [mode(2, j), mode(4, j), mode(2, j + 1)],
            ]
        # The open boundary bonds are size 1: list them as output modes and
        # read the single entry.
        boundary = [mode(k, 0) for k in range(3)] + [mode(k, L) for k in range(3)]
        options = self._linalg.options
        value = self._cq.contract(*operands, boundary, options=options)
        return complex(value.reshape(-1)[0])
