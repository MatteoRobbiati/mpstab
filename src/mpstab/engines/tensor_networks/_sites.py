"""
Plain site-array MPS and MPO, for engines that keep their own arrays.

An engine whose states are just lists of site arrays (on any array module:
numpy, cupy, ...) can use :class:`SiteMPS` as its state and :class:`SiteMPO`
as its operator: :class:`SiteMPS` is already the canonical site view the shared
algorithms of :class:`~mpstab.engines.TensorNetworkEngine` run on, so the
engine's :meth:`~mpstab.engines.TensorNetworkEngine.state_sites` returns the
state itself.
"""

from __future__ import annotations

import numpy as np
from qibo.backends import NumpyBackend

from mpstab.pauli import PAULI_MATRICES


class SiteMPS:
    """
    An MPS as ``(left, phys, right)`` site arrays in mixed canonical form
    around ``center``.

    The shared kernels replace site arrays rather than write into them, so
    :meth:`copy` shares the arrays and only copies the list.
    """

    __slots__ = ("sites", "center")

    def __init__(self, sites: list, center: int = 0):
        self.sites = list(sites)
        self.center = center

    @property
    def L(self) -> int:
        return len(self.sites)

    def __len__(self) -> int:
        return len(self.sites)

    def __getitem__(self, i: int):
        return self.sites[i]

    def __setitem__(self, i: int, array):
        self.sites[i] = array

    def commit(self):
        """Nothing to write back: the state is its own site view."""

    def copy(self) -> SiteMPS:
        """An independent copy sharing the (never written into) arrays."""
        return SiteMPS(self.sites, self.center)

    def max_bond(self) -> int:
        """The largest bond dimension."""
        return max((site.shape[2] for site in self.sites[:-1]), default=1)


class SiteMPO:
    """
    An MPO as ``(left, 4, right)`` site arrays, the middle index
    ``2 * out + in``, with the Pauli string it stands for when it is one.
    """

    __slots__ = ("sites", "pauli")

    def __init__(self, sites: list, pauli: str | None = None):
        self.sites = list(sites)
        self.pauli = pauli

    @property
    def L(self) -> int:
        return len(self.sites)

    def max_bond(self) -> int:
        """The largest bond dimension."""
        return max((site.shape[2] for site in self.sites[:-1]), default=1)


def pauli_mpo_sites(pauli: str) -> list[np.ndarray]:
    """The bond-1 ``(1, 4, 1)`` site arrays of a Pauli string, on numpy."""
    return [
        np.asarray(PAULI_MATRICES[label], dtype=complex).reshape(1, 4, 1)
        for label in pauli
    ]


#: Gate matrices are read on qibo's numpy backend, whatever the global one.
_QIBO_NUMPY = NumpyBackend()


def _gate_matrix(gate) -> np.ndarray:
    """A qibo gate's matrix as a numpy array."""
    return np.asarray(gate.matrix(_QIBO_NUMPY), dtype=complex)


def product_state_sites(n: int, circuit) -> list[np.ndarray]:
    """
    The ``(1, 2, 1)`` site arrays of the product state a qibo circuit of
    one-qubit gates prepares from ``|0...0>``, on numpy. Measurement gates
    are skipped.

    Raises:
        ValueError: if a gate acts on more than one qubit.
    """
    unitaries = [np.eye(2, dtype=complex) for _ in range(n)]
    for gate in circuit.queue:
        if getattr(gate, "name", None) == "measure":
            continue
        if len(gate.qubits) != 1:
            raise ValueError(
                "Only product initial states are supported: the circuit has "
                f"{gate.name} on qubits {gate.qubits}."
            )
        (qubit,) = gate.qubits
        unitaries[qubit] = _gate_matrix(gate) @ unitaries[qubit]
    return [u[:, 0].reshape(1, 2, 1) for u in unitaries]
