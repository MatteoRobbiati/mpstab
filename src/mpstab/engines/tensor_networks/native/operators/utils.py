"""Single-qubit states and Pauli S-tensors used to build the native MPOs."""

from functools import lru_cache

import numpy as np

from mpstab.engines.tensor_networks.native.tensor_network import TensorNetwork
from mpstab.pauli import PAULI_LABELS

_BASIS_STATES = {
    "0": np.array([1.0, 0.0]),
    "1": np.array([0.0, 1.0]),
    "+": np.array([1.0, 1.0]) / np.sqrt(2),
    "-": np.array([1.0, -1.0]) / np.sqrt(2),
}


def basis(state_name: str) -> np.ndarray:
    """A single-qubit basis state, one of ``"0"``, ``"1"``, ``"+"`` or ``"-"``."""
    if state_name not in _BASIS_STATES:
        raise ValueError(
            f'Unknown basis state "{state_name}", expected one of 0, 1, + or -.'
        )
    return _BASIS_STATES[state_name]


@lru_cache(maxsize=None)
def theta_state(theta: float) -> np.ndarray:
    """The state ``RY(theta)|0>``, which carries a rotation's angle into an MPO."""
    return np.array([np.cos(theta / 2), -1j * np.sin(theta / 2)])


@lru_cache(maxsize=None)
def x_state() -> np.ndarray:
    """The un-normalised ``|+>`` state, which closes a rotation MPO's bond."""
    return np.array([1.0, 1.0])


def _s_tensor(pauli: str) -> np.ndarray:
    """
    The S-tensor of a Pauli matrix: an ``(I, P)`` pair joined to a copy tensor.

    Contracting one S-tensor per site against :func:`theta_state` and
    :func:`x_state` builds ``cos(theta/2) I - i sin(theta/2) P`` as an MPO. Axes
    are ordered ``(i, j, alpha, beta)``: ``i`` and ``j`` index the Pauli matrix in
    the Z basis, ``alpha`` and ``beta`` are the copy tensor's bonds.
    """
    tn = TensorNetwork()
    tn.add_pauli_pair("gamma", p0="I", p1=pauli)
    tn.add_copy_tensor("copy", n=2)

    tn.add_edge("gamma", "copy", "link", (0, 0))
    tn.contract("gamma", "copy", "link", "S")
    return tn.tensornet.nodes["S"]["tensor"]


#: One :func:`_s_tensor` per Pauli label, built once at import.
S_TENSORS = {label: _s_tensor(label) for label in PAULI_LABELS}
