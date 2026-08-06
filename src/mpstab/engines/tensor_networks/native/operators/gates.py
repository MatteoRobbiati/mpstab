"""Quantum gates as MPOs, for the pure-Python tensor network."""

from typing import Union

import numpy as np

from mpstab.engines.stabilizers.native.pauli_string import Pauli
from mpstab.engines.tensor_networks.native.operators import MPO
from mpstab.engines.tensor_networks.native.operators.utils import (
    S_TENSORS,
    theta_state,
    x_state,
)
from mpstab.pauli import PAULI_MATRICES


class PauliExp(MPO):
    """
    The MPO for ``exp(-i theta/2 P)``, with ``P`` a Pauli string.

    Built from one S-tensor per site, closed on the left by
    :func:`~mpstab.engines.tensor_networks.native.operators.utils.theta_state`
    (which carries the angle) and on the right by
    :func:`~mpstab.engines.tensor_networks.native.operators.utils.x_state`.
    """

    def __init__(self, pauli_string: Union[Pauli, str], theta: float):
        if type(pauli_string) is str:
            pauli_string = Pauli(pauli_string)

        phase = pauli_string.complex_phase()
        labels = pauli_string.to_string(ignore_phase=True)

        tensors = [S_TENSORS[label] for label in labels[:-1]] or [S_TENSORS[labels[0]]]
        if len(tensors) > 1:
            tensors.append(S_TENSORS[labels[-1]])

        super().__init__(tensors, tensor_prefix=f"exp(-i{theta/2:.2f}{labels})")

        self.add_tensor("Angle", tensor=theta_state(phase * theta))
        self.add_edge("Angle", self.prefix + "0", "tmp", (0, 3))
        self.add_tensor("X", tensor=x_state())
        self.add_edge(
            "X",
            self.prefix + f"{len(labels) - 1}",
            "tmp",
            (0, 3 if len(tensors) > 1 else 2),
        )

        self.contract("Angle", self.prefix + "0", "tmp", self.prefix + "0")
        self.contract(
            "X",
            self.prefix + f"{len(labels) - 1}",
            "tmp",
            self.prefix + f"{len(labels) - 1}",
        )


CNOT = MPO(
    tensors=[
        np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]),
        np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
    ],
    tensor_prefix="CNOT",
)

CNOT_inv = MPO(
    tensors=[
        np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]]),
        np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]),
    ],
    tensor_prefix="CNOT_inv",
)

CZ = MPO(
    tensors=[
        np.array([[[1, 0], [0, 0]], [[0, 0], [0, 1]]]),
        np.array([[[1, 1], [0, 0]], [[0, 0], [1, -1]]]),
    ],
    tensor_prefix="CZ",
)

SWAP = MPO(
    tensors=[
        np.array([[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]]),
        np.array([[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 0, 1]]]),
    ],
    tensor_prefix="SWAP",
)

H = MPO(tensors=[np.array([[1, 1], [1, -1]]) / np.sqrt(2)], tensor_prefix="Hadamard")

X = MPO(tensors=[PAULI_MATRICES["X"]], tensor_prefix="X_gate")

Y = MPO(tensors=[PAULI_MATRICES["Y"]], tensor_prefix="Y_gate")

Z = MPO(tensors=[PAULI_MATRICES["Z"]], tensor_prefix="Z_gate")

S = MPO(tensors=[np.array([[1, 0], [0, 1j]])], tensor_prefix="S_gate")

T = MPO(
    tensors=[np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])],
    tensor_prefix="T_gate",
)
