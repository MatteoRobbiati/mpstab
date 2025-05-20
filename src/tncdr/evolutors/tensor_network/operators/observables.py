from typing import Union

import numpy as np

from tncdr.evolutors.stabilizer.pauli_string import Pauli

from tncdr.evolutors.tensor_network.operators import MPO
from tncdr.evolutors.tensor_network.tn_utils import paulis


class PauliMPO(MPO):
    """
    Implementation as an MPO of a single Pauli string
    """

    def __init__(self, pauli_string: Union[Pauli, str]):

        if type(pauli_string) is str:
            pauli_string = Pauli(pauli_string)

        phase = pauli_string.complex_phase()
        desc = pauli_string.to_string(ignore_phase=True)

        tensors = [phase * np.reshape(paulis[desc[0]], (2, 2, 1))]
        for d in desc[1:-1]:
            tensors.append(np.reshape(paulis[d], (2, 2, 1, 1)))

        if len(desc) > 1:
            tensors.append(np.reshape(paulis[desc[-1]], (2, 2, 1)))
        else:
            tensors[0] = np.squeeze(tensors[0])

        return super().__init__(tensors)
