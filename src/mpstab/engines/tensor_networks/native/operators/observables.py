"""Observables as MPOs, for the pure-Python tensor network."""

from typing import Union

import numpy as np

from mpstab.engines.stabilizers.native.pauli_string import Pauli
from mpstab.engines.tensor_networks.native.operators import MPO
from mpstab.pauli import PAULI_MATRICES


class PauliMPO(MPO):
    """A single Pauli string as an MPO, one bond-dimension-1 site per qubit."""

    def __init__(self, pauli_string: Union[Pauli, str]):
        if type(pauli_string) is str:
            pauli_string = Pauli(pauli_string)

        phase = pauli_string.complex_phase()
        labels = pauli_string.to_string(ignore_phase=True)

        # The phase rides on the first site. Boundary sites carry one bond, the
        # rest carry two; a single-qubit string has no bonds at all.
        tensors = [phase * np.reshape(PAULI_MATRICES[labels[0]], (2, 2, 1))]
        tensors += [
            np.reshape(PAULI_MATRICES[label], (2, 2, 1, 1)) for label in labels[1:-1]
        ]
        if len(labels) > 1:
            tensors.append(np.reshape(PAULI_MATRICES[labels[-1]], (2, 2, 1)))
        else:
            tensors[0] = np.squeeze(tensors[0])

        super().__init__(tensors)
