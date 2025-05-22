import numpy as np

from tncdr.evolutors.stabilizer.pauli_string import Pauli


def _single_qubit_pauli_expval(pauli_descriptor, statevector):

    if pauli_descriptor == "I":
        return 1.0
    if pauli_descriptor == "Z":
        return np.abs(statevector[0]) - np.abs(statevector[1])
    if pauli_descriptor == "X":
        return np.real(np.conj(statevector[0]) * statevector[1])
    if pauli_descriptor == "Y":
        return np.imag(np.conj(statevector[0]) * statevector[1])

    raise ValueError(f"Unknown Pauli descriptor {pauli_descriptor} given.")


def commute(p1: Pauli, p2: Pauli):
    return (p1 @ p2).phase == (p2 @ p1).phase


def _attenuation_factor(pauli: Pauli, noise_table: dict):
    pauli = pauli.to_string(ignore_phase=True)
    return np.prod(
        [noise_table[p] if q in noise_table["qs"] else 1 for q, p in enumerate(pauli)]
    )


def _spread_to_sites(pauli: Pauli, sites: list[int], n: int):

    p_str = pauli.to_string(ignore_phase=True)
    p_phase = pauli.phase

    new_p_str = ""
    for i in range(n):
        if i in sites:
            new_p_str += p_str[sites.index(i)]
        else:
            new_p_str += "I"

    pauli = Pauli(new_p_str)
    pauli.phase = p_phase

    return pauli
