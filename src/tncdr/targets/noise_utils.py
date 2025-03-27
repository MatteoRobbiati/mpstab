import numpy as np
from qibo.noise import NoiseModel, PauliError, ReadoutError
from qibo import gates

def build_noise_model(
        nqubits:int, 
        readout_bit_flip_prob:float, 
        local_pauli_noise_sigma:float
    ):
    """Costruct noise model as a local Pauli noise channel + readout noise."""
    noise_model = NoiseModel()
    for q in range(nqubits):
        noise_model.add(
            PauliError([
                ("X", np.abs(np.random.normal(0, local_pauli_noise_sigma))),
                ("Y", np.abs(np.random.normal(0, local_pauli_noise_sigma))),
                ("Z", np.abs(np.random.normal(0, local_pauli_noise_sigma)))
            ]),
            qubits=q
        )

    # single_readout_matrix = np.array(
    #     [
    #         [1 - readout_bit_flip_prob, readout_bit_flip_prob],
    #         [readout_bit_flip_prob, 1 - readout_bit_flip_prob]
    #     ]
    # )
    # readout_noise = ReadoutError(single_readout_matrix)
    # noise_model.add(readout_noise, gates.M)
    return noise_model
    