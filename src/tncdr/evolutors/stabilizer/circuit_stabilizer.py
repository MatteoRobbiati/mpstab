from copy import copy
from typing import Optional

import numpy as np

from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer.tableaus import (
    CNOT,
    CZ,
    GPI2,
    RX,
    RY,
    RZ,
    SWAP,
    H,
    S,
    Sdg,
    Tableau,
    X,
    Y,
    Z,
)
from tncdr.evolutors.stabilizer.utils import (
    _attenuation_factor,
    _single_qubit_pauli_expval,
    _spread_to_sites,
    commute,
)
from tncdr.evolutors.tensor_network.operators.utils import basis


class CircuitPauliBackpropagation:

    def __init__(
        self,
        n: int,
        initial_state: Optional[str | np.ndarray] = None,
        attenuation_threshold: float = 1e-5,
    ):
        # little bit of code duplication here
        if initial_state is None:
            initial_state = n * "0"
        if type(initial_state) is str:
            initial_state = [basis(bit) for bit in initial_state]

        assert n == len(
            initial_state
        ), f"Intial state qubits ({len(initial_state)}) and circuit qubits ({n}) must match."

        self.n = n
        self.initial_state = initial_state
        self.queue = []

        self.attenuation_factor = 1.0
        self.attenuation_threshold = attenuation_threshold

    def cnot(self, control, target):
        """
        Apply a CNOT gate to the circuit
        """
        return self.apply(CNOT(control, target))

    def cz(self, control, target):
        """
        Apply a CZ gate to the circuit
        """
        return self.apply(CZ(control, target))

    def swap(self, control, target):
        """
        Apply a SWAP gate to the circuit
        """
        return self.apply(SWAP(control, target))

    def h(self, qubit):
        """
        Apply a Hadamard gate to the circuit
        """
        self.apply(H(qubit))

    def x(self, qubit):
        """
        Apply a bit flip (X gate) to the circuit
        """
        self.apply(X(qubit))

    def y(self, qubit):
        """
        Apply a bit and phase flip (Y gate) to the circuit
        """
        self.apply(Y(qubit))

    def z(self, qubit):
        """
        Apply a phase flip (Z gate) to the circuit
        """
        self.apply(Z(qubit))

    def s(self, qubit):
        """
        Apply a S gate (sqrt(Z)) to the circuit
        """
        self.apply(Sdg(qubit))

    def sdg(self, qubit):
        """
        Apply a S+ gate (Z sqrt(Z)) to the circuit
        """
        self.apply(S(qubit))

    def t(self, qubit):
        """
        Apply a T gate to the circuit
        """
        self.pauli_rot("Z", np.pi / 4, qubits=[qubit])

    def gpi2(self, alpha: float, qubit: int):
        self.apply(GPI2(qubit, -alpha))

    def rz(self, theta: float, qubit: int):
        """
        Apply a RZ(theta) gate to the circuit
        """
        try:
            self.apply(RZ(qubit, -theta))
        except ValueError:
            self.pauli_rot("Z", theta, qubits=[qubit])

    def rx(self, theta: float, qubit: int):
        """
        Apply a RX(theta) gate to the circuit
        """
        try:
            self.apply(RX(qubit, -theta))
        except ValueError:
            self.pauli_rot("X", theta, qubits=[qubit])

    def ry(self, theta: float, qubit: int):
        """
        Apply a RY(theta) gate to the circuit
        """
        try:
            self.apply(RY(qubit, -theta))
        except ValueError:
            self.pauli_rot("Y", theta, qubits=[qubit])

    def pauli_rot(
        self, pauli_generator: Pauli, theta: float, qubits: Optional[list[int]] = None
    ):
        """
        Apply a generic e^(i theta/2 G) gate to the circuit, where G is a Pauli word
        """
        if type(pauli_generator) is str:
            pauli_generator = Pauli(pauli_generator)
        if qubits is not None:
            pauli_generator = _spread_to_sites(pauli_generator, qubits, self.n)

        self.apply((-theta / 2, pauli_generator))

    def pauli_channel(self, qubit: int, px: float, py: float, pz: float):
        """
        Apply a single-qubit noise channel of the form

            N(rho) = (1-p_x-p_y-p_z) rho + p_x X rho X + p_y Y rho Y + p_z Z rho Z

        to the circuit.
        """
        self.apply(
            {
                "qs": [qubit],
                "I": 1,
                "X": 1 - 2 * (py + pz),
                "Y": 1 - 2 * (px + pz),
                "Z": 1 - 2 * (py + px),
            }
        )

    def apply(self, gate):
        self.queue.append(gate)

    def _backpropagate_pauli(self, pauli: Pauli):

        for k, operation in enumerate(reversed(self.queue)):

            if isinstance(operation, Tableau):
                pauli.apply(operation)
                continue

            if isinstance(operation, dict):
                self.attenuation_factor *= _attenuation_factor(pauli, operation)
                if self.attenuation_factor < self.attenuation_threshold:
                    raise InterruptedError

                continue

            if isinstance(operation, tuple):
                if commute(pauli, operation[1]):
                    continue
                return pauli, operation, len(self.queue) - k

            raise ValueError(f"Unrecognized operation {operation}")

        return pauli

    def expval(self, obs: Pauli, sites: Optional[list[int]] = None):

        if sites is None:
            sites = list(range(obs.n))

        self.attenuation_factor = 1
        obs = _spread_to_sites(obs, sites, self.n)
        return self._recursive_expval(obs)

    def _recursive_expval(self, obs):

        try:
            result = self._backpropagate_pauli(obs)
        except InterruptedError:
            return 0

        if isinstance(result, Pauli):

            local_expvals = [
                _single_qubit_pauli_expval(p, s)
                for p, s in zip(result.to_string(ignore_phase=True), self.initial_state)
            ]
            return (
                self.attenuation_factor
                * result.complex_phase()
                * np.prod(local_expvals)
            )

        obs_at_branching, (theta, pauli_generator), k = result

        shorter_circuit = copy(self)
        shorter_circuit.queue = self.queue[: k - 1]

        # needs to be first, because paulis are passed by reference!
        gen_branch = shorter_circuit._recursive_expval(
            obs_at_branching @ pauli_generator
        )

        # reset the attenuation factor
        shorter_circuit.attenuation_factor = self.attenuation_factor
        id_branch = shorter_circuit._recursive_expval(obs_at_branching)

        return (np.cos(theta) ** 2 - np.sin(theta) ** 2) * id_branch + 2j * np.cos(
            theta
        ) * np.sin(theta) * gen_branch
