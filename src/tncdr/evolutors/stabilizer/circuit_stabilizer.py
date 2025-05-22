from typing import Optional

import numpy as np

from tncdr.evolutors.tensor_network.operators.utils import basis

from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer.tableaus import Tableau, CNOT, CZ, SWAP, H, S, Sdg, X, Y, Z, RX, RY, RZ, GPI2
from tncdr.evolutors.stabilizer.utils import commute, _single_qubit_pauli_expval, _spread_to_sites

class CircuitPauliBackpropagation():

    def __init__(
            self,
            n:int,
            initial_state:Optional[str | np.ndarray] = None
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
        self.pauli_rot('Z', np.pi/4, qubits=[qubit])
    
    def gpi2(self, alpha:float, qubit:int):
        self.apply(GPI2(qubit, -alpha))

    def rz(self, theta:float, qubit:int):
        """
        Apply a RZ(theta) gate to the circuit
        """
        try:
            self.apply(RZ(qubit, -theta))
        except ValueError:
            self.pauli_rot('Z', theta, qubits=[qubit])

    def rx(self, theta:float, qubit:int):
        """
        Apply a RX(theta) gate to the circuit
        """
        try:
            self.apply(RX(qubit, -theta))
        except ValueError:
            self.pauli_rot('X', theta, qubits=[qubit])

    def ry(self, theta:float, qubit:int):
        """
        Apply a RY(theta) gate to the circuit
        """
        try:
            self.apply(RY(qubit, -theta))
        except ValueError:
            self.pauli_rot('Y', theta, qubits=[qubit])


    def pauli_rot(self, pauli_generator:Pauli, theta:float, qubits:Optional[list[int]] = None):

        if type(pauli_generator) is str: pauli_generator = Pauli(pauli_generator)
        if qubits is not None: pauli_generator = _spread_to_sites(pauli_generator, qubits, self.n)

        self.apply((-theta/2, pauli_generator))

    def apply(self, gate):
        self.queue.append(gate)

    def _backpropagate_pauli(self, pauli:Pauli, sites:list[int]):
        
        pauli = _spread_to_sites(pauli, sites, self.n)
        for k, gate in enumerate(reversed(self.queue)):

            if isinstance(gate, Tableau): 
                pauli.apply(gate)

            if isinstance(gate, tuple):
                if commute(pauli, gate[1]): continue
                return pauli, gate, len(self.queue)-k
        
        return pauli

    def expval(self, obs: Pauli, sites: Optional[list[int]] = None):

        if sites is None: 
            sites=list(range(obs.n))
        
        result = self._backpropagate_pauli(obs, sites)

        if isinstance(result, Pauli):

            local_expvals = [_single_qubit_pauli_expval(p, s) for p,s in zip(result.to_string(ignore_phase=True), self.initial_state)]
            return result.complex_phase()*np.prod(local_expvals)

        obs_at_branching, (theta, pauli_generator), k = result

        shorter_circuit = CircuitPauliBackpropagation(self.n, self.initial_state)
        shorter_circuit.queue = self.queue[:k-1]

        gen_branch_obs = obs_at_branching@pauli_generator
        
        id_branch_coeff = (np.cos(theta)**2 - np.sin(theta)**2)
        gen_branch_coeff = 2j*np.cos(theta)*np.sin(theta)
        
        return id_branch_coeff*shorter_circuit.expval(obs_at_branching)+gen_branch_coeff*shorter_circuit.expval(gen_branch_obs)

        
