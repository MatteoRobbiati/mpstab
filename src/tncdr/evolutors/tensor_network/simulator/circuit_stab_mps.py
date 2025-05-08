from typing import Optional, Union

import numpy as np

from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer.tableaus import CNOT, CZ, SWAP, H, S, X, Y, Z

from tncdr.evolutors.tensor_network.simulator.circuit_mps import CircuitMPS
from tncdr.evolutors.tensor_network.operators.gates import PauliExp
from tncdr.evolutors.tensor_network.operators.observables import PauliMPO

class CircuitStabilizerMPS(CircuitMPS):

    def __init__(self, n, initial_state = None, max_bond_dimension = None):
        super().__init__(n, initial_state, max_bond_dimension)
        self.cliffords = []

    def cnot(self, control, target):
        self.cliffords.append(CNOT(control, target))
    
    def cz(self, control, target):
        self.cliffords.append(CZ(control, target))
    
    def swap(self, control, target):
        self.cliffords.append(SWAP(control, target))

    def h(self, qubit):
        self.cliffords.append(H(qubit))
    
    def x(self, qubit):
        self.cliffords.append(X(qubit))
     
    def y(self, qubit):
        self.cliffords.append(Y(qubit))
    
    def z(self, qubit):
        self.cliffords.append(Z(qubit))
    
    def s(self, qubit):
        self.cliffords.append(S(qubit))
    
    def t(self, qubit):
        
        p = Pauli(qubit*'I'+'Z'+(self.n_qubits-qubit-1)*'I')
        for tab in reversed(self.cliffords):
            p.apply(tab)

        self.apply(PauliExp(p, np.pi/4))
    
    def expval(self, obs:str, sites:Optional[list[int]]=None):

        obs = Pauli(obs)
        for tab in reversed(self.cliffords):
            obs.apply(tab)

        return super().expval(PauliMPO(obs), sites)