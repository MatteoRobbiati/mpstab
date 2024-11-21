from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qibo import Circuit, gates

@dataclass
class Ansatz(ABC):
    """Abstract ansatz to generate quantum states."""
    nqubits: int
    density_matrix: bool = False

    def __post_init__(
        self,
    ):
        self._circuit = Circuit(nqubits=self.nqubits, density_matrix=self.density_matrix)

    @property
    def circuit(
        self,
    ):
        return self._circuit
    
    @property
    def nparams(self):
        return len(self.circuit.get_parameters())
    
    def random_unitary(self, random_seed: int = 42):
        """Sample circuit's params in [-pi, pi], execute and return state."""
        np.random.seed(random_seed)
        self.circuit.set_parameters(np.random.uniform(-np.pi, np.pi, self.nparams))
        return self.circuit

    def random_quasi_clifford_unitary(self, cliff_fraction: float = 0.7, random_seed: int = 42):
        """
        Sample circuit's params so that a portion `0 < cliff_fraction < 1` 
        of the gates are cliffordized by setting an angle which is a multiple
        of pi/2.
        """
        np.random.seed(random_seed)
        new_parameters = []
        for _ in range(self.nparams):
            coin = np.random.uniform(0,1)
            if coin >= cliff_fraction:
                new_parameters.append(np.random.uniform(-np.pi, np.pi))
            else:
                new_parameters.append(np.random.randint(-2, 3) * np.pi / 2)
        self.circuit.set_parameters(new_parameters)
        return self.circuit
    
    def execute(self, nshots: int = None):
        """Execute the circuit and return the outcome."""
        return self.circuit(nshots=nshots)

@dataclass
class HardwareEfficient(Ansatz):
    """Hardware Efficient ansatz."""
    nlayers: int = 1

    def __post_init__(self):
        super().__post_init__()
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=0.))
                self.circuit.add(gates.RZ(q=q, theta=0.))
            [ self.circuit.add(gates.CNOT(q0=q%self.nqubits, q1=(q+1)%self.nqubits)) for q in range(self.nqubits) ]
        self.circuit.add(gates.M(*range(self.nqubits)))
    

