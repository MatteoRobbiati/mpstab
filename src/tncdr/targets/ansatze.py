from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from qibo import Circuit, gates

@dataclass
class Ansatz(ABC):
    """Abstract ansatz to generate quantum states."""
    nqubits: int

    def __post_init__(
        self,
    ):
        self._circuit = Circuit(self.nqubits)

    @property
    def circuit(
        self,
    ):
        return self._circuit
    
    @property
    def nparams(self):
        return len(self.circuit.get_parameters())
    
    def sample_random_state(self, random_seed: int = 42):
        """Sample circuit's params in [-pi, pi], execute and return state."""
        np.random.seed(random_seed)
        self.circuit.set_parameters(np.random.uniform(-np.pi, np.pi, self.nparams))
        return self.circuit().state()

    def sample_random_quasi_clifford_state(self, cliff_fraction: float = 0.7, random_seed: int = 42):
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
                new_parameters.append(np.random.randint(-5, 6) * np.pi / 2)
        self.circuit.set_parameters(new_parameters)
        return self.circuit().state() 


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
    

