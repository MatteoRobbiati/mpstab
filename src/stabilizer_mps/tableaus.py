from pauli_string import Pauli

from typing import List

# GENERAL STRUCTURE

class HalfTableau():
    """
    Stores the rules to update either the X or Z components of a Pauli in the XZ encoding
    """

    def __init__(self, qubits:List[int], signs:List[int], conjugates:List[Pauli])->None:
        
        assert len(qubits) == len(conjugates), 'Number of qubits must match the number of conjugates.'
        
        self.qubits = qubits # Qubits over which we act on with the Tableau 
        self.signs = signs # List of the signs (phases) to be added to the updated strings
        self.conjugates = conjugates # Updated strings after the application of the operation

    def __repr__(self):

        conj_repr = []
        for s, q, conj in zip(self.signs, self.qubits, self.conjugates):
            sign = '-' if s else '+'
            conj_repr.append(f'{q} -> {sign}{conj}')
        return'\n'.join(conj_repr)

class Tableau():
    """
    Full Tableau, stores the rules to update both the X or Z components of a Pauli in the XZ encoding.
    """

    def __init__(self, XTableau:HalfTableau, ZTableau:HalfTableau, name:str|None = None)->None:
        
        assert XTableau.qubits == ZTableau.qubits, 'X and Z tableaus must share the same qubits'
        self.qubits = XTableau.qubits
        self.XTableau = XTableau
        self.ZTableau = ZTableau
        self.name = name

    def __repr__(self):
        return f'Z Tableau:\n{self.ZTableau}\nX Tableau\n{self.XTableau}'
        
# IMPLEMENTATIONS

class CNOT(Tableau):
    """
    Implements the Controlled-NOT (CNOT) Tableau
    """

    def __init__(self, control:int, target:int,)->None:

        XTableau = HalfTableau([control, target], signs=[0, 0], conjugates=[Pauli('XX'),Pauli('XI')])
        ZTableau = HalfTableau([control, target], signs=[0, 0], conjugates=[Pauli('ZZ'),Pauli('ZI')])

        super().__init__(XTableau, ZTableau, name=f'CNOT({control}->{target})')

class H(Tableau):
    """
    Implements the Hadamard (H) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], signs=[0], conjugates=[Pauli('Z')])
        ZTableau = HalfTableau([target], signs=[0], conjugates=[Pauli('X')])

        super().__init__(XTableau, ZTableau, name=f'H({target})')

class S(Tableau):
    """
    Implements the Phase-gate (S) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], signs=[0], conjugates=[Pauli('Y')])
        ZTableau = HalfTableau([target], signs=[0], conjugates=[Pauli('Z')])

        super().__init__(XTableau, ZTableau, name=f'S({target})')