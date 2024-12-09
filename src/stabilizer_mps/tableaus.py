from pauli_string import Pauli

from typing import List

# GENERAL STRUCTURE

class HalfTableau():

    def __init__(self, qubits:List[int], signs:int, conjugates:List[Pauli])->None:
        
        assert len(qubits) == len(conjugates), 'Number of qubits must match the number of conjugates.'
        
        self.qubits = qubits
        self.signs = signs
        self.conjugates = conjugates

    def __repr__(self):
        conj_repr = [f'{q} -> {conj}' for q, conj in zip(self.qubits, self.conjugates)]
        return'\n'.join(conj_repr)

class Tableau():

    def __init__(self, XTableau:HalfTableau, ZTableau:HalfTableau)->None:
        
        assert XTableau.qubits == ZTableau.qubits, 'X and Z tableaus must share the same qubits'
        self.qubits = XTableau.qubits
        self.XTableau = XTableau
        self.ZTableau = ZTableau

    def __repr__(self):
        return f'Z Tableau:\n{self.ZTableau}\nX Tableau\n{self.XTableau}'
        
# IMPLEMENTATIONS

class CNOT(Tableau):

    def __init__(self, control:int, target:int,)->None:

        XTableau = HalfTableau([target, control], signs=0, conjugates=[Pauli('IX'),Pauli('XX')])
        ZTableau = HalfTableau([target, control], signs=0, conjugates=[Pauli('IZ'),Pauli('ZZ')])

        super().__init__(XTableau, ZTableau)

class H(Tableau):

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], signs=0, conjugates=[Pauli('Z')])
        ZTableau = HalfTableau([target], signs=0, conjugates=[Pauli('X')])

        super().__init__(XTableau, ZTableau)

class S(Tableau):

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], signs=0, conjugates=[Pauli('Y')])
        ZTableau = HalfTableau([target], signs=0, conjugates=[Pauli('Z')])

        super().__init__(XTableau, ZTableau)