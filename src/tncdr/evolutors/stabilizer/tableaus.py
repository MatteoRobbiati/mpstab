import math
from typing import List

from tncdr.evolutors.stabilizer.pauli_string import Pauli

# GENERAL STRUCTURE

class HalfTableau():
    """
    Stores the rules to update either the X or Z components of a Pauli in the XZ encoding
    """

    def __init__(self, qubits:List[int], conjugates:List[Pauli])->None:
        
        assert len(qubits) == len(conjugates), 'Number of qubits must match the number of conjugates.'
        
        self.qubits = qubits # Qubits over which we act on with the Tableau 
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

        XTableau = HalfTableau([control, target], conjugates=[Pauli('X'),Pauli('XX')])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli('ZZ'),Pauli('ZI')])

        super().__init__(XTableau, ZTableau, name=f'CNOT({control}->{target})')

class H(Tableau):
    """
    Implements the Hadamard (H) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], conjugates=[Pauli('Z')])
        ZTableau = HalfTableau([target], conjugates=[Pauli('X')])

        super().__init__(XTableau, ZTableau, name=f'H({target})')

class S(Tableau):
    """
    Implements the Phase-gate (S) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], conjugates=[Pauli('Y')])
        ZTableau = HalfTableau([target], conjugates=[Pauli('Z')])

        super().__init__(XTableau, ZTableau, name=f'S({target})')

class SWAP(Tableau):
    """
    Implements the Swap (SWAP) Tableau
    """

    def __init__(self, control:int, target:int,)->None:

        XTableau = HalfTableau([control, target], conjugates=[Pauli('XI'),Pauli('X')])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli('ZI'),Pauli('Z')])

        super().__init__(XTableau, ZTableau, name=f'SWAP({control}<->{target})')

class X(Tableau):
    """
    Implements the bit-filp (X) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], conjugates=[Pauli('X')])
        ZTableau = HalfTableau([target], conjugates=[Pauli('-Z')])

        super().__init__(XTableau, ZTableau, name=f'X({target})')

class Z(Tableau):
    """
    Implements the phase-flip (X) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], conjugates=[Pauli('-X')])
        ZTableau = HalfTableau([target], conjugates=[Pauli('Z')])

        super().__init__(XTableau, ZTableau, name=f'Z({target})')

class Y(Tableau):
    """
    Implements the phase and bit-flip (Y) Tableau
    """

    def __init__(self, target:int,)->None:

        XTableau = HalfTableau([target], conjugates=[Pauli('-X')])
        ZTableau = HalfTableau([target], conjugates=[Pauli('-Z')])

        super().__init__(XTableau, ZTableau, name=f'Y({target})')


class CZ(Tableau):
    """
    Implement the Controlled-Z (CZ) Tableau using its conjugation properties.
    """
    def __init__(self, control: int, target: int) -> None:

        XTableau = HalfTableau([control, target], conjugates=[Pauli('X'), Pauli('XI')])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli('XZ'), Pauli('ZX')])
        
        super().__init__(XTableau, ZTableau, name=f'CZ({control},{target})')


class RY(Tableau):
    def __init__(self, target: int, angle: float = math.pi/2) -> None:
        # Compute the integer k so that angle should equal k*(pi/2)
        k = round(angle / (math.pi / 2))
        # Check that the angle is close enough to a multiple of pi/2
        if not math.isclose(angle, k * (math.pi / 2), rel_tol=1e-9):
            raise ValueError("RY gate angle must be a multiple of pi/2 for a Clifford operation.")
        k_mod = k % 4  # Only 4 distinct rotations

        mapping_X = {0: 'X', 1: 'Z', 2: '-X', 3: '-Z'}
        mapping_Z = {0: 'Z', 1: '-X', 2: '-Z', 3: 'X'}

        XTableau = HalfTableau([target], conjugates=[Pauli(mapping_X[k_mod])])
        ZTableau = HalfTableau([target], conjugates=[Pauli(mapping_Z[k_mod])])

        super().__init__(XTableau, ZTableau, name=f'RY({angle})({target})')
