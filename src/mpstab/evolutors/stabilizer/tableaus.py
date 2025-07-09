import math
from typing import List

from mpstab.evolutors.stabilizer.pauli_string import Pauli

# GENERAL STRUCTURE


class HalfTableau:
    """
    Stores the rules to update either the X or Z components of a Pauli in the XZ encoding
    """

    def __init__(self, qubits: List[int], conjugates: List[Pauli]) -> None:

        assert len(qubits) == len(
            conjugates
        ), "Number of qubits must match the number of conjugates."

        self.qubits = qubits  # Qubits over which we act on with the Tableau
        self.conjugates = (
            conjugates  # Updated strings after the application of the operation
        )

    def __repr__(self):

        conj_repr = []
        for q, conj in zip(self.qubits, self.conjugates):
            conj_repr.append(f"{q} -> {conj}")
        return "\n".join(conj_repr)


class Tableau:
    """
    Full Tableau, stores the rules to update both the X or Z components of a Pauli in the XZ encoding.
    """

    def __init__(
        self, XTableau: HalfTableau, ZTableau: HalfTableau, name: str | None = None
    ) -> None:

        assert (
            XTableau.qubits == ZTableau.qubits
        ), "X and Z tableaus must share the same qubits"
        self.qubits = XTableau.qubits
        self.XTableau = XTableau
        self.ZTableau = ZTableau
        self.name = name

    def __repr__(self):
        return f"Z Tableau:\n{self.ZTableau}\nX Tableau\n{self.XTableau}"


# IMPLEMENTATIONS


class CNOT(Tableau):
    """
    Implements the Controlled-NOT (CNOT) Tableau
    """

    def __init__(
        self,
        control: int,
        target: int,
    ) -> None:

        XTableau = HalfTableau([control, target], conjugates=[Pauli("XX"), Pauli("IX")])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli("ZI"), Pauli("ZZ")])

        super().__init__(XTableau, ZTableau, name=f"CNOT({control}->{target})")


class SWAP(Tableau):
    """
    Implements the Swap (SWAP) Tableau
    """

    def __init__(
        self,
        control: int,
        target: int,
    ) -> None:

        XTableau = HalfTableau([control, target], conjugates=[Pauli("IX"), Pauli("XI")])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli("IZ"), Pauli("ZI")])

        super().__init__(XTableau, ZTableau, name=f"SWAP({control}<->{target})")


class CZ(Tableau):
    """
    Implements the Swap (SWAP) Tableau
    """

    def __init__(
        self,
        control: int,
        target: int,
    ) -> None:

        XTableau = HalfTableau([control, target], conjugates=[Pauli("XZ"), Pauli("ZX")])
        ZTableau = HalfTableau([control, target], conjugates=[Pauli("ZI"), Pauli("IZ")])

        super().__init__(XTableau, ZTableau, name=f"SWAP({control}<->{target})")


class H(Tableau):
    """
    Implements the Hadamard (H) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("Z")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("X")])

        super().__init__(XTableau, ZTableau, name=f"H({target})")


class S(Tableau):
    """
    Implements the Phase-gate (S) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("Y")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])

        super().__init__(XTableau, ZTableau, name=f"S({target})")


class Sdg(Tableau):
    """
    Implements the Phase-gate (S) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("-Y")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])

        super().__init__(XTableau, ZTableau, name=f"S({target})")


class X(Tableau):
    """
    Implements the bit-filp (X) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("X")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("-Z")])

        super().__init__(XTableau, ZTableau, name=f"X({target})")


class Z(Tableau):
    """
    Implements the phase-flip (X) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("-X")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])

        super().__init__(XTableau, ZTableau, name=f"Z({target})")


class Y(Tableau):
    """
    Implements the phase and bit-flip (Y) Tableau
    """

    def __init__(
        self,
        target: int,
    ) -> None:

        XTableau = HalfTableau([target], conjugates=[Pauli("-X")])
        ZTableau = HalfTableau([target], conjugates=[Pauli("-Z")])

        super().__init__(XTableau, ZTableau, name=f"Y({target})")


class RZ(Tableau):
    """
    Implements the single-qubit RZ rotation as a Tableau.

    The angle must be an integer multiple of π/2 for the operation to be Clifford.
    Acceptable angles include negative multiples.

    The transformation rules on the target qubit are:
      - RZ(0):       Identity (X → X, Z → Z)
      - RZ(π/2):     X → Y,  Z → Z.
      - RZ(π) or RZ(-π):   X → -X, Z → Z. (Equivalent to the Z gate)
      - RZ(-π/2):    X → -Y, Z → Z.
    """

    def __init__(self, target: int, angle: float) -> None:
        tol = 1e-8
        # Check that the angle is a multiple of π/2.
        factor = angle / (math.pi / 2)
        if abs(factor - round(factor)) > tol:
            raise ValueError(
                "Angle must be a multiple of π/2 for a Clifford operation."
            )

        # Normalize the rotation: k ∈ {0,1,2,3} corresponding to 0, π/2, π, 3π/2.
        k = int(round(factor)) % 4

        if k == 0:
            # Identity: X -> X, Z -> Z.
            XTableau = HalfTableau([target], conjugates=[Pauli("X")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            name = f"RZ({angle}) Identity"
        elif k == 1:
            # RZ(π/2): X -> Y, Z -> Z.
            XTableau = HalfTableau([target], conjugates=[Pauli("Y")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            name = f"RZ({angle})"
        elif k == 2:
            # RZ(π) (or -π): X -> -X, Z -> Z.
            XTableau = HalfTableau([target], conjugates=[Pauli("-X")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            name = f"RZ({angle})"
        elif k == 3:
            # RZ(3π/2) which is equivalent to RZ(-π/2): X -> -Y, Z -> Z.
            XTableau = HalfTableau([target], conjugates=[Pauli("-Y")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            name = f"RZ({angle})"

        super().__init__(XTableau, ZTableau, name=name)


class RY(Tableau):
    """
    Implements the single-qubit RY rotation as a Tableau.

    The angle must be an integer multiple of π/2 for the operation to be Clifford.
    Acceptable angles include negative multiples.

    The transformation rules on the target qubit are:
      - RY(0):        Identity (X → X,  Z → Z)
      - RY(π/2):      X → Z,  Z → -X
      - RY(π) or -π:  X → -X, Z → -Z        (Equivalent to combined X and Z flips)
      - RY(3π/2):     X → -Z, Z → X         (Equivalent to RY(-π/2))
    """

    def __init__(self, target: int, angle: float) -> None:
        tol = 1e-8
        factor = angle / (math.pi / 2)
        if abs(factor - round(factor)) > tol:
            raise ValueError(
                "Angle must be a multiple of π/2 for a Clifford operation."
            )

        k = int(round(factor)) % 4

        if k == 0:
            XTableau = HalfTableau([target], conjugates=[Pauli("X")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            name = f"RY({angle}) Identity"
        elif k == 1:
            XTableau = HalfTableau([target], conjugates=[Pauli("-Z")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("X")])
            name = f"RY({angle})"
        elif k == 2:
            XTableau = HalfTableau([target], conjugates=[Pauli("-X")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("-Z")])
            name = f"RY({angle})"
        else:  # k == 3
            XTableau = HalfTableau([target], conjugates=[Pauli("Z")])
            ZTableau = HalfTableau([target], conjugates=[Pauli("-X")])
            name = f"RY({angle})"

        super().__init__(XTableau, ZTableau, name=name)


class GPI2(Tableau):
    """
    π/2 rotation around the XY‐axis at angle phi, restricted to Clifford angles.
    """

    def __init__(self, target: int, angle: float) -> None:
        tol = 1e-8
        # Number of quarter‐turns in phi
        k = int(round(angle / (math.pi / 2))) % 4
        if abs((angle / (math.pi / 2)) - k) > tol:
            raise ValueError("phi must be a multiple of π/2 for a Clifford GPI2.")

        if k == 0:
            x_conj, z_conj = Pauli("X"), Pauli("-Y")
        elif k == 1:
            x_conj, z_conj = Pauli("-Z"), Pauli("X")
        elif k == 2:
            x_conj, z_conj = Pauli("X"), Pauli("Y")
        else:  # k == 3
            x_conj, z_conj = Pauli("Z"), Pauli("-X")

        XTableau = HalfTableau([target], conjugates=[x_conj])
        ZTableau = HalfTableau([target], conjugates=[z_conj])
        super().__init__(XTableau, ZTableau, name=f"GPI2({target}, φ={angle})")


class RX(Tableau):
    """
    Implements the single-qubit RX rotation as a Tableau.

    The angle must be an integer multiple of π/2 for the operation to be Clifford.
    Acceptable angles include negative multiples.

    The conjugation rules on the target qubit are:
      - RX(0):         X → X,   Z → Z
      - RX(π/2):       X → X,   Z → Y
      - RX(π) or -π:   X → X,   Z → -Z   (equivalent to the Z gate up to phase)
      - RX(3π/2):      X → X,   Z → -Y
    """

    def __init__(self, target: int, angle: float) -> None:
        tol = 1e-8
        # Check that the angle is a multiple of π/2
        factor = angle / (math.pi / 2)
        if abs(factor - round(factor)) > tol:
            raise ValueError(
                "Angle must be a multiple of π/2 for a Clifford operation."
            )
        # Normalize to k = 0,1,2,3 corresponding to 0, π/2, π, 3π/2
        k = int(round(factor)) % 4

        # X always maps to X under any RX(k·π/2)
        x_conj = Pauli("X")
        # Z maps according to k
        if k == 0:
            z_conj = Pauli("Z")
            name = f"RX({angle}) Identity"
        elif k == 1:
            z_conj = Pauli("Y")
            name = f"RX({angle})"
        elif k == 2:
            z_conj = Pauli("-Z")
            name = f"RX({angle})"
        else:  # k == 3
            z_conj = Pauli("-Y")
            name = f"RX({angle})"

        XTableau = HalfTableau([target], conjugates=[x_conj])
        ZTableau = HalfTableau([target], conjugates=[z_conj])
        super().__init__(XTableau, ZTableau, name=name)
