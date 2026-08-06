"""
Per-gate conjugation rules, as tableaus.

A Clifford gate is fully described by where it sends X and Z on each qubit it
touches. Each class below records exactly that, as
:class:`~mpstab.engines.stabilizers.native.pauli_string.Pauli` images, and
:meth:`~mpstab.engines.stabilizers.native.pauli_string.Pauli.apply` composes them
onto an observable.

The rotation gates (``RX``, ``RY``, ``RZ``, ``GPI2``) are Clifford only at
multiples of pi/2, so they take an angle and select from the four quarter-turn
cases; anything in between raises.
"""

import math
from typing import List

from mpstab.engines.stabilizers.native.pauli_string import Pauli


class HalfTableau:
    """Where a gate sends either X or Z on each of the qubits it acts on."""

    def __init__(self, qubits: List[int], conjugates: List[Pauli]) -> None:
        if len(qubits) != len(conjugates):
            raise ValueError(
                f"{len(qubits)} qubits but {len(conjugates)} images; there must be "
                "one image per qubit."
            )
        self.qubits = qubits
        self.conjugates = conjugates

    def __repr__(self):
        return "\n".join(
            f"{qubit} -> {image}" for qubit, image in zip(self.qubits, self.conjugates)
        )


class Tableau:
    """A gate's full conjugation rule: where it sends both X and Z."""

    def __init__(
        self, XTableau: HalfTableau, ZTableau: HalfTableau, name: str | None = None
    ) -> None:
        if XTableau.qubits != ZTableau.qubits:
            raise ValueError(
                f"The X tableau acts on {XTableau.qubits} but the Z tableau on "
                f"{ZTableau.qubits}; they must share the same qubits."
            )
        self.qubits = XTableau.qubits
        self.XTableau = XTableau
        self.ZTableau = ZTableau
        self.name = name

    def __repr__(self):
        return f"Z Tableau:\n{self.ZTableau}\nX Tableau\n{self.XTableau}"


class _SingleQubitTableau(Tableau):
    """A one-qubit gate, given the images of X and Z."""

    def __init__(self, target: int, x_image: str, z_image: str, name: str) -> None:
        super().__init__(
            HalfTableau([target], conjugates=[Pauli(x_image)]),
            HalfTableau([target], conjugates=[Pauli(z_image)]),
            name=name,
        )


class _TwoQubitTableau(Tableau):
    """A two-qubit gate, given the two-qubit images of X and Z on each qubit."""

    def __init__(
        self,
        control: int,
        target: int,
        x_images: tuple,
        z_images: tuple,
        name: str,
    ) -> None:
        qubits = [control, target]
        super().__init__(
            HalfTableau(qubits, conjugates=[Pauli(p) for p in x_images]),
            HalfTableau(qubits, conjugates=[Pauli(p) for p in z_images]),
            name=name,
        )


class _QuarterTurnTableau(Tableau):
    """
    A one-qubit rotation restricted to Clifford angles.

    Subclasses set :attr:`IMAGES` to the ``(x_image, z_image)`` pair for each of
    the four quarter turns ``k = 0, 1, 2, 3``, meaning angles ``0``, ``pi/2``,
    ``pi`` and ``3 pi/2``.
    """

    #: One ``(x_image, z_image)`` pair per quarter turn.
    IMAGES: tuple = ()
    TOLERANCE = 1e-8

    def __init__(self, target: int, angle: float) -> None:
        turns = angle / (math.pi / 2)
        if abs(turns - round(turns)) > self.TOLERANCE:
            raise ValueError(
                f"{type(self).__name__}({angle}) is not Clifford; the angle must be "
                "a multiple of pi/2."
            )
        x_image, z_image = self.IMAGES[int(round(turns)) % 4]
        super().__init__(
            HalfTableau([target], conjugates=[Pauli(x_image)]),
            HalfTableau([target], conjugates=[Pauli(z_image)]),
            name=f"{type(self).__name__}({target}, angle={angle})",
        )


class CNOT(_TwoQubitTableau):
    """Controlled-NOT."""

    def __init__(self, control: int, target: int) -> None:
        super().__init__(
            control,
            target,
            x_images=("XX", "IX"),
            z_images=("ZI", "ZZ"),
            name=f"CNOT({control}->{target})",
        )


class CZ(_TwoQubitTableau):
    """Controlled-Z."""

    def __init__(self, control: int, target: int) -> None:
        super().__init__(
            control,
            target,
            x_images=("XZ", "ZX"),
            z_images=("ZI", "IZ"),
            name=f"CZ({control},{target})",
        )


class SWAP(_TwoQubitTableau):
    """Swap."""

    def __init__(self, control: int, target: int) -> None:
        super().__init__(
            control,
            target,
            x_images=("IX", "XI"),
            z_images=("IZ", "ZI"),
            name=f"SWAP({control}<->{target})",
        )


class H(_SingleQubitTableau):
    """Hadamard: X <-> Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "Z", "X", name=f"H({target})")


class S(_SingleQubitTableau):
    """Phase gate: X -> Y, Z -> Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "Y", "Z", name=f"S({target})")


class Sdg(_SingleQubitTableau):
    """Inverse phase gate: X -> -Y, Z -> Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "-Y", "Z", name=f"Sdg({target})")


class X(_SingleQubitTableau):
    """Bit flip: X -> X, Z -> -Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "X", "-Z", name=f"X({target})")


class Y(_SingleQubitTableau):
    """Bit and phase flip: X -> -X, Z -> -Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "-X", "-Z", name=f"Y({target})")


class Z(_SingleQubitTableau):
    """Phase flip: X -> -X, Z -> Z."""

    def __init__(self, target: int) -> None:
        super().__init__(target, "-X", "Z", name=f"Z({target})")


class RX(_QuarterTurnTableau):
    """Rotation about X. Leaves X alone; sends Z to -Y, -Z, Y."""

    IMAGES = (("X", "Z"), ("X", "-Y"), ("X", "-Z"), ("X", "Y"))


class RY(_QuarterTurnTableau):
    """Rotation about Y. Leaves Y alone; exchanges X and Z with signs."""

    IMAGES = (("X", "Z"), ("-Z", "X"), ("-X", "-Z"), ("Z", "-X"))


class RZ(_QuarterTurnTableau):
    """Rotation about Z. Leaves Z alone; sends X to Y, -X, -Y."""

    IMAGES = (("X", "Z"), ("Y", "Z"), ("-X", "Z"), ("-Y", "Z"))


class GPI2(_QuarterTurnTableau):
    """
    A pi/2 rotation about the axis ``(cos phi, sin phi, 0)``, at Clifford ``phi``.

    Unlike the others this is not the identity at ``phi = 0``: the rotation angle
    is always pi/2 and ``phi`` only picks the axis.
    """

    IMAGES = (("X", "-Y"), ("-Z", "X"), ("X", "Y"), ("Z", "-X"))
