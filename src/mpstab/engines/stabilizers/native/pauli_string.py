"""
Pauli strings in the XZ (symplectic) encoding.

Each single-qubit Pauli is two bits -- one for X, one for Z -- so an ``n``-qubit
string is a single ``2n``-bit Python integer, and multiplication up to phase is
one XOR. The phase is tracked separately as a power of ``i`` in ``{0, 1, 2, 3}``,
since ``X Z = -i Y`` means the encoding alone does not determine it.

This is the representation :class:`~mpstab.engines.stabilizers.NativeStabilizersEngine`
propagates observables in.
"""

from copy import copy

single_pauli_to_xz = {"I": 0, "X": 1, "Z": 2, "Y": 3}
xz_to_single_pauli = {0: "I", 1: "X", 2: "Z", 3: "Y"}

phase_to_xz = {"i": 1, "-": 2, "-i": 3}
xz_to_phase = {0: "", 1: "i", 2: "-", 3: "-i"}
string_to_complex = {"": 1.0, "i": 1.0j, "-": -1.0, "-i": -1.0j}


def string_to_xz(description: str) -> int:
    """Encode an explicit ``IXYZ`` string into its XZ integer."""
    return sum(
        (single_pauli_to_xz[p] << (2 * q) for q, p in enumerate(description)), start=0
    )


def xz_to_string(xz_desc: int, n: int) -> str:
    """Decode an ``n``-qubit XZ integer back into an explicit ``IXYZ`` string."""
    return "".join([xz_to_single_pauli[3 & xz_desc >> (2 * q)] for q in range(n)])


def xz_to_string_phase(xz_desc: int, phase: int, n: int) -> str:
    """
    The phase prefix of the explicit string, as one of ``""``, ``"i"``, ``"-"`` or
    ``"-i"``.

    Undoes the ``-i`` per ``Y`` that :func:`initial_phase` folded in, so the
    result matches the ``IXYZ`` string rather than the encoding.
    """
    shift = 0
    for q in range(n):
        shift += (
            phase_to_xz["-i"]
            if ((xz_desc >> 2 * q) & (xz_desc >> (2 * q + 1))) & 1
            else 0
        )
    phase = update_phase(phase, shift)
    return xz_to_phase[phase]


def xz_prod(xz_desc1: int, xz_desc2: int) -> int:
    """Multiply two Pauli strings up to phase: a bitwise XOR of their encodings."""
    return xz_desc1 ^ xz_desc2


def has_X(xz_desc: int, qubit: int) -> bool:
    """Whether the Pauli at ``qubit`` has an X component, i.e. is ``X`` or ``Y``."""
    return bool(1 << (2 * qubit) & xz_desc)


def has_Z(xz_desc: int, qubit: int) -> bool:
    """Whether the Pauli at ``qubit`` has a Z component, i.e. is ``Z`` or ``Y``."""
    return bool(2 << (2 * qubit) & xz_desc)


def reset_qubit(xz_desc: int, qubit: int) -> int:
    """The same string with ``qubit`` set to the identity."""
    return xz_desc & (~(3 << (2 * qubit)))


def ith_qubit(xz_desc: int, qubit: int) -> int:
    """The encoding of the single Pauli at ``qubit``."""
    return 3 & (xz_desc >> (2 * qubit))


def replace_qubit(xz_desc: int, qubit: int, replacement: int) -> int:
    """The same string with ``qubit`` set to ``replacement``."""
    return reset_qubit(xz_desc, qubit) | (replacement << 2 * qubit)


def num_qubits(xz_desc: int) -> int:
    """
    How many qubits an encoding covers, ignoring trailing identities.

    So ``XIZXYII`` reports 5, not 7.
    """
    return int((len(bin(xz_desc)) - 1) // 2)


def phase_filp(xz_desc1: int, xz_desc2: int, n: int) -> bool:
    """Whether multiplying two Pauli strings flips the overall sign."""
    det = (xz_desc1 >> 1) & xz_desc2
    flip = False
    for q in range(n):
        flip = not flip if ((det >> (2 * q)) & 1) else flip
    return flip


def initial_phase(xz_desc: int, n: int, phase0: int = 0) -> int:
    """
    The phase the encoding implies, on top of ``phase0``.

    ``Y`` is stored as the pair ``XZ``, and ``X Z = -i Y``, so each ``Y`` in the
    string contributes a factor of ``i``.
    """
    shift = 0
    for q in range(n):
        shift += (
            phase_to_xz["i"]
            if ((xz_desc >> 2 * q) & (xz_desc >> (2 * q + 1))) & 1
            else 0
        )
    return update_phase(phase0, shift)


def update_phase(phase, shift):
    return (phase + shift) & 3


# HUMAN FRIENDLY INTERFACE


class Pauli:
    """An XZ-encoded Pauli string with a global phase in ``{1, -1, i, -i}``."""

    def __init__(self, description: str | int, n: int | None = None) -> None:
        """
        Args:
            description: an ``IXYZ`` string, optionally prefixed by ``i``, ``-`` or
                ``-i``; or an XZ encoding as an integer.
            n: number of qubits, only used with an integer ``description``, where
                trailing identities are otherwise unrecoverable.
        """
        if type(description) is int:
            self.xz = description
            self.n = n if n is not None else num_qubits(description)
            self.phase = initial_phase(self.xz, self.n)
            return

        phase0 = 0
        if description[0] in phase_to_xz.keys():
            phase0 += phase_to_xz[description[0]]
            description = description[1:]
            if description[0] in phase_to_xz.keys():
                phase0 += phase_to_xz[description[0]]
                description = description[1:]

        self.xz = string_to_xz(description)
        self.n = len(description)
        self.phase = initial_phase(self.xz, self.n, phase0)

    def __repr__(self) -> str:
        return self.to_string(ignore_phase=False)

    def to_string(self, ignore_phase=False):
        """The explicit ``IXYZ`` string, with its phase prefix unless suppressed."""
        string = xz_to_string(self.xz, self.n)
        if not ignore_phase:
            string = xz_to_string_phase(self.xz, self.phase, self.n) + string
        return string

    def complex_phase(self):
        """The phase as a complex number, one of ``1``, ``-1``, ``1j`` or ``-1j``."""
        return string_to_complex[xz_to_string_phase(self.xz, self.phase, self.n)]

    def __matmul__(self, other: "Pauli") -> "Pauli":
        """The product with ``other``, as a new instance; never in place."""
        result = copy(self)
        result.xz = xz_prod(self.xz, other.xz)
        result._update_phase(other.phase)

        if phase_filp(self.xz, other.xz, self.n):
            result._update_phase(phase_to_xz["-"])
        return result

    def __getitem__(self, qubit: int) -> int:
        return ith_qubit(self.xz, qubit)

    def __setitem__(self, qubit: int, pauli: int) -> None:
        self.xz = replace_qubit(self.xz, qubit, pauli)

    def _has_X(self, qubit: int) -> bool:
        return has_X(self.xz, qubit)

    def _has_Z(self, qubit: int) -> bool:
        return has_Z(self.xz, qubit)

    def _update_phase(self, new_phase: int) -> None:
        self.phase = update_phase(self.phase, new_phase)

    def apply(self, tableau) -> None:
        """
        Conjugate this string in place by a
        :class:`~mpstab.engines.stabilizers.native.tableaus.Tableau`.

        The tableau gives the image of X and Z on each qubit it touches, and
        conjugation is multiplicative, so the images of the components present on
        those qubits are multiplied together and substituted back in.
        """
        # Accumulate the images over the qubits the tableau touches.
        image = Pauli(0, n=len(tableau.qubits))
        for i, qubit in enumerate(tableau.qubits):
            if self._has_X(qubit):
                image = image @ tableau.XTableau.conjugates[i]
            if self._has_Z(qubit):
                image = image @ tableau.ZTableau.conjugates[i]

        for i, qubit in enumerate(tableau.qubits):
            self[qubit] = image[i]
        self._update_phase(image.phase)
