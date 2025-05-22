import random as rnd
from typing import Generator

from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer.tableaus import CNOT, SWAP, H, S, Tableau, X, Z


def random_pauli(n: int) -> Pauli:
    """
    Sample a random Pauli string.
    """
    return Pauli(rnd.randint(0, (1 << (2 * n)) - 1), n)


def commute(p1: Pauli, p2: Pauli) -> bool:
    """
    Check wether two pauli strings commute.
    """
    return (p1 @ p2).phase == (p2 @ p1).phase


def sample_anticommuting(n: int) -> tuple[Pauli]:
    """
    Randomly sample two anticommuting Paulis.
    """
    p1 = random_pauli(n)
    # check if p1 = I, in this case it will commute with every other Pauli, hence needs to be discarded.
    while not p1.xz:
        p1 = random_pauli(n)

    p2 = random_pauli(n)
    while commute(p1, p2):
        p2 = random_pauli(n)

    return p1, p2


def clear_ZY(
    to_be_cleared: Pauli, to_be_updated: Pauli, shift: int = 0
) -> Generator[Tableau, None, None]:
    """
    Make the first Pauli string into one composed of only X, I by means of single qubit cliffords.
    At the same time updated the other string with the same operations.
    """
    n = to_be_cleared.n
    for i in range(n):
        if to_be_cleared._has_Z(i):
            if to_be_cleared._has_X(i):
                # If the qubit is in Y, turn it into X by a phase gate
                op = S(i)
                op_shift = S(i + shift)
            else:
                # Else it must be in Z, so turn it into an X by a Hadamard
                op = H(i)
                op_shift = H(i + shift)
            to_be_cleared.apply(op)
            to_be_updated.apply(op)
            yield op_shift


def first_X(p: Pauli) -> int:
    for i in range(p.n):
        if p._has_X(i):
            return i
    raise ValueError(f"The string {p} has no X.")


def reduce_Xs(
    to_be_reduced: Pauli, to_be_updated: Pauli, shift: int = 0
) -> Generator[Tableau, None, None]:
    """
    Reduces the number of Xs in a Pauli string containing only X,I to one.
    Produces circuits of depth ~O(n), could be improved to ~O(log n).
    """
    i0 = first_X(to_be_reduced)
    for i in range(i0 + 1, to_be_reduced.n):
        if to_be_reduced._has_X(i):
            op = CNOT(i, i0)
            to_be_reduced.apply(op)
            to_be_updated.apply(op)
            yield CNOT(i + shift, i0 + shift)

    if i0 != 0:
        op = SWAP(i0, 0)
        to_be_reduced.apply(op)
        to_be_updated.apply(op)
        yield SWAP(i0 + shift, shift)


def layer(
    tableau_x: Pauli, tableau_z: Pauli, shift: int = 0
) -> Generator[Tableau, None, None]:
    """
    Samples a layer of a random clifford circuit
    """

    yield from clear_ZY(tableau_x, tableau_z, shift)
    yield from reduce_Xs(tableau_x, tableau_z, shift)

    op = H(0)
    tableau_x.apply(op)
    tableau_z.apply(op)
    yield H(shift)

    yield from clear_ZY(tableau_z, tableau_x, shift)
    yield from reduce_Xs(tableau_z, tableau_x, shift)

    tableau_x.apply(op)
    tableau_z.apply(op)

    if tableau_x.phase != 0:
        yield Z(shift)
    if tableau_z.phase != 0:
        yield X(shift)


def random_clifford_circuit(n: int) -> Generator[Tableau, None, None]:
    """
    Samples a random clifford circuit of n qubits as a native gate sequence (Pauli, H, S and CNOT),
    and yields the gates as a python generator.
    """

    for i in range(n, 1, -1):
        tabx, tabz = sample_anticommuting(n)
        yield from layer(tabx, tabz, n - i)
