from tncdr.evolutors.stabilizer.tableaus import CNOT, H, S
from tncdr.evolutors.stabilizer.pauli_string import Pauli

two_qubit_pauli = [
    "I",
    "X",
    "Y",
    "Z",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]
single_qubit_pauli = ["I", "X", "Y", "Z"]


def print_cayley_table(strings):
    paulis = [Pauli(s) for s in strings]
    print("\t| " + "\t".join(f"{p}" for p in paulis))
    print("-" * 36)
    for p_row in paulis:
        print(f"{p_row}\t| " + "\t".join([f"{p_row@p}" for p in paulis]))
    print("-" * 36)


def main():

    print("Pauli multiplication. Example: Cayley Table computation")
    print_cayley_table(single_qubit_pauli)

    ops = [CNOT(0, 1), S(2), H(3), CNOT(2, 0)]
    p = Pauli("YYIX")
    print("\nApply Clifford operations to update the Pauli instance.")
    print(f"Initial state:\t{p}")
    print("Circuit:\t" + ", ".join([o.name for o in ops]))

    for op in ops:
        p.apply(op)

    # check order of operations, is it consistent?
    print(f"Updated string:\t{p}")


if __name__ == "__main__":
    main()
