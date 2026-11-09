"""The qibo backend that evaluates expectation values through an HSMPO."""

from dataclasses import dataclass

from qibo.backends import NumpyBackend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from mpstab.engines import (
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)
from mpstab.evolutors.hsmpo import HSMPO

_SYMBOLS = {"X": X, "Y": Y, "Z": Z}


@dataclass
class MPStabBackend(NumpyBackend):
    """Qibo backend backed by the hybrid stabilizer-MPO surrogate."""

    def __init__(
        self,
        stab_engine: StabilizersEngine = None,
        tn_engine: TensorNetworkEngine = None,
    ):
        """
        Args:
            stab_engine: stabilizers engine, ``StimEngine`` by default.
            tn_engine: tensor-network engine, ``QuimbEngine`` by default.
        """
        super().__init__()

        self.name = "mpstab"
        self.stab_engine = stab_engine or StimEngine()
        self.tn_engine = tn_engine or QuimbEngine()
        self.max_bond_dimension = None

    def exp_value_observable_symbolic(
        self, circuit, operators_list, sites_list, coeffs_list, nqubits
    ):
        """
        Expectation value of a symbolic Hamiltonian on ``circuit``, via an HSMPO.

        Args:
            circuit: the qibo circuit to evaluate.
            operators_list: Pauli labels per term, e.g. ``["XYZ", "XYZ"]``.
            sites_list: the qubits each term's operators act on, e.g.
                ``[(1, 2, 3), (1, 2, 3)]``. Within a term they must be distinct.
            coeffs_list: one coefficient per term.
            nqubits: circuit width.

        Returns:
            The expectation value of the Hamiltonian.
        """
        form = 0
        for coefficient, operators, sites in zip(
            coeffs_list, operators_list, sites_list
        ):
            term = 1
            for label, qubit in zip(operators, sites):
                term = term * _SYMBOLS[label](qubit)
            form = form + coefficient * term

        surrogate = HSMPO(ansatz=circuit)
        surrogate.set_engines(stab_engine=self.stab_engine, tn_engine=self.tn_engine)
        return surrogate.expectation(SymbolicHamiltonian(nqubits=nqubits, form=form))
