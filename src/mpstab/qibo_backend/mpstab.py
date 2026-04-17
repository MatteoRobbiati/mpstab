from dataclasses import dataclass

from qibo.backends import NumpyBackend
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from mpstab import HSMPO
from mpstab.engines import (
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)


@dataclass
class MPStabBackend(NumpyBackend):
    """Qibo backend for hybrid stabilizer-MPO surrogate simulation."""

    def __init__(
        self,
        stab_engine: StabilizersEngine = StimEngine(),
        tn_engine: TensorNetworkEngine = QuimbEngine(),
    ):
        """Initialize backend with Stabilizer and Tensor-Network engines."""
        super(self.__class__, self).__init__()

        self.name = "mpstab"
        self.stab_engine = stab_engine
        self.tn_engine = tn_engine
        self.max_bond_dimension = None

    def exp_value_observable_symbolic(
        self, circuit, operators_list, sites_list, coeffs_list, nqubits
    ):
        r"""
        Compute expectation value of a symbolic Hamiltonian using HSMPO surrogate.

        Evaluate the expectation value of a symbolic Hamiltonian on a quantum
        circuit using the hybrid stabilizer-MPO (HSMPO) surrogate representation.
        This provides an efficient approximation by combining stabilizer
        simulation with matrix product state tensor networks.

        Args:
            circuit: Quantum circuit to evaluate (qibo.models.Circuit)
            operators_list: List of Pauli operator strings (e.g., ['xyz', 'xyz'])
            sites_list: Tuples of qubits each operator acts on
                (e.g., [(1,2,3), (1,2,3)])
            coeffs_list: Coefficients for each Hamiltonian term
            nqubits: Number of qubits in the circuit

        Returns:
            float: Real part of the expectation value of the Hamiltonian

        Note:
            Each operator string must act on distinct qubits within the same term.
            The HSMPO representation uses the configured stabilizer and
            tensor-network engines to efficiently compute the expectation value.
        """

        # Build symbolic Hamiltonian from component lists
        hamiltonian = 0
        pauli_map = {"X": X, "Y": Y, "Z": Z}

        for coeff, operators, sites in zip(coeffs_list, operators_list, sites_list):
            # Build the term for this Hamiltonian component
            term = 1
            for op, qubit in zip(operators, sites):
                term = term * pauli_map[op](qubit)
            hamiltonian = hamiltonian + coeff * term

        # Create SymbolicHamiltonian object
        symbolic_h = SymbolicHamiltonian(nqubits=nqubits, form=hamiltonian)

        # Defining Hybrid Stabilizer MPO with proper engines
        hybrid_surrogate = HSMPO(ansatz=circuit)
        hybrid_surrogate.set_engines(
            stab_engine=self.stab_engine,
            tn_engine=self.tn_engine,
        )

        # Computing expectation value from symbolic Hamiltonian
        return hybrid_surrogate.expectation(symbolic_h)
