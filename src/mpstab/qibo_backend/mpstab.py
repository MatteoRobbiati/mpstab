from dataclasses import dataclass

from qibo.backends import NumpyBackend
from qibo.config import raise_error
from qibo.models import Circuit

from mpstab import HSMPO
from mpstab.engines import (
    QuimbEngine,
    StabilizersEngine,
    StimEngine,
    TensorNetworkEngine,
)


@dataclass
class MPStabBackend(NumpyBackend):

    def __init__(
        self,
        stab_engine: StabilizersEngine = StimEngine(),
        tn_engine: TensorNetworkEngine = QuimbEngine(),
    ):
        super(self.__class__, self).__init__()

        self.name = "mpstab"
        self.stab_engine = stab_engine
        self.tn_engine = tn_engine
        self.max_bond_dimension = None

    def exp_value_observable_symbolic(
        self, circuit, operators_list, sites_list, coeffs_list, nqubits
    ):
        """
        Compute the expectation value of a symbolic Hamiltonian on a quantum circuit using tensor network contraction.
        This method takes a Qibo circuit, converts it to a Quimb tensor network circuit, and evaluates the expectation value
        of a Hamiltonian specified by three lists of strings: operators, sites, and coefficients.
        The expectation value is computed by summing the contributions from each term in the Hamiltonian, where each term's
        expectation is calculated using Quimb's `local_expectation` function.
        Each operator string must act on all different qubits, i.e., for each term, the corresponding sites tuple must contain unique qubit indices.
        Example: operators_list = ['xyz', 'xyz'], sites_list = [(1,2,3), (1,2,3)], coeffs_list = [1, 2]


        Parameters
        ----------
        circuit : qibo.models.Circuit
            The quantum circuit to evaluate, provided as a Qibo circuit object.
        operators_list : list of str
            List of operator strings representing the symbolic Hamiltonian terms.
        sites_list : list of tuple of int
            Tuples each specifying the qubits (sites) the corresponding operator acts on.
        coeffs_list : list of real/complex
            The coefficients for each Hamiltonian term.
        Returns
        -------
        float
            The real part of the expectation value of the Hamiltonian on the given circuit state.
        """

        # Defining Hybrid Stabilizer MPO with proper engines
        hybrid_surrogate = HSMPO(ansatz=circuit)
        hybrid_surrogate.set_engines(
            stab_engine=self.stab_engine,
            tn_engine=self.tn_engine,
        )
        # Computing expectation value from symbolic Hamiltonian
        return hybrid_surrogate._expectation_from_symbolic_hamiltonian(
            coefficients_list=coeffs_list,
            operators_list=operators_list,
            sites_list=sites_list,
            nqubits=nqubits,
        )
