"""The qibo backend that evaluates expectation values through an HSMPO."""

from qibo.backends import NumpyBackend
from qibo.config import raise_error

from mpstab.engines import StabilizersEngine, TensorNetworkEngine
from mpstab.evolutors.hsmpo import HSMPO


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
        self.stab_engine = stab_engine
        self.tn_engine = tn_engine

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
        terms = {}
        for coefficient, operators, sites in zip(
            coeffs_list, operators_list, sites_list
        ):
            labels = ["I"] * nqubits
            for label, qubit in zip(operators, sites):
                labels[qubit] = label
            pauli = "".join(labels)
            terms[pauli] = terms.get(pauli, 0.0) + complex(coefficient).real

        surrogate = HSMPO(
            ansatz=circuit, stab_engine=self.stab_engine, tn_engine=self.tn_engine
        )
        return surrogate.expectation(terms)


class MetaBackend:
    """Backend loader qibo calls to obtain an mpstab backend."""

    @staticmethod
    def load(backend_name: str = "mpstab", **kwargs):
        """
        Build an :class:`MPStabBackend`.

        Args:
            backend_name: must be ``"mpstab"``.
            stab_engine: stabilizers engine, ``StimEngine`` by default.
            tn_engine: tensor-network engine, ``QuimbEngine`` by default.

        Raises:
            ValueError: if ``backend_name`` is anything else.
        """
        if backend_name != "mpstab":
            raise_error(
                ValueError, f"Backend {backend_name} is not supported. Use 'mpstab'."
            )

        return MPStabBackend(
            stab_engine=kwargs.get("stab_engine"), tn_engine=kwargs.get("tn_engine")
        )

    def list_available(self) -> dict:
        """The backends this loader can provide."""
        return {"mpstab": True}
