from qibo.config import raise_error

from mpstab.engines import QuimbEngine, StimEngine


class MetaBackend:
    """
    Backend loader for the MPSTAB hybrid stabilizer-MPO simulator.

    Provides simple initialization of the hybrid stabilizer-MPO representation
    as a Qibo backend. Configure with stabilizer and tensor-network engines
    to customize the simulation strategy.
    """

    @staticmethod
    def load(backend_name: str = "mpstab", **kwargs):
        """
        Load and initialize the MPSTAB backend.

        Create an MPSTAB backend instance configured with specified engines
        for stabilizer simulation and tensor-network contraction.

        Args:
            backend_name: Must be "mpstab" to load the MPSTAB backend
            stab_engine: Stabilizer engine (default: StimEngine)
            tn_engine: Tensor-network engine (default: QuimbEngine)

        Returns:
            MPStabBackend: Configured backend instance

        Raises:
            ValueError: If backend_name is not "mpstab"
        """
        if backend_name != "mpstab":
            raise_error(
                ValueError, f"Backend {backend_name} is not supported. Use 'mpstab'."
            )

        from mpstab.qibo_backend.mpstab import MPStabBackend

        return MPStabBackend(
            stab_engine=kwargs.get("stab_engine", StimEngine()),
            tn_engine=kwargs.get("tn_engine", QuimbEngine()),
        )

    def list_available(self) -> dict:
        """Lists available backends."""
        return {"mpstab": True}
