from qibo.config import raise_error

from mpstab.engines import QuimbEngine, StimEngine


class MetaBackend:
    """
    Simplified Meta-backend loader for mpstab.
    Focuses on initializing the hybrid engines directly.
    """

    @staticmethod
    def load(backend_name: str = "mpstab", **kwargs):
        """
        Loads the mpstab backend.

        Args:
            backend_name (str): Must be "mpstab".
            tn_engine (TensorNetworkEngine, optional): Engine for TN operations.
            stab_engine (StabilizersEngine, optional): Engine for Stabilizer operations.
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
