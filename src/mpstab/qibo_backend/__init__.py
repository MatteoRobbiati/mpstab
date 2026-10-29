"""mpstab as a qibo backend, so qibo circuits can be run through the surrogate."""

from qibo.config import raise_error

from mpstab.engines import QuimbEngine, StimEngine


class MetaBackend:
    """Backend loader qibo calls to obtain an mpstab backend."""

    @staticmethod
    def load(backend_name: str = "mpstab", **kwargs):
        """
        Build an :class:`~mpstab.qibo_backend.mpstab.MPStabBackend`.

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

        from mpstab.qibo_backend.mpstab import MPStabBackend

        return MPStabBackend(
            stab_engine=kwargs.get("stab_engine", StimEngine()),
            tn_engine=kwargs.get("tn_engine", QuimbEngine()),
        )

    def list_available(self) -> dict:
        """The backends this loader can provide."""
        return {"mpstab": True}
