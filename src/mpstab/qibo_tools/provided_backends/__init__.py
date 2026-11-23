"""Backends mpstab provides to qibo.

- :mod:`~mpstab.qibo_tools.provided_backends.mpstab_backend`:
  :class:`~mpstab.qibo_tools.provided_backends.mpstab_backend.MPStabBackend`,
  loaded by ``qibo.construct_backend("mpstab")`` through
  :class:`~mpstab.qibo_tools.provided_backends.mpstab_backend.MetaBackend`.
"""

from mpstab.qibo_tools.provided_backends.mpstab_backend import (
    MetaBackend,
    MPStabBackend,
)

__all__ = ["MPStabBackend", "MetaBackend"]
