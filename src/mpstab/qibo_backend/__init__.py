"""Deprecated alias of :mod:`mpstab.qibo_tools.provided_backends`, kept for old imports."""

import warnings

from mpstab.qibo_tools.provided_backends import *  # noqa: F401,F403

# TODO: name the version that removes this alias.
warnings.warn(
    "mpstab.qibo_backend moved to mpstab.qibo_tools.provided_backends; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
