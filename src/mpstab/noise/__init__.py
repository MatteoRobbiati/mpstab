"""Deprecated alias of :mod:`mpstab.models.noise`, kept for old imports."""

import warnings

from mpstab.models.noise import *  # noqa: F401,F403

warnings.warn(
    "mpstab.noise moved to mpstab.models.noise; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
