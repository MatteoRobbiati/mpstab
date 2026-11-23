"""Deprecated alias of :mod:`mpstab.models.noise.compile`, kept for old imports."""

import warnings

from mpstab.models.noise.compile import *  # noqa: F401,F403

warnings.warn(
    "mpstab.noise.compile moved to mpstab.models.noise.compile; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
