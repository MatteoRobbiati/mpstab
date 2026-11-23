"""Deprecated alias of :mod:`mpstab.models.noise.events`, kept for old imports."""

import warnings

from mpstab.models.noise.events import *  # noqa: F401,F403

warnings.warn(
    "mpstab.noise.events moved to mpstab.models.noise.events; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
