"""Deprecated alias of :mod:`mpstab.models.noise.model`, kept for old imports."""

import warnings

from mpstab.models.noise.model import *  # noqa: F401,F403

# TODO: name the version that removes this alias.
warnings.warn(
    "mpstab.noise.model moved to mpstab.models.noise.model; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
