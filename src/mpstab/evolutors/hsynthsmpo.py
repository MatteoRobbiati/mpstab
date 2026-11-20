"""Deprecated alias of :mod:`mpstab.evolutors.hqsmpo`, kept for old imports."""

import warnings

from mpstab.evolutors.hqsmpo import *  # noqa: F401,F403
from mpstab.evolutors.hqsmpo import HQSMPO

warnings.warn(
    "mpstab.evolutors.hsynthsmpo was renamed mpstab.evolutors.hqsmpo, and "
    "HSynthSMPO HQSMPO; the old names will be removed.",
    DeprecationWarning,
    stacklevel=2,
)

HSynthSMPO = HQSMPO
