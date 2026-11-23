"""Deprecated alias of :mod:`mpstab.models.noise.mc_noise_estimator`, kept for old imports."""

import warnings

from mpstab.models.noise.mc_noise_estimator import *  # noqa: F401,F403

warnings.warn(
    "mpstab.noise.mc_noise_estimator moved to mpstab.models.noise.mc_noise_estimator; the old path will be removed.",
    DeprecationWarning,
    stacklevel=2,
)
