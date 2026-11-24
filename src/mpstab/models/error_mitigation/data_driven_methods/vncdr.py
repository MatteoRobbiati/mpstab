"""Variable-noise Clifford data regression (vnCDR).

Fits the noiseless value as a combination of the noisy values at several noise
levels, on low-magic training circuits whose noiseless values HSMPO computes.

Reference:
    A. Lowe, M. H. Gordon et al., *Unified approach to data-driven quantum
    error mitigation*, `arXiv:2011.01157 <https://arxiv.org/abs/2011.01157>`_.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from qibo import Circuit

from mpstab.hamiltonians import Observable
from mpstab.models.ansatze import Ansatz
from mpstab.models.error_mitigation.data_driven_methods.regression import (
    MitigationNoiseMap,
    NoiseSource,
    fit_noise_map,
    training_data,
)


def vncdr(
    ansatz: Ansatz | Circuit,
    observable: Observable,
    noise: NoiseSource,
    noise_levels: Sequence[float] = (1, 3, 5),
    amplification: str = "scale",
    n_training_circuits: int = 50,
    replacement_probability: float = 0.9,
    replacement_method: str = "closest",
    model: Callable | None = None,
    initial_state: Circuit | None = None,
    max_bond_dimension: int | None = None,
    n_samples: int = 1000,
    seed: int | None = None,
    n_workers: int = 1,
) -> MitigationNoiseMap:
    """
    Learn the variable-noise CDR noise map of ``observable`` on ``ansatz``.

    The map takes the target's noisy values at every level of
    ``noise_levels``, so evaluate the target the same way. For a noise model,
    :func:`~mpstab.models.error_mitigation.noisy_value` does it::

        levels = (1, 3, 5)
        noise_map = vncdr(ansatz, "ZZII", noise, noise_levels=levels, seed=0)
        target = HSMPO(ansatz)
        mitigated = noise_map(
            [noisy_value(target, "ZZII", noise, level) for level in levels]
        )

    Args:
        ansatz: the circuit to mitigate.
        observable: a Pauli string or any other format
            :func:`~mpstab.hamiltonians.pauli_terms` accepts.
        noise: a :class:`~mpstab.models.noise.NoiseModel`, simulated with
            HSMPO, or a callable ``f(circuit, level) -> float`` giving the noisy
            value of a circuit at a noise level, for instance by running
            :func:`~mpstab.models.error_mitigation.fold_circuit` of it on
            hardware. The circuit it receives starts with ``initial_state``, if
            any.
        noise_levels: the noise amplification factors, ``1`` being the
            unamplified noise.
        amplification: how a noise model is amplified: ``"scale"`` multiplies
            every error's strength by the level
            (:meth:`~mpstab.models.noise.NoiseModel.scaled`), ``"fold"`` runs
            the folded circuit
            (:func:`~mpstab.models.error_mitigation.fold_circuit`), which needs
            odd integer levels. Ignored for a callable.
        n_training_circuits: training circuits to sample.
        replacement_probability: chance of replacing each non-Clifford gate in
            a training circuit
            (:func:`~mpstab.models.error_mitigation.training_circuits`).
        replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
        model: ``model(x, *params)`` to fit, with ``x`` an array whose last
            axis runs over the noise levels. The default ``x @ params``, a
            linear combination of the levels as in [2], is fitted by least
            squares; any other with ``scipy.optimize.curve_fit``, starting
            from ones.
        initial_state: one-qubit-gates-only circuit preparing the initial
            state, ``|0...0>`` if ``None``.
        max_bond_dimension: MPS truncation cap for HSMPO.
        n_samples: fault configurations HSMPO samples per noisy value, when
            ``noise`` is a noise model.
        seed: seed of the training circuits and of the noise sampling.
        n_workers: processes evaluating the training circuits in parallel,
            when ``noise`` is a noise model; a callable always runs in this
            process. The result does not depend on it.

    Returns:
        :class:`~mpstab.models.error_mitigation.MitigationNoiseMap`, with
        :attr:`~mpstab.models.error_mitigation.MitigationNoiseMap.noise_levels`
        set.
    """
    noise_levels = tuple(noise_levels)
    if amplification not in ("scale", "fold"):
        raise ValueError(
            f"Unknown amplification {amplification!r}, expected 'scale' or 'fold'."
        )
    circuits, exact, noisy, stderr = training_data(
        ansatz,
        observable,
        noise,
        noise_levels,
        amplification,
        n_training_circuits,
        replacement_probability,
        replacement_method,
        initial_state,
        max_bond_dimension,
        n_samples,
        seed,
        n_workers,
    )
    model, params = fit_noise_map(model, noisy, exact, _combination, noisy)
    return MitigationNoiseMap(
        params,
        exact,
        noisy,
        model,
        noise_levels=noise_levels,
        noisy_stderr=stderr,
        circuits=circuits,
    )


def _combination(x, *params):
    """``x @ params``, the vnCDR model: a linear combination of the noise levels."""
    return x @ np.asarray(params)
