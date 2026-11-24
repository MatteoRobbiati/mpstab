"""Clifford data regression (CDR).

Fits ``exact = a * noisy + b`` on low-magic training circuits whose noiseless
values HSMPO computes, then applies it to the target's noisy value.

Reference:
    P. Czarnik, A. Arrasmith et al., *Error mitigation with Clifford
    quantum-circuit data*, `arXiv:2005.10189 <https://arxiv.org/abs/2005.10189>`_.
"""

from __future__ import annotations

from typing import Callable

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


def cdr(
    ansatz: Ansatz | Circuit,
    observable: Observable,
    noise: NoiseSource,
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
    Learn the CDR noise map of ``observable`` on ``ansatz``.

    Example::

        from mpstab import HSMPO, DepolarizingError, NoiseModel, cdr
        from mpstab.models.ansatze import HardwareEfficient

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        noise = NoiseModel().add(DepolarizingError(0.01))

        noise_map = cdr(ansatz, "ZZII", noise, n_training_circuits=20, seed=0)
        noisy = HSMPO(ansatz).noisy_expectation("ZZII", noise, seed=0).mean
        mitigated = noise_map(noisy)

    Args:
        ansatz: the circuit to mitigate.
        observable: a Pauli string or any other format
            :func:`~mpstab.hamiltonians.pauli_terms` accepts.
        noise: a :class:`~mpstab.models.noise.NoiseModel`, simulated with
            HSMPO, or a callable ``f(circuit) -> float`` giving the noisy value
            of a circuit, for instance by running it on hardware. The circuit
            it receives starts with ``initial_state``, if any.
        n_training_circuits: training circuits to sample.
        replacement_probability: chance of replacing each non-Clifford gate in
            a training circuit
            (:func:`~mpstab.models.error_mitigation.training_circuits`).
        replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
        model: ``model(x, *params)`` to fit, with ``x`` the noisy value. The
            default ``a * x + b`` is fitted by least squares; any other is
            fitted with ``scipy.optimize.curve_fit``, starting from ones.
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
        :class:`~mpstab.models.error_mitigation.MitigationNoiseMap`
    """
    circuits, exact, noisy, stderr = training_data(
        ansatz,
        observable,
        noise,
        None,
        "scale",
        n_training_circuits,
        replacement_probability,
        replacement_method,
        initial_state,
        max_bond_dimension,
        n_samples,
        seed,
        n_workers,
    )
    design = np.column_stack([noisy, np.ones_like(noisy)])
    model, params = fit_noise_map(model, noisy, exact, _linear, design)
    return MitigationNoiseMap(
        params, exact, noisy, model, noisy_stderr=stderr, circuits=circuits
    )


def _linear(x, a, b):
    """``a * x + b``, the CDR model."""
    return a * x + b
