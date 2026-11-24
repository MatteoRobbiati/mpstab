"""The machinery shared by Clifford data regression and its variable-noise
extension: training data, noisy values and the fitted noise map.

Both methods learn a map from noisy expectation values to noiseless ones on
training circuits close to the target, then apply it to the target's noisy
value:

- the training circuits are low-magic variants of the target
  (:func:`~mpstab.models.error_mitigation.training_circuits`);
- their noiseless values come from HSMPO, cheap because they carry little
  magic;
- their noisy values come either from a :class:`~mpstab.models.noise.NoiseModel`,
  simulated by :meth:`~mpstab.HSMPO.noisy_expectation`, or from any callable,
  such as one running the circuit on hardware.

The methods themselves are :func:`~mpstab.models.error_mitigation.cdr` and
:func:`~mpstab.models.error_mitigation.vncdr`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Union

import numpy as np
from qibo import Circuit
from scipy.optimize import curve_fit

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import Observable
from mpstab.models.error_mitigation.data_driven_methods.training import (
    training_circuits,
)
from mpstab.models.error_mitigation.folding import fold_circuit
from mpstab.models.noise import NoiseModel

#: Noisy values of a circuit: a noise model to simulate, or a callable returning
#: the value, ``f(circuit)`` for CDR and ``f(circuit, level)`` for vnCDR.
NoiseSource = Union[NoiseModel, Callable[..., float]]


@dataclass(frozen=True)
class MitigationNoiseMap:
    """
    A learned map from noisy expectation values to noiseless ones.

    Call it on the target circuit's noisy value (CDR) or on its values at each
    of :attr:`noise_levels` (vnCDR) to get the mitigated value. Several values
    at once go along the leading axes.

    Attributes:
        params: the fitted parameters of :attr:`model`.
        exact: noiseless value of each training circuit, shape ``(n,)``.
        noisy: their noisy values, shape ``(n,)`` for CDR and
            ``(n, len(noise_levels))`` for vnCDR.
        noise_levels: the noise levels of vnCDR, ``None`` for CDR.
        noisy_stderr: standard error of each noisy value, same shape as
            :attr:`noisy`, when a noise model was sampled; ``None`` for a
            callable.
        circuits: the training circuits.
        model: ``model(x, *params)``, with ``x`` a noisy value (CDR) or an array
            whose last axis runs over the noise levels (vnCDR).
    """

    params: np.ndarray
    exact: np.ndarray
    noisy: np.ndarray
    model: Callable = field(repr=False)
    noise_levels: tuple[float, ...] | None = None
    noisy_stderr: np.ndarray | None = field(default=None, repr=False)
    circuits: list[Circuit] = field(default_factory=list, repr=False)

    def __call__(self, noisy):
        """The mitigated value(s) of ``noisy``."""
        noisy = np.asarray(noisy, dtype=float)
        if self.noise_levels is not None and noisy.shape[-1:] != (
            len(self.noise_levels),
        ):
            raise ValueError(
                f"Expected one value per noise level {self.noise_levels}, "
                f"got shape {noisy.shape}."
            )
        return self.model(noisy, *self.params)

    @property
    def residuals(self) -> np.ndarray:
        """Mitigated minus noiseless value of each training circuit."""
        return self(self.noisy) - self.exact


def noisy_value(
    hsmpo: HSMPO,
    observable: Observable,
    noise: NoiseModel,
    level: float = 1,
    amplification: str = "scale",
    n_samples: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Noisy expectation value of ``observable`` on the circuit of ``hsmpo``,
    with ``noise`` amplified to ``level``, as the training circuits get it.

    Args:
        hsmpo: the surrogate of the circuit.
        observable: a Pauli string or any other format
            :func:`~mpstab.hamiltonians.pauli_terms` accepts.
        noise: the noise model.
        level: the noise amplification factor.
        amplification: ``"scale"`` or ``"fold"``, as in
            :func:`~mpstab.models.error_mitigation.vncdr`.
        n_samples: fault configurations sampled by
            :meth:`~mpstab.HSMPO.noisy_expectation`.
        seed: seed of the sampler.
    """
    return float(
        _noisy_result(
            hsmpo, observable, noise, level, amplification, n_samples, seed
        ).mean
    )


def _noisy_result(hsmpo, observable, noise, level, amplification, n_samples, seed):
    """:func:`noisy_value`, as the full :class:`~mpstab.models.noise.NoisyExpectation`."""
    if amplification == "fold" and level != 1:
        hsmpo = HSMPO(
            fold_circuit(hsmpo.ansatz.circuit, level),
            max_bond_dimension=hsmpo.max_bond_dimension,
            initial_state=hsmpo.initial_state,
            progress=False,
        )
    elif amplification == "scale" and level != 1:
        noise = noise.scaled(level)
    elif amplification not in ("scale", "fold"):
        raise ValueError(
            f"Unknown amplification {amplification!r}, expected 'scale' or 'fold'."
        )
    return hsmpo.noisy_expectation(observable, noise, n_samples=n_samples, seed=seed)


def training_data(
    ansatz,
    observable,
    noise,
    levels,
    amplification,
    n_circuits,
    replacement_probability,
    replacement_method,
    initial_state,
    max_bond_dimension,
    n_samples,
    seed,
    n_workers,
):
    """
    ``(circuits, exact, noisy, stderr)`` of the training set: ``noisy`` has
    shape ``(n,)`` if ``levels`` is ``None`` and ``(n, len(levels))``
    otherwise; ``stderr`` is ``None`` for a callable ``noise``.
    """
    rng = np.random.default_rng(seed)
    circuits = training_circuits(
        ansatz,
        n_circuits,
        replacement_probability,
        replacement_method,
        seed=_seed(rng),
    )
    jobs = [
        (
            circuit,
            observable,
            noise,
            levels,
            amplification,
            initial_state,
            max_bond_dimension,
            n_samples,
            [_seed(rng) for _ in levels or (None,)],
        )
        for circuit in circuits
    ]
    if n_workers > 1 and isinstance(noise, NoiseModel) and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(n_workers, len(jobs))) as pool:
            results = list(pool.map(_evaluate, *zip(*jobs)))
    else:
        results = [_evaluate(*job) for job in jobs]

    exact = np.array([value for value, _, _ in results])
    noisy = np.array([values for _, values, _ in results])
    stderr = np.array([errors for _, _, errors in results])
    if levels is None:
        noisy, stderr = noisy[:, 0], stderr[:, 0]
    if not isinstance(noise, NoiseModel):
        stderr = None
    return circuits, exact, noisy, stderr


def _evaluate(
    circuit,
    observable,
    noise,
    levels,
    amplification,
    initial_state,
    max_bond_dimension,
    n_samples,
    seeds,
):
    """
    ``(exact, noisy, stderr)`` of one training circuit, the last two with one
    entry per level (one entry if ``levels`` is ``None``).
    """
    hsmpo = HSMPO(
        circuit,
        max_bond_dimension=max_bond_dimension,
        initial_state=initial_state,
        progress=False,
    )
    exact = float(hsmpo.expectation(observable))
    noisy, stderr = [], []
    for level, seed in zip(levels or (None,), seeds):
        if isinstance(noise, NoiseModel):
            result = _noisy_result(
                hsmpo,
                observable,
                noise,
                1 if level is None else level,
                amplification,
                n_samples,
                seed,
            )
            noisy.append(float(result.mean))
            stderr.append(float(result.stderr))
        else:
            full = hsmpo.initial_state + hsmpo.ansatz.circuit
            value = noise(full) if level is None else noise(full, level)
            noisy.append(float(np.real(value)))
            stderr.append(np.nan)
    return exact, noisy, stderr


def fit_noise_map(model, noisy, exact, default, design):
    """
    ``(model, params)`` fitted to map ``noisy`` onto ``exact``: the ``default``
    model, linear in its parameters, by least squares on its ``design`` matrix,
    any other with ``curve_fit``.
    """
    if model is None:
        params, *_ = np.linalg.lstsq(design, exact, rcond=None)
        return default, params
    arguments = list(inspect.signature(model).parameters.values())
    if any(a.kind is inspect.Parameter.VAR_POSITIONAL for a in arguments):
        raise ValueError(
            "A custom model must name its parameters, as in f(x, a, b), so "
            "that their number is known."
        )
    params, _ = curve_fit(model, noisy, exact, p0=np.ones(len(arguments) - 1))
    return model, params


def _seed(rng: np.random.Generator) -> int:
    """A fresh seed drawn from ``rng``."""
    return int(rng.integers(2**32))
