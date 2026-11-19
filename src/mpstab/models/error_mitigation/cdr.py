"""Clifford data regression (CDR) and its variable-noise extension (vnCDR).

Both learn a map from noisy expectation values to noiseless ones on training
circuits close to the target, then apply it to the target's noisy value:

- the training circuits are low-magic variants of the target
  (:func:`~mpstab.models.error_mitigation.training_circuits`);
- their noiseless values come from HSMPO, cheap because they carry little
  magic;
- their noisy values come either from a :class:`~mpstab.models.noise.NoiseModel`,
  simulated by :meth:`~mpstab.HSMPO.noisy_expectation`, or from any callable,
  such as one running the circuit on hardware.

References:
    1. P. Czarnik, A. Arrasmith et al., *Error mitigation with Clifford
       quantum-circuit data*, `arXiv:2005.10189 <https://arxiv.org/abs/2005.10189>`_.
    2. A. Lowe, M. H. Gordon et al., *Unified approach to data-driven quantum
       error mitigation*, `arXiv:2011.01157 <https://arxiv.org/abs/2011.01157>`_.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Callable, Sequence, Union

import numpy as np
from qibo import Circuit
from scipy.optimize import curve_fit

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import Observable
from mpstab.models.ansatze import Ansatz
from mpstab.models.error_mitigation.training import fold_circuit, training_circuits
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

        from mpstab import HSMPO
        from mpstab.models.ansatze import HardwareEfficient
        from mpstab.models.error_mitigation import cdr
        from mpstab.models.noise import DepolarizingError, NoiseModel

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
            a training circuit (:func:`training_circuits`).
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
        :class:`MitigationNoiseMap`
    """
    circuits, exact, noisy, stderr = _training_data(
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
    model, params = _fit(model, noisy, exact, _linear, design)
    return MitigationNoiseMap(
        params, exact, noisy, model, noisy_stderr=stderr, circuits=circuits
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
    :func:`noisy_value` does it::

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
            :func:`fold_circuit` of it on hardware. The circuit it receives
            starts with ``initial_state``, if any.
        noise_levels: the noise amplification factors, ``1`` being the
            unamplified noise.
        amplification: how a noise model is amplified: ``"scale"`` multiplies
            every error's strength by the level
            (:meth:`~mpstab.models.noise.NoiseModel.scaled`), ``"fold"`` runs
            the folded circuit (:func:`fold_circuit`), which needs odd integer
            levels. Ignored for a callable.
        n_training_circuits: training circuits to sample.
        replacement_probability: chance of replacing each non-Clifford gate in
            a training circuit (:func:`training_circuits`).
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
        :class:`MitigationNoiseMap`, with :attr:`~MitigationNoiseMap.noise_levels` set.
    """
    noise_levels = tuple(noise_levels)
    if amplification not in ("scale", "fold"):
        raise ValueError(
            f"Unknown amplification {amplification!r}, expected 'scale' or 'fold'."
        )
    circuits, exact, noisy, stderr = _training_data(
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
    model, params = _fit(model, noisy, exact, _combination, noisy)
    return MitigationNoiseMap(
        params,
        exact,
        noisy,
        model,
        noise_levels=noise_levels,
        noisy_stderr=stderr,
        circuits=circuits,
    )


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
        amplification: ``"scale"`` or ``"fold"``, as in :func:`vncdr`.
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


def _training_data(
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


def _fit(model, noisy, exact, default, design):
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


def _linear(x, a, b):
    """``a * x + b``, the CDR model."""
    return a * x + b


def _combination(x, *params):
    """``x @ params``, the vnCDR model: a linear combination of the noise levels."""
    return x @ np.asarray(params)


def _seed(rng: np.random.Generator) -> int:
    """A fresh seed drawn from ``rng``."""
    return int(rng.integers(2**32))
