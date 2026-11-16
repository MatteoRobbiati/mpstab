"""Noisy expectation values as averages over sign patterns of the rotation chain.

Every set of faults maps to a sign pattern over the dressed rotations and the
observable terms (:mod:`mpstab.noise.compile`), so the noisy expectation value
is a probability-weighted average of noiseless HSMPO expectation values with
some angles negated. The average is stratified by the number of faults ``k``:

- ``k = 0`` and ``k = 1`` are summed exactly, with each fault's probability;
- ``k >= 2`` is sampled exactly from its conditional distribution, and its mean
  is weighted by ``Pr(k >= 2)``.

Patterns are deduplicated, then evaluated in lexicographic order, each one
restarting from the MPS checkpoint before its first rotation that differs from
the previous pattern.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.noise.compile import CompiledNoise


@dataclass
class NoisyExpectation:
    """
    Result of :func:`noisy_expectation`.

    Attributes
    ----------
    mean : float
        The noisy expectation value.
    stderr : float
        Its standard error, from the sampled ``k >= 2`` stratum only: the
        lower strata are exact. Zero when nothing was sampled.
    n_patterns : int
        Distinct rotation sign patterns evaluated on the MPS.
    fidelities : numpy.ndarray
        Truncation fidelity of the MPS of each evaluated pattern, shape
        ``(n_patterns,)``; ``nan`` for an engine that does not expose a norm.
    stratum_weights : tuple of float
        ``(Pr(k = 0), Pr(k = 1), Pr(k >= 2))``.
    n_samples : int
        Fault configurations sampled for the ``k >= 2`` stratum.
    """

    mean: float
    stderr: float
    n_patterns: int
    fidelities: np.ndarray
    stratum_weights: tuple[float, float, float]
    n_samples: int


def noisy_expectation(
    compiled: CompiledNoise,
    observable: Observable,
    n_samples: int = 1000,
    seed: int | None = None,
) -> NoisyExpectation:
    """
    Expectation value of ``observable`` on the noisy circuit.

    Parameters
    ----------
    compiled : CompiledNoise
        The noisy circuit, from :func:`~mpstab.noise.compile_noise`.
    observable : Observable
        A Pauli string or any other format
        :func:`~mpstab.hamiltonians.pauli_terms` accepts.
    n_samples : int, optional
        Fault configurations to sample for the ``k >= 2`` stratum. With ``0``
        that stratum is dropped, which biases the mean by at most
        ``stratum_weights[2]`` times the sum of the absolute coefficients.
    seed : int, optional
        Seed of the sampler.

    Returns
    -------
    NoisyExpectation

    Examples
    --------
    ::

        from qibo import Circuit, gates
        from mpstab import HSMPO
        from mpstab.noise import PauliChannel, compile_noise, noisy_expectation

        circuit = Circuit(2)
        circuit.add(gates.RY(0, theta=0.4))
        circuit.add(gates.CNOT(0, 1))
        events = [
            PauliChannel(after_gate=g, qubits=(0, 1), probs={"XX": 0.02, "ZI": 0.01})
            for g in range(2)
        ]
        compiled = compile_noise(HSMPO(circuit), events)
        result = noisy_expectation(compiled, "ZZ", n_samples=200, seed=0)
        print(f"{result.mean:.4f} +- {result.stderr:.4f}")
    """
    hsmpo = compiled.hsmpo
    constant, terms = pauli_terms(observable, compiled.nqubits)
    dressed = [compiled.dress_observable(pauli) for pauli in terms]
    observable_flips = compiled.observable_flips([pauli for pauli, _ in dressed])

    fault_free = 1.0 - compiled.location_probs
    leave_one_out = _leave_one_out_products(fault_free)
    order_one = compiled.fault_probs * leave_one_out[compiled.fault_location]
    tails = _tail_probabilities(compiled.location_probs)
    weights = (float(np.prod(fault_free)), float(order_one.sum()), float(tails[0, 2]))

    n_rotations = len(compiled.generators)
    if n_samples > 0 and weights[2] > 0:
        rng = np.random.default_rng(seed)
        sampled_rotations, sampled_observable = _sample_two_or_more(
            compiled, observable_flips, tails, n_samples, rng
        )
    else:
        n_samples = 0
        sampled_rotations = np.zeros((0, n_rotations), dtype=bool)
        sampled_observable = np.zeros((0, len(terms)), dtype=bool)

    rotation_patterns = np.vstack(
        [np.zeros((1, n_rotations), dtype=bool), compiled.flips, sampled_rotations]
    )
    observable_patterns = np.vstack(
        [np.zeros((1, len(terms)), dtype=bool), observable_flips, sampled_observable]
    )

    values, fidelities = _pattern_values(compiled, rotation_patterns, dressed)
    total = 0
    for i, coefficient in enumerate(terms.values()):
        signs = 1.0 - 2.0 * observable_patterns[:, i]
        total = total + coefficient * (signs * values[:, i])
    outcomes = constant + total

    n_faults = len(compiled.fault_probs)
    mean = weights[0] * outcomes[0] + order_one @ outcomes[1 : 1 + n_faults]
    stderr = 0.0
    if n_samples > 0:
        sampled = outcomes[1 + n_faults :]
        mean = mean + weights[2] * sampled.mean()
        if n_samples > 1:
            stderr = weights[2] * sampled.std(ddof=1) / np.sqrt(n_samples)

    return NoisyExpectation(
        mean=float(np.real(mean)),
        stderr=float(stderr),
        n_patterns=len(fidelities),
        fidelities=fidelities,
        stratum_weights=weights,
        n_samples=n_samples,
    )


def _leave_one_out_products(values: np.ndarray) -> np.ndarray:
    """``prod(values[m] for m != l)`` for every ``l``, without dividing by zero."""
    prefix = np.concatenate([[1.0], np.cumprod(values)[:-1]])
    suffix = np.concatenate([np.cumprod(values[::-1])[::-1][1:], [1.0]])
    return prefix * suffix


def _tail_probabilities(fault_probs: np.ndarray) -> np.ndarray:
    """
    ``tails[l, j] = Pr(at least j faults among locations l, l+1, ...)`` for
    ``j = 0, 1, 2``, by a backward recursion with only positive terms, so it
    stays accurate when the probabilities are tiny.
    """
    tails = np.zeros((len(fault_probs) + 1, 3))
    tails[:, 0] = 1.0
    for l in range(len(fault_probs) - 1, -1, -1):
        p, q = fault_probs[l], 1.0 - fault_probs[l]
        tails[l, 1] = p + q * tails[l + 1, 1]
        tails[l, 2] = p * tails[l + 1, 1] + q * tails[l + 1, 2]
    return tails


def _sample_two_or_more(compiled, observable_flips, tails, n_samples, rng):
    """
    Sample fault configurations conditioned on at least two faults, and return
    their rotation and observable sign patterns.

    Locations are visited in order; each faults with its probability
    conditioned on the faults still needed being reachable from the
    locations left, so every sample is accepted.
    """
    location_probs = compiled.location_probs
    starts = np.searchsorted(
        compiled.fault_location, np.arange(len(location_probs) + 1)
    )

    counts = np.zeros(n_samples, dtype=int)
    rotations = np.zeros((n_samples, compiled.flips.shape[1]), dtype=bool)
    observable = np.zeros((n_samples, observable_flips.shape[1]), dtype=bool)
    for l, p in enumerate(location_probs):
        needed = np.maximum(2 - counts, 0)
        reachable_after = np.where(needed > 0, tails[l + 1, needed - 1], 1.0)
        reachable_now = tails[l, needed]
        conditional = np.divide(
            p * reachable_after,
            reachable_now,
            out=np.zeros(n_samples),
            where=reachable_now > 0,
        )
        hits = np.flatnonzero(rng.random(n_samples) < conditional)
        if hits.size == 0:
            continue
        cumulative = np.cumsum(compiled.fault_probs[starts[l] : starts[l + 1]])
        choice = np.searchsorted(
            cumulative, rng.random(hits.size) * cumulative[-1], side="right"
        )
        faults = starts[l] + np.minimum(choice, len(cumulative) - 1)
        rotations[hits] ^= compiled.flips[faults]
        observable[hits] ^= observable_flips[faults]
        counts[hits] += 1
    return rotations, observable


def _pattern_values(compiled, patterns, dressed):
    """
    Each observable term's dressed expectation value for each rotation sign
    pattern, as ``(values, fidelities)`` with ``values`` shaped
    ``(len(patterns), len(dressed))``.

    Distinct patterns are evaluated once, in lexicographic order, from a stack
    of checkpoints: ``checkpoints[j]`` is the MPS after the first ``j``
    rotations of the current pattern. Each rotation's MPO is built once per
    sign and reused.
    """
    hsmpo = compiled.hsmpo
    engine = hsmpo.tn_engine
    n_rotations = len(compiled.generators)
    if n_rotations == 0:
        unique = np.zeros((1, 0), dtype=bool)
        inverse = np.zeros(len(patterns), dtype=int)
    else:
        unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

    operators = {}

    def rotation(j: int, flipped: bool):
        if (j, flipped) not in operators:
            angle = -compiled.angles[j] if flipped else compiled.angles[j]
            operators[j, flipped] = engine.rotation_operator(
                compiled.generators[j], angle
            )
        return operators[j, flipped]

    observables = [(engine.pauli_mpo(pauli), sign) for pauli, sign in dressed]
    unique_values = np.zeros((len(unique), len(dressed)))
    fidelities = np.zeros(len(unique))

    checkpoints = [hsmpo.initial_mps()]
    previous = None
    for row, pattern in enumerate(unique):
        start = 0 if previous is None else int(np.argmax(pattern != previous))
        del checkpoints[start + 1 :]
        for j in range(start, n_rotations):
            state = copy.deepcopy(checkpoints[-1])
            engine.apply_rotation(
                state, rotation(j, bool(pattern[j])), hsmpo.max_bond_dimension
            )
            checkpoints.append(state)
        previous = pattern

        state = checkpoints[-1]
        for i, (mpo, sign) in enumerate(observables):
            unique_values[row, i] = (
                np.real(engine.expval(state_circuit=state, operator=mpo)) * sign
            )
        fidelities[row] = (
            np.real(state.norm(squared=True)) if hasattr(state, "norm") else np.nan
        )

    return unique_values[inverse], fidelities
