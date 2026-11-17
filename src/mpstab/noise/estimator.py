"""Noisy expectation values as averages over sign patterns of the rotation chain.

Every set of faults maps to a sign pattern over the dressed rotations and the
observable terms (:mod:`mpstab.noise.compile`), so the noisy expectation value
of Pauli noise is a probability-weighted average of noiseless HSMPO
expectation values with some angles negated. A coherence ``(A, B)`` of a
non-Pauli channel flips the ket's and the bra's angles differently, and
contributes the transition amplitude ``<phi_B| B' O A' |phi_A>`` between the
two patterns with the complex weight ``chi[A, B]``.

The sum over configurations is stratified by the number ``k`` of locations
that do something, each configuration weighted by the product of its entries:
the no-fault probability, a fault's probability or a coherence's complex
weight.

- ``k = 0`` and ``k = 1`` are summed exactly;
- ``k >= 2`` is sampled from its conditional distribution, with coherences
  drawn by ``|chi|`` and their phases carried as a weight, and its mean is
  multiplied by the total weight of that stratum.

For Pauli noise the weights are probabilities. Coherences make them a
quasi-probability, whose total exceeds 1 by about their summed ``|chi|``.

Before any of this, the circuit is restricted to the light cone of the
observable and the coherences
(:meth:`~mpstab.noise.CompiledNoise.restricted_to`): rotations that cannot
affect them are dropped, and so are faults that then do nothing. Patterns are
deduplicated, then evaluated in lexicographic order, each one restarting from
the MPS checkpoint before its first rotation that differs from the previous
pattern; the bra of a transition branches off the ket's checkpoint at their
first difference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

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
        Distinct rotation sign patterns evaluated on the MPS as expectation
        values.
    fidelities : numpy.ndarray
        Truncation fidelity of the MPS of each of those patterns, shape
        ``(n_patterns,)``; ``nan`` for an engine that does not expose a norm.
    stratum_weights : tuple of float
        Total weight of the configurations with ``k = 0``, ``k = 1`` and
        ``k >= 2`` locations doing something, counting only the faults and
        coherences that can affect the observable. For Pauli noise these are
        probabilities summing to 1; coherences count with ``|chi|``.
    n_samples : int
        Configurations sampled for the ``k >= 2`` stratum.
    n_transitions : int
        Transition amplitudes evaluated for coherences, exact and sampled.
    """

    mean: float
    stderr: float
    n_patterns: int
    fidelities: np.ndarray
    stratum_weights: tuple[float, float, float]
    n_samples: int
    n_transitions: int = 0


def noisy_expectation(
    compiled: CompiledNoise,
    observable: Observable,
    n_samples: int = 1000,
    seed: int | None = None,
    light_cone: bool = True,
    n_workers: int = 1,
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
        Configurations to sample for the ``k >= 2`` stratum. With ``0`` that
        stratum is dropped, which biases the mean by at most
        ``stratum_weights[2]`` times the sum of the absolute coefficients.
    seed : int, optional
        Seed of the sampler.
    light_cone : bool, optional
        Restrict the circuit to the light cone first. This is exact, and it
        leaves fewer rotations per pattern and fewer faults, so a smaller
        ``k >= 2`` weight and a smaller standard error. Turn it off only to
        compare against the unrestricted computation.
    n_workers : int, optional
        Processes evaluating the sign patterns of Pauli faults. The distinct
        patterns are split into contiguous blocks, so the result is identical
        for any value; each extra process costs a start-up of about a second,
        so it pays off only when there are many patterns or the MPS is large.
        Transition amplitudes are evaluated in this process.

    Returns
    -------
    NoisyExpectation

    Examples
    --------
    ::

        from qibo import Circuit, gates
        from mpstab import HSMPO
        from mpstab.noise.compile import compile_noise
        from mpstab.noise.estimator import noisy_expectation
        from mpstab.noise.events import PauliChannel

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
    constant, terms = pauli_terms(observable, compiled.nqubits)
    dressed = [compiled.dress_observable(pauli) for pauli in terms]
    if light_cone:
        compiled = compiled.restricted_to([pauli for pauli, _ in dressed])
    observable_flips = compiled.observable_flips([pauli for pauli, _ in dressed])
    coefficients = np.array(list(terms.values()), dtype=float)

    null = 1.0 - compiled.location_probs
    mass = compiled.location_probs + compiled.coherence_mass
    leave_one_out = _leave_one_out_products(null)
    order_one = compiled.fault_probs * leave_one_out[compiled.fault_location]
    coherence_one = (
        2 * compiled.coherence_weights * leave_one_out[compiled.coherence_location]
    )
    tails = _tail_weights(mass, null)
    weights = (
        float(np.prod(null)),
        float(order_one.sum() + np.abs(coherence_one).sum()),
        float(tails[0, 2]),
    )

    n_rotations = len(compiled.generators)
    if n_samples > 0 and weights[2] > 0:
        rng = np.random.default_rng(seed)
        samples = _sample_two_or_more(compiled, observable_flips, tails, n_samples, rng)
    else:
        n_samples = 0
        samples = _Samples.empty(n_rotations, len(terms))
    diagonal = ~samples.coherent

    # Expectation values: no fault, each fault, each sample without coherences.
    rotation_patterns = np.vstack(
        [np.zeros((1, n_rotations), dtype=bool), compiled.flips, samples.ket[diagonal]]
    )
    observable_patterns = np.vstack(
        [
            np.zeros((1, len(terms)), dtype=bool),
            observable_flips,
            samples.observable[diagonal],
        ]
    )
    values, fidelities = _pattern_values(
        compiled, rotation_patterns, dressed, n_workers
    )
    outcomes = constant + ((1.0 - 2.0 * observable_patterns) * values) @ coefficients

    # Transition amplitudes: each coherence, each sample with coherences.
    n_coherences = len(compiled.coherence_weights)
    jobs = [
        (
            compiled.ket_flips[c],
            compiled.bra_flips[c],
            [compiled.coherence_ket[c]],
            [compiled.coherence_bra[c]],
        )
        for c in range(n_coherences)
    ]
    for s in np.flatnonzero(samples.coherent):
        jobs.append((samples.ket[s], samples.bra[s], *samples.operators[s]))
    transitions = _transition_values(compiled, jobs, dressed, coefficients, constant)

    n_faults = len(compiled.fault_probs)
    mean = weights[0] * outcomes[0] + order_one @ outcomes[1 : 1 + n_faults]
    mean += float(np.real(coherence_one @ transitions[:n_coherences]))
    stderr = 0.0
    if n_samples > 0:
        sampled = np.empty(n_samples)
        sampled[diagonal] = outcomes[1 + n_faults :]
        sampled[~diagonal] = np.real(
            samples.phase[~diagonal] * transitions[n_coherences:]
        )
        mean += weights[2] * sampled.mean()
        if n_samples > 1:
            stderr = weights[2] * sampled.std(ddof=1) / np.sqrt(n_samples)

    return NoisyExpectation(
        mean=float(np.real(mean)),
        stderr=float(stderr),
        n_patterns=len(fidelities),
        fidelities=fidelities,
        stratum_weights=weights,
        n_samples=n_samples,
        n_transitions=len(jobs),
    )


def _leave_one_out_products(values: np.ndarray) -> np.ndarray:
    """``prod(values[m] for m != l)`` for every ``l``, without dividing by zero."""
    prefix = np.concatenate([[1.0], np.cumprod(values)[:-1]])
    suffix = np.concatenate([np.cumprod(values[::-1])[::-1][1:], [1.0]])
    return prefix * suffix


def _tail_weights(mass: np.ndarray, null: np.ndarray) -> np.ndarray:
    """
    ``tails[l, j]``: total weight of the configurations of locations
    ``l, l+1, ...`` in which at least ``j`` of them do something, for
    ``j = 0, 1, 2``, where a location weighs ``mass[l]`` when it does and
    ``null[l]`` when it does not. A backward recursion with only positive
    terms, so it stays accurate when the masses are tiny. For Pauli noise,
    ``mass + null = 1`` and these are probabilities.
    """
    tails = np.zeros((len(mass) + 1, 3))
    tails[-1, 0] = 1.0
    for l in range(len(mass) - 1, -1, -1):
        m, q = mass[l], null[l]
        tails[l, 0] = (m + q) * tails[l + 1, 0]
        tails[l, 1] = m * tails[l + 1, 0] + q * tails[l + 1, 1]
        tails[l, 2] = m * tails[l + 1, 1] + q * tails[l + 1, 2]
    return tails


@dataclass
class _Samples:
    """
    Configurations sampled from the ``k >= 2`` stratum.

    ``ket`` and ``bra`` are their rotation sign patterns and ``observable``
    the observable sign flips of their faults. A sample with a coherence
    (``coherent``) has a complex ``phase`` and its Paulis in ``operators``:
    ``(ket Paulis, bra Paulis)`` in circuit order, faults included.
    """

    ket: np.ndarray
    bra: np.ndarray
    observable: np.ndarray
    coherent: np.ndarray
    phase: np.ndarray
    operators: dict[int, tuple[list[str], list[str]]]

    @classmethod
    def empty(cls, n_rotations: int, n_terms: int) -> _Samples:
        return cls(
            ket=np.zeros((0, n_rotations), dtype=bool),
            bra=np.zeros((0, n_rotations), dtype=bool),
            observable=np.zeros((0, n_terms), dtype=bool),
            coherent=np.zeros(0, dtype=bool),
            phase=np.ones(0, dtype=complex),
            operators={},
        )


def _sample_two_or_more(compiled, observable_flips, tails, n_samples, rng) -> _Samples:
    """
    Sample configurations conditioned on at least two locations doing
    something.

    Locations are visited in order; each one acts with its weight
    conditioned on the actions still needed being reachable from the
    locations left, so every sample is accepted. An acting location picks
    one of its faults by probability or one of its coherences by ``2 |chi|``,
    in either orientation with equal odds.
    """
    n_locations = len(compiled.location_probs)
    fault_starts = np.searchsorted(compiled.fault_location, np.arange(n_locations + 1))
    coherence_starts = np.searchsorted(
        compiled.coherence_location, np.arange(n_locations + 1)
    )
    coherence_mass = 2 * np.abs(compiled.coherence_weights)
    unit_phase = np.divide(
        compiled.coherence_weights,
        np.abs(compiled.coherence_weights),
        out=np.ones_like(compiled.coherence_weights),
        where=coherence_mass > 0,
    )
    mass = compiled.location_probs + compiled.coherence_mass

    samples = _Samples.empty(compiled.flips.shape[1], observable_flips.shape[1])
    samples.ket = np.zeros((n_samples, compiled.flips.shape[1]), dtype=bool)
    samples.bra = samples.ket.copy()
    samples.observable = np.zeros((n_samples, observable_flips.shape[1]), dtype=bool)
    samples.coherent = np.zeros(n_samples, dtype=bool)
    samples.phase = np.ones(n_samples, dtype=complex)
    counts = np.zeros(n_samples, dtype=int)
    log = []  # per acting location: (samples, fault index or -1 - coherence, swapped)

    for l in range(n_locations):
        if mass[l] == 0:
            continue
        needed = np.maximum(2 - counts, 0)
        reachable_after = tails[l + 1, np.maximum(needed - 1, 0)]
        reachable_now = tails[l, needed]
        conditional = np.divide(
            mass[l] * reachable_after,
            reachable_now,
            out=np.zeros(n_samples),
            where=reachable_now > 0,
        )
        hits = np.flatnonzero(rng.random(n_samples) < conditional)
        if hits.size == 0:
            continue

        faults = np.arange(fault_starts[l], fault_starts[l + 1])
        coherences = np.arange(coherence_starts[l], coherence_starts[l + 1])
        options = np.concatenate(
            [compiled.fault_probs[faults], coherence_mass[coherences]]
        )
        cumulative = np.cumsum(options)
        choice = np.searchsorted(
            cumulative, rng.random(hits.size) * cumulative[-1], side="right"
        )
        choice = np.minimum(choice, len(options) - 1)
        is_fault = choice < len(faults)
        swapped = rng.random(hits.size) < 0.5

        fault_hits, fault = hits[is_fault], faults[choice[is_fault]]
        samples.ket[fault_hits] ^= compiled.flips[fault]
        samples.bra[fault_hits] ^= compiled.flips[fault]
        samples.observable[fault_hits] ^= observable_flips[fault]

        coherence_hits = hits[~is_fault]
        coherence = coherences[choice[~is_fault] - len(faults)]
        swap = swapped[~is_fault, None]
        samples.ket[coherence_hits] ^= np.where(
            swap, compiled.bra_flips[coherence], compiled.ket_flips[coherence]
        )
        samples.bra[coherence_hits] ^= np.where(
            swap, compiled.ket_flips[coherence], compiled.bra_flips[coherence]
        )
        phase = unit_phase[coherence]
        samples.phase[coherence_hits] *= np.where(swap[:, 0], phase.conj(), phase)
        samples.coherent[coherence_hits] = True

        index = np.empty(hits.size, dtype=int)
        index[is_fault] = fault
        index[~is_fault] = -1 - coherence
        log.append((hits, index, swapped))
        counts[hits] += 1

    coherent = set(np.flatnonzero(samples.coherent).tolist())
    for hits, index, swapped in log:
        for s, i, swap in zip(hits.tolist(), index.tolist(), swapped.tolist()):
            if s not in coherent:
                continue
            kets, bras = samples.operators.setdefault(s, ([], []))
            if i >= 0:
                kets.append(compiled.fault_paulis[i])
                bras.append(compiled.fault_paulis[i])
            else:
                ket, bra = (
                    compiled.coherence_ket[-1 - i],
                    compiled.coherence_bra[-1 - i],
                )
                kets.append(bra if swap else ket)
                bras.append(ket if swap else bra)
    return samples


def _pattern_values(compiled, patterns, dressed, n_workers: int = 1):
    """
    Each observable term's dressed expectation value for each rotation sign
    pattern, as ``(values, fidelities)`` with ``values`` shaped
    ``(len(patterns), len(dressed))``, and fidelities one per distinct pattern.

    Distinct patterns are sorted lexicographically and split into
    ``n_workers`` contiguous blocks, each evaluated by
    :func:`_evaluate_patterns` in its own process (in this one if
    ``n_workers == 1``).
    """
    hsmpo = compiled.hsmpo
    n_rotations = len(compiled.generators)
    if n_rotations == 0:
        unique = np.zeros((1, 0), dtype=bool)
        inverse = np.zeros(len(patterns), dtype=int)
    else:
        unique, inverse = np.unique(patterns, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)

    args = (
        hsmpo.tn_engine,
        hsmpo.initial_mps(),
        list(compiled.generators),
        np.asarray(compiled.angles),
        hsmpo.max_bond_dimension,
        [pauli for pauli, _ in dressed],
        np.array([sign for _, sign in dressed], dtype=float),
    )
    n_workers = max(1, min(int(n_workers), len(unique)))
    if n_workers == 1:
        values, fidelities = _evaluate_patterns(*args, unique)
    else:
        from concurrent.futures import ProcessPoolExecutor

        blocks = np.array_split(unique, n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(
                pool.map(_evaluate_patterns, *zip(*[(*args, b) for b in blocks]))
            )
        values = np.concatenate([v for v, _ in results])
        fidelities = np.concatenate([f for _, f in results])
    return values[inverse], fidelities


class _Chain:
    """
    MPS checkpoints along the rotation chain, for sign patterns visited in
    lexicographic order: ``checkpoints[j]`` is the state after the first ``j``
    rotations of the current pattern, so consecutive patterns share their
    common prefix. Each rotation's operator is built once per sign.
    """

    def __init__(self, engine, initial_mps, generators, angles, max_bond):
        self.engine, self.generators, self.angles = engine, generators, angles
        self.max_bond = max_bond
        self.operators = {}
        self.checkpoints = [initial_mps]
        self.current = None

    def _rotation(self, j: int, flipped: bool):
        if (j, flipped) not in self.operators:
            angle = -self.angles[j] if flipped else self.angles[j]
            self.operators[j, flipped] = self.engine.rotation_operator(
                self.generators[j], angle
            )
        return self.operators[j, flipped]

    def _evolve(self, state, pattern, start: int, record: bool):
        for j in range(start, len(self.generators)):
            state = self.engine.copy_state(state)
            self.engine.apply_rotation(
                state, self._rotation(j, bool(pattern[j])), self.max_bond
            )
            if record:
                self.checkpoints.append(state)
        return state

    def state(self, pattern):
        """The final state of ``pattern``, which becomes the current one."""
        if self.current is not None and np.array_equal(pattern, self.current):
            return self.checkpoints[-1]
        start = 0 if self.current is None else _first_difference(pattern, self.current)
        del self.checkpoints[start + 1 :]
        self.current = np.array(pattern, dtype=bool)
        return self._evolve(self.checkpoints[-1], pattern, start, record=True)

    def branch(self, pattern):
        """
        The final state of ``pattern``, branched off the current pattern's
        checkpoint at their first difference; the current one is unchanged.
        """
        if np.array_equal(pattern, self.current):
            return self.checkpoints[-1]
        start = _first_difference(pattern, self.current)
        return self._evolve(self.checkpoints[start], pattern, start, record=False)


def _first_difference(a, b) -> int:
    return int(np.argmax(np.asarray(a) != np.asarray(b)))


def _evaluate_patterns(
    engine, initial_mps, generators, angles, max_bond, paulis, signs, patterns
):
    """
    Evaluate sorted, distinct sign ``patterns`` along one :class:`_Chain`.
    """
    chain = _Chain(engine, initial_mps, generators, angles, max_bond)
    values = np.zeros((len(patterns), len(paulis)))
    fidelities = np.zeros(len(patterns))
    for row, pattern in enumerate(patterns):
        state = chain.state(pattern)
        if paulis:
            expvals = engine.pauli_expvals(state, paulis)
            values[row] = np.real(np.asarray(expvals, dtype=complex)) * signs
        fidelities[row] = (
            np.real(state.norm(squared=True)) if hasattr(state, "norm") else np.nan
        )
    return values, fidelities


def _transition_values(compiled, jobs, dressed, coefficients, constant):
    """
    ``<phi_bra| B^dag O A |phi_ket>`` for each job ``(ket pattern, bra
    pattern, ket Paulis, bra Paulis)``, with ``A`` and ``B`` the products of
    the Paulis in circuit order and ``O`` the dressed observable, its
    constant included.

    Jobs are visited in lexicographic order of their ket patterns along one
    :class:`_Chain`, each bra branching off its ket.
    """
    values = np.zeros(len(jobs), dtype=complex)
    if not jobs:
        return values
    hsmpo = compiled.hsmpo
    engine = hsmpo.tn_engine
    n = compiled.nqubits
    observables = [stim.PauliString(pauli) * int(sign) for pauli, sign in dressed]
    observables.append(stim.PauliString(n))
    weights = np.append(coefficients, constant).astype(complex)

    chain = _Chain(
        engine,
        hsmpo.initial_mps(),
        list(compiled.generators),
        np.asarray(compiled.angles),
        hsmpo.max_bond_dimension,
    )
    kets = np.array([job[0] for job in jobs], dtype=bool).reshape(len(jobs), -1)
    order = np.lexsort(kets.T[::-1]) if kets.shape[1] else np.arange(len(jobs))
    for k in order:
        ket_pattern, bra_pattern, ket_paulis, bra_paulis = jobs[k]
        ket_state = chain.state(ket_pattern)
        bra_state = chain.branch(bra_pattern)

        a, b = stim.PauliString(n), stim.PauliString(n)
        for pauli in ket_paulis:
            a = stim.PauliString(pauli) * a
        for pauli in bra_paulis:
            b = stim.PauliString(pauli) * b
        b_dagger = b.copy()
        b_dagger.sign = complex(b.sign).conjugate()

        labels, phases = [], []
        for observable in observables:
            product = b_dagger * observable * a
            labels.append("".join("IXYZ"[p] for p in product))
            phases.append(product.sign)
        amplitudes = np.asarray(
            engine.pauli_transitions(bra_state, ket_state, labels), dtype=complex
        )
        values[k] = np.sum(weights * np.array(phases) * amplitudes)
    return values
