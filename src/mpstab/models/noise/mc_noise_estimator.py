"""Noisy expectation values as averages over sign patterns of the rotation chain.

Every set of faults maps to a sign pattern over the dressed rotations and the
observable terms (:mod:`mpstab.models.noise.compile`), so the noisy expectation
value of Pauli noise is a probability-weighted average of noiseless HSMPO
expectation values with some angles negated. A coherence ``(A, B)`` of a
non-Pauli channel flips the ket's and the bra's angles differently, and
contributes the transition amplitude ``<phi_B| B' O A' |phi_A>`` between the
two patterns with the complex weight ``chi[A, B]``. A reset block contributes
the sum of such amplitudes over its terms, with a probability.

The sum over configurations is stratified by the number ``k`` of locations
that do something, each configuration weighted by the product of its entries:
the no-fault probability, a fault's or a block's probability, or a
coherence's complex weight.

- ``k = 0`` is the noiseless value;
- ``k = 1`` is summed exactly or, with ``order_one_samples``, summed exactly
  over its heaviest entries and importance-sampled over the rest, by a bound
  on how far each entry can move the value;
- ``k >= 2`` is sampled from its conditional distribution, with coherences
  drawn by ``|chi|`` and their phases carried as a weight, and its mean is
  multiplied by the total weight of that stratum. Each sample is compared
  with two models of its configuration built from the ``k = 1`` values, an
  additive and a multiplicative one, whose sums over the stratum are known
  in closed form, so that only the residual is sampled (control variates).

For Pauli noise and reset blocks the weights are probabilities. Coherences
make them a quasi-probability, whose total exceeds 1 by about their summed
``|chi|``.

Before any of this, the circuit is restricted to the light cone of the
observable and the coherences
(:meth:`~mpstab.models.noise.CompiledNoise.restricted_to`), and commuting
rotations are reordered so that the MPS orthogonality center moves little.
Every quantity is then a job: a ket pattern, a bra pattern and the Paulis
acting on each. Jobs with the same pattern on both sides are expectation
values on one MPS; the others are transition amplitudes between two,
computed once per pair of orientations. Both are evaluated along chains of
MPS checkpoints shared between patterns with a common prefix.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import stim

from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.models.noise.compile import CompiledNoise
from mpstab.pauli import anticommutation

#: Smallest ``|<O_t>|`` of the noiseless circuit for which a term gets the
#: multiplicative model: ratios to a vanishing value carry no information.
_PRODUCT_FLOOR = 1e-3

#: Most terms a sampled configuration is expanded into, exactly, before its
#: further blocks are sampled term by term: two resets (8 terms each) and up.
_MAX_BLOCK_TERMS = 16


@dataclass
class NoisyExpectation:
    """
    Result of :func:`noisy_expectation`.

    Attributes:
        mean: the noisy expectation value.
        stderr: its standard error, from the sampled parts only (the ``k >= 2``
            stratum, and the sampled tail of ``k = 1`` if any). Zero when
            nothing was sampled.
        n_patterns: distinct rotation sign patterns evaluated on the MPS as
            expectation values.
        fidelities: truncation fidelity of the MPS of each of those patterns,
            shape ``(n_patterns,)``; ``nan`` for an engine that does not expose
            a norm.
        stratum_weights: total weight of the configurations with ``k = 0``,
            ``k = 1`` and ``k >= 2`` locations doing something, counting only
            the faults, coherences and blocks that can affect the observable.
            For Pauli noise and blocks these are probabilities summing to 1;
            coherences count with ``|chi|``.
        n_samples: configurations sampled for the ``k >= 2`` stratum.
        n_transitions: transition amplitudes evaluated, each one paired with
            its conjugate.
        order_one_samples: draws from the sampled tail of the ``k = 1`` stratum
            (0 when it is summed exactly).
        control_coefficients: weights of the additive and multiplicative
            control variates fitted on the ``k >= 2`` samples.
        variance_reduction: variance of the raw ``k >= 2`` samples over that
            of their residuals after the control variates; 1 without them.
    """

    mean: float
    stderr: float
    n_patterns: int
    fidelities: np.ndarray
    stratum_weights: tuple[float, float, float]
    n_samples: int
    n_transitions: int = 0
    order_one_samples: int = 0
    control_coefficients: tuple[float, ...] = ()
    variance_reduction: float = 1.0


def noisy_expectation(
    compiled: CompiledNoise,
    observable: Observable,
    n_samples: int = 1000,
    seed: int | None = None,
    light_cone: bool = True,
    n_workers: int = 1,
    control_variates: bool = True,
    order_one_samples: int | None = None,
) -> NoisyExpectation:
    """
    Expectation value of ``observable`` on the noisy circuit.

    Args:
        compiled: the noisy circuit, from :func:`~mpstab.models.noise.compile_noise`.
        observable: a Pauli string or any other format
            :func:`~mpstab.hamiltonians.pauli_terms` accepts.
        n_samples: configurations to sample for the ``k >= 2`` stratum. With
            ``0`` that stratum is dropped, which biases the mean by at most
            ``stratum_weights[2]`` times the sum of the absolute coefficients.
        seed: seed of the sampler.
        light_cone: restrict the circuit to the observable's light cone first.
            This is exact, and it leaves fewer rotations per pattern and fewer
            faults, so a smaller ``k >= 2`` weight and a smaller standard error.
            Turn it off only to compare against the unrestricted computation.
        n_workers: processes evaluating sign patterns and transition
            amplitudes. The jobs are split into contiguous blocks, so the result
            is identical for any value; each extra process costs a start-up of
            about a second, so it pays off only for large runs.
        control_variates: subtract from each ``k >= 2`` sample the additive and
            multiplicative models of its configuration, and add back their
            exact sums. Unbiased up to ``O(1 / n_samples)`` from fitting the
            two coefficients; the standard error shrinks by how well the models
            predict the samples.
        order_one_samples: ``None`` sums the ``k = 1`` stratum exactly. An
            integer ``N`` sums exactly only the entries whose weight times
            bound exceeds ``1/N`` of the remaining total, and samples the rest
            ``N`` times in proportion to that product, without bias. The bound
            on a fault's shift, ``min(2, 4 sum |sin(theta_j / 2)|)`` over the
            rotations it flips, costs nothing, so entries that cannot move the
            value are never evaluated.

    Returns:
        The :class:`NoisyExpectation`.

    Example::

        from qibo import Circuit, gates
        from mpstab import HSMPO
        from mpstab.models.noise.compile import compile_noise
        from mpstab.models.noise.mc_noise_estimator import noisy_expectation
        from mpstab.models.noise.events import PauliChannel

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
    dressed = compiled.dress_observables(terms)
    if light_cone:
        compiled = compiled.restricted_to([pauli for pauli, _ in dressed])
    compiled = _sweep_ordered(compiled)
    weights_per_term = np.append(list(terms.values()), constant).astype(complex)
    rng = np.random.default_rng(seed)

    null = 1.0 - compiled.location_probs
    mass = compiled.location_probs + compiled.coherence_mass
    tails = _tail_weights(mass, null)
    items = _OrderOne(compiled, _leave_one_out_products(null), dressed)
    weights = (float(np.prod(null)), float(items.weight.sum()), float(tails[0, 2]))

    jobs = _Jobs(compiled, dressed)
    base_job = jobs.add_diagonal(np.zeros(len(compiled.generators), bool), ())
    exact, draws = items.split(order_one_samples, weights_per_term, rng)
    item_jobs = {int(i): items.jobs(i, jobs) for i in np.union1d(exact, draws)}

    if n_samples > 0 and weights[2] > 0:
        samples = _sample_two_or_more(compiled, tails, n_samples, rng)
        expansions = [
            _expand(compiled, actions, jobs, rng, _MAX_BLOCK_TERMS)
            for actions in samples.actions
        ]
    else:
        n_samples, samples, expansions = 0, None, []

    values, fidelities, n_transitions = jobs.evaluate(n_workers)
    v0 = np.real(values[base_job])

    def combine(expansion):
        return sum(c * values[j] for j, c in expansion)

    vectors = {i: combine(expansion) for i, expansion in item_jobs.items()}
    scalars = {i: float(np.real(v @ weights_per_term)) for i, v in vectors.items()}
    bases = items.bases(v0, weights_per_term)

    # k = 0 and k = 1.
    mean = weights[0] * float(v0 @ np.real(weights_per_term))
    mean += sum(items.weight[i] * scalars[int(i)] for i in exact)
    variance = 0.0
    if order_one_samples is not None:
        mean += float(items.weight[items.tail] @ bases[items.tail])
        if len(draws):
            shifts = np.array(
                [(scalars[int(i)] - bases[i]) / items.bound[i] for i in draws]
            )
            mean += items.tail_mass * shifts.mean()
            if len(draws) > 1:
                variance += items.tail_mass**2 * shifts.var(ddof=1) / len(draws)

    # k >= 2.
    fitted, reduction = (), 1.0
    if n_samples > 0:
        sampled = np.array(
            [
                float(np.real(phase * (combine(expansion) @ weights_per_term)))
                for phase, expansion in zip(samples.phase, expansions)
            ]
        )
        estimate = sampled.mean()
        spread = sampled.var(ddof=1) if n_samples > 1 else 0.0
        if control_variates and n_samples > 3:
            models = _Models(
                compiled, items, vectors, values, v0, null, weights_per_term
            )
            controls = np.array(
                [
                    models.sample(actions, phase)
                    for actions, phase in zip(samples.actions, samples.phase)
                ]
            )
            expected = models.stratum_sums() / weights[2]
            estimate, residual, fitted = _regress(sampled, controls, expected)
            reduction = spread / residual if residual > 0 else float("inf")
            spread = residual
        mean += weights[2] * estimate
        variance += weights[2] ** 2 * spread / n_samples

    return NoisyExpectation(
        mean=float(mean),
        stderr=float(np.sqrt(variance)),
        n_patterns=len(fidelities),
        fidelities=fidelities,
        stratum_weights=weights,
        n_samples=n_samples,
        n_transitions=n_transitions,
        order_one_samples=len(draws),
        control_coefficients=tuple(float(c) for c in fitted),
        variance_reduction=float(reduction),
    )


# -- Weights -------------------------------------------------------------------------


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


def _two_or_more(active: np.ndarray, null: np.ndarray) -> np.ndarray:
    """
    Sum over the configurations with at least two acting locations of
    ``prod_l (null_l or active[l])``, for ``active`` of shape ``(L, ...)``
    with signed or complex entries: the recursion of :func:`_tail_weights`.
    """
    shape = active.shape[1:]
    t0 = np.ones(shape, complex)
    t1, t2 = np.zeros(shape, complex), np.zeros(shape, complex)
    for l in range(len(null) - 1, -1, -1):
        a, q = active[l], null[l]
        t0, t1, t2 = (a + q) * t0, a * t0 + q * t1, a * t1 + q * t2
    return t2


# -- The k = 1 stratum ----------------------------------------------------------------


class _OrderOne:
    """
    The entries of the ``k = 1`` stratum: each fault, each coherence (both
    orientations, as ``2 Re``) and each block, with its weight (its own times
    the no-fault probability of every other location), and the value it is
    compared with when sampled, which needs no MPS: the noiseless value with
    the fault's observable signs, the noiseless value for a block, zero for a
    coherence.
    """

    def __init__(self, compiled: CompiledNoise, leave_one_out: np.ndarray, dressed):
        self.compiled = compiled
        faults = compiled.fault_probs * leave_one_out[compiled.fault_location]
        coherences = (
            2
            * np.abs(compiled.coherence_weights)
            * leave_one_out[compiled.coherence_location]
        )
        blocks = compiled.block_probs * leave_one_out[compiled.block_location]
        sizes = (len(faults), len(coherences), len(blocks))
        self.kind = np.repeat([0, 1, 2], sizes)
        self.index = np.concatenate([np.arange(s) for s in sizes]).astype(int)
        self.weight = np.concatenate([faults, coherences, blocks])
        # Observable sign of each fault on each term; the constant keeps +1.
        flips = compiled.observable_flips([p for p, _ in dressed])
        self.signs = np.ones((len(faults), len(dressed) + 1))
        self.signs[:, :-1] = 1.0 - 2.0 * flips
        self.bound = np.zeros(len(self.weight))
        self.tail = np.zeros(0, int)
        self.tail_mass = 0.0
        self.forward = {}  # coherence -> its job in the (A, B) orientation

    def jobs(self, i: int, jobs: _Jobs):
        """``[(job, coefficient)]`` whose sum is entry ``i``'s value vector."""
        compiled = self.compiled
        kind, k = self.kind[i], self.index[i]
        if kind == 0:
            pauli = compiled.fault_paulis[k]
            return [(jobs.add_diagonal(compiled.flips[k], (pauli,)), 1.0)]
        if kind == 1:
            # Weighted by 2 |chi|: the value is Re(phase T), the mean of the
            # job and its conjugate with phases conjugate to each other.
            phase = compiled.coherence_weights[k] / abs(compiled.coherence_weights[k])
            forward = jobs.add(
                compiled.ket_flips[k],
                compiled.bra_flips[k],
                (compiled.coherence_ket[k],),
                (compiled.coherence_bra[k],),
            )
            backward = jobs.add(
                compiled.bra_flips[k],
                compiled.ket_flips[k],
                (compiled.coherence_bra[k],),
                (compiled.coherence_ket[k],),
            )
            self.forward[int(k)] = forward
            return [(forward, phase / 2), (backward, np.conj(phase) / 2)]
        return [
            (jobs.add(kf, bf, (a,), (b,)), c)
            for a, b, c, kf, bf in zip(
                compiled.block_kets[k],
                compiled.block_bras[k],
                compiled.block_coefficients[k],
                compiled.block_ket_flips[k],
                compiled.block_bra_flips[k],
            )
        ]

    def bases(self, v0: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """The value each entry is compared with, as a scalar."""
        base = np.zeros(len(self.weight))
        faults = self.kind == 0
        base[faults] = np.real((self.signs * v0) @ weights)[self.index[faults]]
        base[self.kind == 2] = float(np.real(v0 @ weights))
        return base

    def split(self, n_draws: int | None, weights: np.ndarray, rng):
        """
        ``(exact entries, sampled draws)``: every entry exact without
        ``n_draws``; otherwise the entries whose weight times bound exceeds
        ``1/n_draws`` of what remains exact, and ``n_draws`` draws from the
        rest (the tail) in proportion to it.
        """
        everything = np.arange(len(self.weight))
        if n_draws is None:
            return everything, np.zeros(0, int)
        compiled = self.compiled
        scale = float(np.abs(weights[:-1]).sum())
        moved = 2 * np.abs(np.sin(compiled.angles / 2))
        faults = self.kind == 0
        shift = np.minimum(2.0, 2.0 * (compiled.flips.astype(float) @ moved))
        self.bound[faults] = scale * shift[self.index[faults]]
        self.bound[self.kind == 1] = scale + abs(weights[-1])
        self.bound[self.kind == 2] = 2.0 * scale
        importance = self.weight * self.bound
        remaining = importance.sum()
        exact = []
        for i in np.argsort(-importance):
            if importance[i] == 0 or importance[i] * n_draws < remaining:
                break
            exact.append(i)
            remaining -= importance[i]
        exact = np.array(exact, dtype=int)
        self.tail = np.setdiff1d(everything, exact)
        self.tail_mass = float(importance[self.tail].sum())
        if self.tail_mass <= 0 or n_draws <= 0:
            return exact, np.zeros(0, int)
        draws = rng.choice(
            self.tail, size=n_draws, p=importance[self.tail] / self.tail_mass
        )
        return exact, draws


# -- Jobs: what the MPS evaluates --------------------------------------------------------


class _Jobs:
    """
    Every expectation value and transition amplitude the estimate needs, each
    as ``(ket pattern, bra pattern, ket Paulis, bra Paulis)`` with the Paulis
    in circuit order, deduplicated, and valued as the vector of
    ``<phi_bra| B^dag O_t A |phi_ket>`` over the observable terms and the
    constant (normalized by the states' norms).
    """

    def __init__(self, compiled: CompiledNoise, dressed):
        self.compiled = compiled
        n = compiled.nqubits
        self.observables = [stim.PauliString(p) * int(s) for p, s in dressed]
        self.observables.append(stim.PauliString(n))
        self.keys: dict = {}
        self.specs: list = []
        self._strings: dict = {}

    def add(self, ket, bra, ket_paulis, bra_paulis) -> int:
        ket, bra = np.asarray(ket, bool), np.asarray(bra, bool)
        key = (ket.tobytes(), bra.tobytes(), tuple(ket_paulis), tuple(bra_paulis))
        if key not in self.keys:
            self.keys[key] = len(self.specs)
            self.specs.append(key)
        return self.keys[key]

    def add_diagonal(self, pattern, paulis) -> int:
        return self.add(pattern, pattern, paulis, paulis)

    def _product(self, paulis) -> stim.PauliString:
        if paulis not in self._strings:
            product = stim.PauliString(self.compiled.nqubits)
            for pauli in paulis:
                product = stim.PauliString(pauli) * product
            self._strings[paulis] = product
        return self._strings[paulis]

    def _labels(self, ket_paulis, bra_paulis):
        """Label and phase of ``B^dag O_t A`` for every term."""
        a = self._product(ket_paulis)
        b_dagger = self._product(bra_paulis).copy()
        b_dagger.sign = complex(b_dagger.sign).conjugate()
        labels, phases = [], []
        for observable in self.observables:
            product = b_dagger * observable * a
            labels.append(str(product).lstrip("+-i").replace("_", "I"))
            phases.append(complex(product.sign))
        return labels, np.array(phases)

    def evaluate(self, n_workers: int = 1):
        """
        ``(values, fidelities, n_transitions)``: ``values[job]`` the vector of
        each job, and the fidelities of the states measured as expectation
        values.
        """
        n_rotations = len(self.compiled.generators)
        values = np.zeros((len(self.specs), len(self.observables)), complex)
        same, different = {}, {}
        for job, (ket, bra, kp, bp) in enumerate(self.specs):
            if ket == bra:
                same.setdefault(ket, []).append(job)
            else:
                # A transition and its conjugate share one evaluation.
                canonical, conjugate = min(
                    ((ket, bra, kp, bp), False), ((bra, ket, bp, kp), True)
                )
                different.setdefault(canonical, []).append((job, conjugate))

        hsmpo = self.compiled.hsmpo
        args = (
            hsmpo.tn_engine,
            hsmpo.initial_mps(),
            list(self.compiled.generators),
            np.asarray(self.compiled.angles),
            hsmpo.max_bond_dimension,
        )

        def pattern(key):
            return np.frombuffer(key, dtype=bool, count=n_rotations)

        # Expectation values: one state per pattern, every label it needs.
        rows, layouts = [], []
        for key in sorted(same):
            needed, layout = {}, []
            for job in same[key]:
                labels, phases = self._labels(*self.specs[job][2:])
                slots = [needed.setdefault(label, len(needed)) for label in labels]
                layout.append((job, slots, phases))
            rows.append((pattern(key), list(needed)))
            layouts.append(layout)
        results = _run(_evaluate_patterns, args, rows, n_workers)
        for (expvals, _), layout in zip(results, layouts):
            for job, slots, phases in layout:
                values[job] = phases * expvals[slots]
        fidelities = np.array([fidelity for _, fidelity in results])

        # Transitions, in lexicographic order of their (ket, bra) patterns.
        canonicals = sorted(different)
        transitions = []
        for ket, bra, kp, bp in canonicals:
            labels, phases = self._labels(kp, bp)
            transitions.append((pattern(ket), pattern(bra), labels, phases))
        amplitudes = _run(_evaluate_transitions, args, transitions, n_workers)
        for canonical, amplitude in zip(canonicals, amplitudes):
            for job, conjugate in different[canonical]:
                values[job] = np.conj(amplitude) if conjugate else amplitude
        return values, fidelities, len(canonicals)


def _run(function, args, items, n_workers):
    """``function(*args, block)`` over contiguous blocks of ``items``, concatenated."""
    if not items:
        return []
    n_workers = max(1, min(int(n_workers), len(items)))
    if n_workers == 1:
        return function(*args, items)
    blocks = np.array_split(np.arange(len(items)), n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = pool.map(
            function, *zip(*[(*args, [items[i] for i in b]) for b in blocks])
        )
        return [r for block in results for r in block]


class _Chain:
    """
    MPS checkpoints along the rotation chain, for sign patterns visited in
    lexicographic order: ``checkpoints[j]`` is the state after the first ``j``
    rotations of the current pattern, so consecutive patterns share their
    common prefix. Each rotation's operator is built once per sign.
    """

    def __init__(
        self, engine, initial_mps, generators, angles, max_bond, operators=None
    ):
        self.engine, self.generators, self.angles = engine, generators, angles
        self.max_bond = max_bond
        self.operators = {} if operators is None else operators
        self.checkpoints = [engine.working_state(initial_mps)]
        self.current = None

    def _rotation(self, j: int, flipped: bool):
        if (j, flipped) not in self.operators:
            angle = -self.angles[j] if flipped else self.angles[j]
            self.operators[j, flipped] = self.engine.rotation_operator(
                self.generators[j], angle
            )
        return self.operators[j, flipped]

    def _evolve(self, state, pattern, start: int):
        for j in range(start, len(self.generators)):
            state = self.engine.copy_state(state)
            self.engine.apply_rotation(
                state, self._rotation(j, bool(pattern[j])), self.max_bond
            )
            self.checkpoints.append(state)
        return state

    def state(self, pattern):
        """The final state of ``pattern``, which becomes the current one."""
        if self.current is not None and np.array_equal(pattern, self.current):
            return self.checkpoints[-1]
        start = 0 if self.current is None else _first_difference(pattern, self.current)
        del self.checkpoints[start + 1 :]
        self.current = np.array(pattern, dtype=bool)
        return self._evolve(self.checkpoints[-1], pattern, start)

    def follow(self, pattern, other: _Chain):
        """
        Like :meth:`state`, but restarting from ``other``'s checkpoints when
        ``pattern`` shares a longer prefix with ``other``'s current pattern
        than with this chain's own.
        """
        n = len(self.generators)

        def shared(current):
            if current is None:
                return -1
            if np.array_equal(pattern, current):
                return n
            return _first_difference(pattern, current)

        own, theirs = shared(self.current), shared(other.current)
        if own == n:
            return self.checkpoints[-1]
        if theirs > own:
            self.checkpoints = other.checkpoints[: theirs + 1]
            start = theirs
        else:
            del self.checkpoints[own + 1 :]
            start = own
        self.current = np.array(pattern, dtype=bool)
        return self._evolve(self.checkpoints[-1], pattern, start)


def _first_difference(a, b) -> int:
    return int(np.argmax(np.asarray(a) != np.asarray(b)))


def _evaluate_patterns(engine, initial_mps, generators, angles, max_bond, rows):
    """
    ``[(expectation values of the labels, fidelity)]`` for each sorted
    ``(pattern, labels)``, along one :class:`_Chain`.
    """
    chain = _Chain(engine, initial_mps, generators, angles, max_bond)
    out = []
    for pattern, labels in rows:
        state = chain.state(pattern)
        expvals = np.real(
            np.asarray(engine.pauli_expvals(state, labels), dtype=complex)
        )
        out.append((expvals, float(np.real(engine.squared_norm(state)))))
    return out


def _evaluate_transitions(engine, initial_mps, generators, angles, max_bond, jobs):
    """
    Phase times ``<phi_bra| P |phi_ket>`` per label of each job, for jobs
    sorted by ket then bra pattern: kets along one :class:`_Chain`, bras
    along another that restarts from whichever of the previous bra and the
    current ket shares the longer prefix.
    """
    ket_chain = _Chain(engine, initial_mps, generators, angles, max_bond)
    bra_chain = _Chain(
        engine, initial_mps, generators, angles, max_bond, operators=ket_chain.operators
    )
    out = []
    for ket, bra, labels, phases in jobs:
        ket_state = ket_chain.state(ket)
        bra_state = bra_chain.follow(bra, ket_chain)
        amplitudes = np.asarray(
            engine.pauli_transitions(bra_state, ket_state, labels), dtype=complex
        )
        out.append(phases * amplitudes)
    return out


# -- Rotation order ---------------------------------------------------------------------


def _sweep_ordered(compiled: CompiledNoise) -> CompiledNoise:
    """
    The same noisy circuit with its rotations reordered for the MPS kernels.

    Every pattern is a product of rotations with fixed signs, so rotations
    that commute can swap places as long as the columns of every flip matrix
    move with them. Each run of consecutive, pairwise commuting rotations,
    such as a layer of couplings, is put in snake order: its one-site
    rotations first, which leave the orthogonality center alone, then the
    wider ones by position, starting from the end nearer to where the
    previous run left the center. The center then steps from one window to
    the next instead of crossing the chain at every layer.
    """
    generators = compiled.generators
    if len(generators) < 2:
        return compiled
    anticommuting = anticommutation(generators, generators, compiled.nqubits)
    windows = []
    for generator in generators:
        support = [q for q, p in enumerate(generator) if p != "I"]
        windows.append((support[0], support[-1]) if support else (0, 0))

    order, run, center = [], [], 0.0

    def flush():
        nonlocal center
        single = [j for j in run if windows[j][0] == windows[j][1]]
        wide = sorted(
            (j for j in run if windows[j][0] < windows[j][1]),
            key=lambda j: windows[j],
        )
        if wide:
            low, high = windows[wide[0]][0], windows[wide[-1]][1]
            if abs(center - high) < abs(center - low):
                wide.reverse()
            center = sum(windows[wide[-1]]) / 2
        order.extend(single + wide)

    for j in range(len(generators)):
        if run and anticommuting[j, run].any():
            flush()
            run = []
        run.append(j)
    flush()

    order = np.array(order)
    if np.array_equal(order, np.arange(len(order))):
        return compiled
    return compiled.reordered(order)


# -- Sampling k >= 2 ------------------------------------------------------------------


@dataclass
class _Samples:
    """
    Configurations sampled from the ``k >= 2`` stratum: per sample, its
    actions in location order, each ``(kind, index, swapped)`` with kind 0 a
    fault, 1 a coherence (``swapped`` exchanging ket and bra) and 2 a block,
    and the product of the unit phases of its coherences.
    """

    actions: list[list[tuple[int, int, bool]]]
    phase: np.ndarray


def _sample_two_or_more(compiled, tails, n_samples, rng) -> _Samples:
    """
    Sample configurations conditioned on at least two locations doing
    something.

    Locations are visited in order; each one acts with its weight
    conditioned on the actions still needed being reachable from the
    locations left, so every sample is accepted. An acting location picks
    one of its faults or blocks by probability or one of its coherences by
    ``2 |chi|``, in either orientation with equal odds.
    """
    n_locations = len(compiled.location_probs)

    def starts(locations):
        return np.searchsorted(locations, np.arange(n_locations + 1))

    fault_starts = starts(compiled.fault_location)
    coherence_starts = starts(compiled.coherence_location)
    block_starts = starts(compiled.block_location)
    coherence_mass = 2 * np.abs(compiled.coherence_weights)
    unit_phase = np.divide(
        compiled.coherence_weights,
        np.abs(compiled.coherence_weights),
        out=np.ones_like(compiled.coherence_weights),
        where=coherence_mass > 0,
    )
    mass = compiled.location_probs + compiled.coherence_mass

    actions = [[] for _ in range(n_samples)]
    phase = np.ones(n_samples, dtype=complex)
    counts = np.zeros(n_samples, dtype=int)

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
        blocks = np.arange(block_starts[l], block_starts[l + 1])
        options = np.concatenate(
            [
                compiled.fault_probs[faults],
                coherence_mass[coherences],
                compiled.block_probs[blocks],
            ]
        )
        kinds = np.repeat([0, 1, 2], [len(faults), len(coherences), len(blocks)])
        indices = np.concatenate([faults, coherences, blocks]).astype(int)
        cumulative = np.cumsum(options)
        choice = np.searchsorted(
            cumulative, rng.random(hits.size) * cumulative[-1], side="right"
        )
        choice = np.minimum(choice, len(options) - 1)
        swapped = rng.random(hits.size) < 0.5
        for s, c, swap in zip(hits.tolist(), choice.tolist(), swapped.tolist()):
            kind, index = int(kinds[c]), int(indices[c])
            actions[s].append((kind, index, swap if kind == 1 else False))
            if kind == 1:
                phase[s] *= np.conj(unit_phase[index]) if swap else unit_phase[index]
        counts[hits] += 1
    return _Samples(actions=actions, phase=phase)


def _expand(compiled: CompiledNoise, actions, jobs: _Jobs, rng, max_terms: int):
    """
    ``[(job, coefficient)]`` whose sum is a configuration's value vector, or
    an unbiased estimate of it.

    Each block multiplies the number of terms by its own (8 for a reset), so
    blocks are expanded exactly only while the expansion stays within
    ``max_terms``; a block beyond that contributes one of its terms, drawn in
    proportion to ``|chi|`` and reweighted by their sum.
    """
    n_rotations = len(compiled.generators)
    partial = [(np.zeros(n_rotations, bool), np.zeros(n_rotations, bool), (), (), 1.0)]
    for kind, index, swapped in actions:
        if kind == 0:
            flips, pauli = compiled.flips[index], compiled.fault_paulis[index]
            options = [(flips, flips, pauli, pauli, 1.0)]
        elif kind == 1:
            ket = (compiled.ket_flips[index], compiled.coherence_ket[index])
            bra = (compiled.bra_flips[index], compiled.coherence_bra[index])
            if swapped:
                ket, bra = bra, ket
            options = [(ket[0], bra[0], ket[1], bra[1], 1.0)]
        else:
            options = list(
                zip(
                    compiled.block_ket_flips[index],
                    compiled.block_bra_flips[index],
                    compiled.block_kets[index],
                    compiled.block_bras[index],
                    compiled.block_coefficients[index],
                )
            )
            if len(partial) * len(options) > max_terms:
                sizes = np.abs(compiled.block_coefficients[index])
                pick = rng.choice(len(options), p=sizes / sizes.sum())
                kf, bf, ka, ba, w = options[pick]
                options = [(kf, bf, ka, ba, w / sizes[pick] * sizes.sum())]
        partial = [
            (k ^ kf, b ^ bf, kp + (ka,), bp + (ba,), c * w)
            for k, b, kp, bp, c in partial
            for kf, bf, ka, ba, w in options
        ]
    return [(jobs.add(k, b, kp, bp), c) for k, b, kp, bp, c in partial]


# -- Control variates -----------------------------------------------------------------


class _Models:
    """
    Two models of a configuration's value vector built from ``k = 1`` values,
    each with its exact sum over the ``k >= 2`` stratum.

    - Additive, on faults only: ``prod_i s_i (v0 + sum_i (g_i - v0))`` per
      term, ``s_i`` a fault's observable sign and ``g_i`` its unsigned value;
      zero on a configuration with a coherence or a block.
    - Multiplicative: ``v0 prod_a r_a`` per term over the configuration's
      actions, ``r_a`` the action's own ``k = 1`` value over the noiseless
      one (conjugated for a swapped coherence). Its sum factorizes over
      locations. It is exact when the actions act independently, as Pauli
      faults that only flip the observable's sign do, at any number of them.
      Terms whose noiseless value is below ``_PRODUCT_FLOOR`` are left out.

    Entries of ``k = 1`` that were not evaluated (a sampled tail) are taken to
    change nothing beyond their observable sign: a fixed model, so the
    estimate stays unbiased.
    """

    def __init__(self, compiled, items: _OrderOne, vectors, values, v0, null, weights):
        self.compiled, self.null, self.weights, self.v0 = compiled, null, weights, v0
        self.signs = items.signs
        self.active = np.abs(v0) >= _PRODUCT_FLOOR
        safe = np.where(self.active, v0, 1.0)

        n_terms = len(v0)
        self.fault_values = items.signs * v0
        coherence_values = np.zeros((len(compiled.coherence_weights), n_terms), complex)
        block_values = np.tile(v0, (len(compiled.block_probs), 1)).astype(complex)
        for i, vector in vectors.items():
            kind, k = items.kind[i], items.index[i]
            if kind == 0:
                self.fault_values[k] = np.real(vector)
            elif kind == 1:
                coherence_values[k] = values[items.forward[int(k)]]
            else:
                block_values[k] = vector
        self.fault_ratio = np.where(self.active, self.fault_values / safe, 0.0)
        self.coherence_ratio = np.where(self.active, coherence_values / safe, 0.0)
        self.block_ratio = np.where(self.active, block_values / safe, 0.0)

    def sample(self, actions, phase) -> np.ndarray:
        """``(additive, multiplicative)`` model values of one configuration."""
        additive = 0.0
        if all(kind == 0 for kind, _, _ in actions):
            members = [index for _, index, _ in actions]
            sign = np.prod(self.signs[members], axis=0)
            unsigned = self.fault_values[members] * self.signs[members]
            vector = sign * (self.v0 + (unsigned - self.v0).sum(axis=0))
            additive = float(np.real(vector @ self.weights))
        ratio = np.ones(len(self.v0), complex)
        for kind, index, swapped in actions:
            if kind == 0:
                ratio = ratio * self.fault_ratio[index]
            elif kind == 1:
                r = self.coherence_ratio[index]
                ratio = ratio * (np.conj(r) if swapped else r)
            else:
                ratio = ratio * self.block_ratio[index]
        vector = np.where(self.active, self.v0 * ratio, 0.0)
        return np.array([additive, float(np.real(phase * (vector @ self.weights)))])

    def stratum_sums(self) -> np.ndarray:
        """Exact sums of the two models over the ``k >= 2`` configurations."""
        compiled, null = self.compiled, self.null
        n_locations, n_terms = len(null), len(self.v0)
        probs, location = compiled.fault_probs, compiled.fault_location

        # Additive: locations are independent, so with
        # A_l = null_l + sum_{i at l} p_i s_i the sign sums factorize.
        deltas = self.fault_values * self.signs - self.v0
        null_loo = _leave_one_out_products(null)
        no_fault = float(np.prod(null))
        additive = np.zeros(n_terms)
        for t in range(n_terms):
            signed = probs * self.signs[:, t]
            active = np.bincount(location, weights=signed, minlength=n_locations)
            two = float(np.prod(null + active)) - no_fault - active @ null_loo
            active_loo = _leave_one_out_products(null + active)
            member = signed * (active_loo - null_loo)[location]
            additive[t] = self.v0[t] * two + deltas[:, t] @ member

        # Multiplicative: each location contributes null + sum of weight x ratio.
        active = np.zeros((n_locations, n_terms), complex)
        np.add.at(active, location, probs[:, None] * self.fault_ratio)
        np.add.at(
            active,
            compiled.coherence_location,
            2 * np.real(compiled.coherence_weights[:, None] * self.coherence_ratio),
        )
        np.add.at(
            active,
            compiled.block_location,
            compiled.block_probs[:, None] * self.block_ratio,
        )
        multiplicative = np.where(self.active, self.v0 * _two_or_more(active, null), 0)
        return np.array(
            [
                float(np.real(additive @ self.weights)),
                float(np.real(multiplicative @ self.weights)),
            ]
        )


def _regress(values: np.ndarray, controls: np.ndarray, expected: np.ndarray):
    """
    Mean of ``values`` corrected by controls of known means ``expected``:
    ``mean(values - controls @ beta) + expected @ beta``, with ``beta`` the
    least-squares fit. Returns ``(estimate, residual variance, beta)``.
    """
    n = len(values)
    usable = controls.std(axis=0) > 0
    beta = np.zeros(controls.shape[1])
    if usable.any() and n > usable.sum() + 2:
        centered = controls[:, usable] - controls[:, usable].mean(axis=0)
        beta[usable] = np.linalg.lstsq(centered, values - values.mean(), rcond=None)[0]
    residual = values - controls @ beta
    dof = max(n - 1 - int(usable.sum()), 1)
    variance = float(((residual - residual.mean()) ** 2).sum() / dof)
    return float(residual.mean() + expected @ beta), variance, beta
