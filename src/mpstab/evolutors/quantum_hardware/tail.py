"""
Tail-folded MPO -> Pauli terms, the classical half of the head/tail split.

Everything here is what qibo has no notion of: an MPO's exact Pauli-basis
expansion (:func:`mpo_to_pauli_mps`), perfect sampling from that expansion
(:func:`sample_pauli_strings`), and the exact single-snapshot classical-shadow
variance of an MPO (:func:`shadow_variance_from_mpo`). Once a Pauli string has
been sampled, everything downstream (grouping into measurement settings,
building circuits, reading back qibo samples) is plain qibo and lives in
:mod:`~mpstab.evolutors.quantum_hardware.plan` and
:mod:`~mpstab.evolutors.quantum_hardware.estimate`.

Conventions match the rest of ``mpstab``: Pauli strings are qubit-0-leftmost.
MPO site tensors carry upper (output) index ``k{i}`` and lower (input) index
``b{i}``; the recurring contraction is ``Tr[sigma O] = sum_{k,b} O[k,b]
sigma[b,k]``, hence ``"kb,bk"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

#: Pauli labels in the order used as the physical index of the recontracted MPS.
PAULI_LABELS = "IXYZ"

PAULIS = np.array(
    [
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ],
    dtype=complex,
)


def hamming_weight(pauli: str) -> int:
    """Number of non-identity sites in ``pauli``."""
    return sum(label != "I" for label in pauli)


def mpo_site_arrays(mpo) -> list:
    """
    MPO site tensors as arrays ordered ``(left, right, k, b)``.

    Boundary bonds are inserted as singleton axes so every site has rank 4 and
    every consumer's per-site loop is uniform.
    """
    import quimb.tensor as qtn

    arrays = []
    for site in range(mpo.L):
        tensor = mpo[site]
        left = list(qtn.bonds(mpo[site - 1], tensor)) if site > 0 else []
        right = list(qtn.bonds(tensor, mpo[site + 1])) if site < mpo.L - 1 else []
        order = left + right + [mpo.upper_ind(site), mpo.lower_ind(site)]
        array = np.asarray(tensor.transpose(*order).data, dtype=complex)
        if not left:
            array = array[None, ...]
        if not right:
            array = array[:, None, ...]
        arrays.append(array)
    return arrays


def mpo_to_pauli_mps(mpo) -> list:
    """
    Recontract an MPO into the Pauli-coefficient basis, exactly.

    Each site tensor ``A[l, r, k, b]`` becomes ``B[i, l, r]`` with physical
    dimension 4 (one axis per label of :data:`PAULI_LABELS`) and the same bond
    dimension, so contracting the train at a fixed label sequence ``P`` gives
    ``c_P = Tr[O P] / 2**n`` exactly, only a change of basis on the physical
    index. This is what makes :func:`sample_pauli_strings` an exact perfect
    sampler and :func:`shadow_variance_from_mpo` exact and enumeration-free.
    """
    return [
        np.einsum("lrkb,ibk->ilr", array, PAULIS, optimize=True) / 2
        for array in mpo_site_arrays(mpo)
    ]


def shadow_variance_from_mpo(mpo) -> float:
    """
    Exact single-snapshot classical-shadow variance of ``mpo``, ``sum_P |c_P|**2
    * 3**w_P``, computed directly from the tensor network with no Pauli
    enumeration: ``3**w_P`` factorizes site by site, so contracting the
    Pauli-basis MPS against itself with ``diag(1, 3, 3, 3)`` inserted at every
    site sums over all ``4**n`` Pauli strings at once, at ``O(n chi**3)``. This
    is the exact predictor :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.expectation_at_cut`
    uses to size the ``"shadows"`` shot budget from a target ``epsilon``.
    """
    weight_table = np.array([1.0, 3.0, 3.0, 3.0])
    env = np.ones((1, 1), dtype=complex)
    for tensor in mpo_to_pauli_mps(mpo):
        env = np.einsum(
            "lm,ilr,i,ims->rs", env, tensor, weight_table, tensor.conj(), optimize=True
        )
    return float(env[0, 0].real)


@dataclass(frozen=True)
class PauliEnsemble:
    """
    A retained subset of an operator's Pauli expansion, coefficients included.

    ``retained_weight`` is ``sum(|c_P|**2 for P in strings) / total_weight``,
    the L2 fraction of the operator's Frobenius weight the retained strings
    capture.
    """

    strings: tuple
    coefficients: np.ndarray
    retained_weight: float
    total_weight: float


def _pauli_string_coefficient(tensors, labels) -> complex:
    product_ = tensors[0][labels[0]]
    for site in range(1, len(tensors)):
        product_ = product_ @ tensors[site][labels[site]]
    return complex(product_[0, 0])


def enumerate_pauli_coefficients(mpo, top_k=None):
    """
    Every ``(pauli_string, c_P)`` by brute-force enumeration of all ``4**n``
    labels, sorted by ``|c_P|**2`` descending. Exponential in ``n``: exists to
    validate :func:`sample_pauli_strings` / :func:`top_k_pauli_strings` on
    small systems, not for production use.
    """
    tensors = mpo_to_pauli_mps(mpo)
    n = len(tensors)
    results = [
        (
            "".join(PAULI_LABELS[label] for label in labels),
            _pauli_string_coefficient(tensors, labels),
        )
        for labels in product(range(4), repeat=n)
    ]
    results.sort(key=lambda item: -abs(item[1]) ** 2)
    return results[:top_k] if top_k is not None else results


def _right_environments(tensors) -> list:
    n = len(tensors)
    envs = [None] * (n + 1)
    envs[n] = np.ones((1, 1), dtype=complex)
    for site in range(n - 1, -1, -1):
        envs[site] = np.einsum(
            "ilr,rs,ims->lm",
            tensors[site],
            envs[site + 1],
            tensors[site].conj(),
            optimize=True,
        )
    return envs


def _draw_one_string(tensors, envs, rng) -> str:
    """One left-to-right perfect-sampling sweep, cost ``O(n chi**2)``: the
    single-sample reference :func:`_draw_many_strings` batches and generalizes."""
    left = np.ones((1, 1), dtype=complex)
    labels = []
    for site, tensor in enumerate(tensors):
        weights = np.einsum(
            "lm,ilr,ims,rs->i",
            left,
            tensor,
            tensor.conj(),
            envs[site + 1],
            optimize=True,
        ).real
        weights = np.clip(weights, 0.0, None)
        probabilities = weights / weights.sum()
        label = int(rng.choice(4, p=probabilities))
        labels.append(label)
        left = (
            np.einsum("lm,lr,ms->rs", left, tensor[label], tensor[label].conj())
            / weights[label]
        )
    return "".join(PAULI_LABELS[label] for label in labels)


def _draw_many_strings(tensors, envs, n_samples: int, rng) -> np.ndarray:
    """``n_samples`` independent left-to-right perfect-sampling sweeps, batched
    over the sample axis. Cost ``O(n_samples * n * chi**2)``."""
    n = len(tensors)
    left = np.ones((n_samples, 1, 1), dtype=complex)
    labels = np.empty((n_samples, n), dtype=np.int64)
    sample_range = np.arange(n_samples)
    for site, tensor in enumerate(tensors):
        weights = np.einsum(
            "klm,ilr,ims,rs->ki",
            left,
            tensor,
            tensor.conj(),
            envs[site + 1],
            optimize=True,
        ).real
        weights = np.clip(weights, 0.0, None)
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probabilities, axis=1)
        cumulative[:, -1] = 1.0
        draws = rng.random((n_samples, 1))
        chosen = (draws < cumulative).argmax(axis=1)
        labels[:, site] = chosen

        chosen_tensor = tensor[chosen]
        chosen_weight = weights[sample_range, chosen]
        left = (
            np.einsum(
                "klm,klr,kms->krs",
                left,
                chosen_tensor,
                chosen_tensor.conj(),
                optimize=True,
            )
            / chosen_weight[:, None, None]
        )
    return labels


def sample_pauli_strings(mpo, n_samples: int, seed=None) -> PauliEnsemble:
    """
    Perfect-sample Pauli strings from ``mpo`` with probability ``|c_P|**2 /
    sum|c|**2``, deduplicated. Cost ``O(n_samples * n * chi**2)``; exact (no
    MCMC burn-in), since the environments marginalize over the unfixed suffix
    regardless of which prefix ends up sampled.
    """
    tensors = mpo_to_pauli_mps(mpo)
    envs = _right_environments(tensors)
    total_weight = float(envs[0][0, 0].real)

    rng = np.random.default_rng(seed)
    labels = _draw_many_strings(tensors, envs, n_samples, rng)
    drawn = {"".join(PAULI_LABELS[label] for label in row) for row in labels}

    strings = tuple(sorted(drawn))
    coefficients = np.array(
        [
            _pauli_string_coefficient(
                tensors, [PAULI_LABELS.index(label) for label in string]
            )
            for string in strings
        ],
        dtype=complex,
    )
    retained_weight = (
        float(np.sum(np.abs(coefficients) ** 2)) / total_weight if total_weight else 0.0
    )
    return PauliEnsemble(
        strings=strings,
        coefficients=coefficients,
        retained_weight=retained_weight,
        total_weight=total_weight,
    )


def top_k_pauli_strings(mpo, k: int) -> PauliEnsemble:
    """Deterministic top-``k`` strings by ``|c_P|**2``, via full ``4**n``
    enumeration; for validating :func:`sample_pauli_strings` on small systems."""
    ranked = enumerate_pauli_coefficients(mpo)
    total_weight = float(sum(abs(coefficient) ** 2 for _, coefficient in ranked))
    top = ranked[:k]
    strings = tuple(string for string, _ in top)
    coefficients = np.array([coefficient for _, coefficient in top], dtype=complex)
    retained_weight = (
        float(np.sum(np.abs(coefficients) ** 2)) / total_weight if total_weight else 0.0
    )
    return PauliEnsemble(
        strings=strings,
        coefficients=coefficients,
        retained_weight=retained_weight,
        total_weight=total_weight,
    )


def truncation_error_estimate(ensemble: PauliEnsemble):
    """
    Estimate the systematic error from the Pauli strings ``ensemble`` discarded.

    ``|<Delta>| <= sum_{P not in ensemble} |c_P|`` (L1, rigorous) and
    ``sqrt(sum_{P not in ensemble} |c_P|**2)`` (L2, typical-case) bound the
    discarded operator's contribution. The L2 mass is exact from what
    :class:`PauliEnsemble` already carries; the L1 mass needs ``sum |c_P|``
    over a discarded set that is, in general, exponentially large and never
    enumerated, so it is estimated by extrapolating the sampled tail: the
    smallest retained ``|c_P|`` sets the coefficient scale at the sampling
    threshold, and the exact discarded L2 mass divided by that scale gives an
    effective discarded-term count (Cauchy-Schwarz saturated by equal-magnitude
    terms), whose L1 mass is then ``discarded_l2_mass / c_min``.

    Returns:
        ``(l1_estimate, l2_exact)``.
    """
    discarded_mass = max(0.0, ensemble.total_weight * (1.0 - ensemble.retained_weight))
    l2 = float(np.sqrt(discarded_mass))
    if ensemble.coefficients.size == 0 or discarded_mass == 0.0:
        return l2, l2
    c_min = float(np.min(np.abs(ensemble.coefficients)))
    l1 = discarded_mass / c_min if c_min > 0 else l2
    return l1, l2


def pool_pauli_terms(terms) -> dict:
    """
    Pool several Hamiltonian terms' sampled ensembles into one shared
    Pauli-coefficient map, since the head state does not depend on the
    observable and every term's sampled strings can be measured from the same
    shots once pooled.

    Args:
        terms: iterable of ``(term_coefficient, PauliEnsemble)`` pairs.

    Returns:
        ``pauli_string -> combined coefficient``.
    """
    pooled: dict = {}
    for term_coefficient, ensemble in terms:
        for string, coefficient in zip(ensemble.strings, ensemble.coefficients):
            pooled[string] = pooled.get(string, 0.0) + term_coefficient * coefficient
    return pooled


def fold_pool_through_tableau(pool: dict, tail_tableau, stab_engine) -> dict:
    """
    Conjugate every pooled ``{pauli: coefficient}`` entry through ``tail_tableau``,
    accumulating the fold sign into the coefficient.

    A device running a resynthesized head prepares ``U(head)|psi_0>``, not the
    exact head state the tail-folded Pauli coefficients were derived against;
    the two differ by the resynthesis Clifford residual ``U(tail)``, so every
    sampled string must be conjugated through it *before* QWC grouping
    (:mod:`~mpstab.evolutors.quantum_hardware.plan`): conjugation does not
    preserve qubit-wise commutativity, so folding after grouping would
    silently break the measurement settings. Folded strings that collide are
    summed, exactly as :func:`pool_pauli_terms` already does pre-fold.

    Args:
        pool: ``pauli -> coefficient`` (already pooled across Hamiltonian terms).
        tail_tableau: the resynthesis Clifford residual, a ``stim.Tableau``.
        stab_engine: a :class:`~mpstab.engines.StimEngine`.
    """
    folded: dict = {}
    for label, coefficient in pool.items():
        folded_label, sign = stab_engine.fold_pauli_through_tableau(
            label, tail_tableau, 1.0
        )
        folded[folded_label] = folded.get(folded_label, 0.0) + coefficient * sign
    return folded
