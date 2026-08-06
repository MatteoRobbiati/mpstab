"""
Pauli-basis expansion of an MPO: exact coefficients, sampling and variances.

The tail of a head/tail split is a Pauli rotation chain folded into the
observable, which leaves an MPO. To measure that observable on a device it has
to be turned back into Pauli strings with coefficients. This module does that
change of basis and everything that follows directly from it.

MPO site tensors carry an upper (output) index ``k{i}`` and a lower (input)
index ``b{i}``. The recurring contraction is ``Tr[sigma O] = sum_{k,b} O[k,b]
sigma[b,k]``, hence the ``"kb,bk"`` index patterns below.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from mpstab.pauli import PAULI_ARRAY, PAULI_LABELS


def mpo_site_arrays(mpo) -> list:
    """
    MPO site tensors as arrays ordered ``(left, right, k, b)``.

    Boundary bonds become singleton axes so every site has rank 4 and callers
    need no special case for the first and last site.
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
    Rewrite an MPO in the Pauli basis, exactly.

    Each site tensor ``A[l, r, k, b]`` becomes ``B[i, l, r]`` with physical
    dimension 4, one entry per label of :data:`~mpstab.pauli.PAULI_LABELS`, and
    unchanged bond dimension. Contracting the resulting train at a fixed label
    sequence ``P`` gives ``c_P = Tr[O P] / 2**n``. Being only a change of basis
    on the physical index, it makes :func:`sample_pauli_strings` exact and
    :func:`shadow_variance_from_mpo` enumeration-free.
    """
    return [
        np.einsum("lrkb,ibk->ilr", array, PAULI_ARRAY, optimize=True) / 2
        for array in mpo_site_arrays(mpo)
    ]


def shadow_variance_from_mpo(mpo) -> float:
    """
    Exact single-snapshot classical-shadow variance of ``mpo``,
    ``sum_P |c_P|**2 * 3**w_P``.

    The weight factor ``3**w_P`` factorises site by site, so inserting
    ``diag(1, 3, 3, 3)`` at every site of the Pauli-basis MPS contracted against
    itself sums over all ``4**n`` strings at once, in ``O(n chi**3)``.
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

    Attributes:
        strings: the retained Pauli strings.
        coefficients: their ``c_P``, in the same order.
        retained_weight: fraction of the operator's Frobenius weight the
            retained strings capture, ``sum |c_P|**2 / total_weight``.
        total_weight: ``sum_P |c_P|**2`` over the whole expansion.
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


def _retained_fraction(coefficients: np.ndarray, total_weight: float) -> float:
    if not total_weight:
        return 0.0
    return float(np.sum(np.abs(coefficients) ** 2)) / total_weight


def _ensemble(tensors, strings: tuple, total_weight: float) -> PauliEnsemble:
    """Look up the exact coefficient of each string in ``strings``."""
    coefficients = np.array(
        [
            _pauli_string_coefficient(
                tensors, [PAULI_LABELS.index(label) for label in string]
            )
            for string in strings
        ],
        dtype=complex,
    )
    return PauliEnsemble(
        strings=strings,
        coefficients=coefficients,
        retained_weight=_retained_fraction(coefficients, total_weight),
        total_weight=total_weight,
    )


def enumerate_pauli_coefficients(mpo, top_k=None):
    """
    Every ``(pauli_string, c_P)``, sorted by ``|c_P|**2`` descending.

    Enumerates all ``4**n`` label sequences, so it is exponential in ``n``: use
    it to validate the samplers on small systems, not in production.
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
    """One left-to-right perfect-sampling sweep, cost ``O(n chi**2)``."""
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
    """:func:`_draw_one_string` batched over a sample axis, ``O(n_samples n chi**2)``."""
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
    Sample Pauli strings from ``mpo`` with probability ``|c_P|**2 / sum |c|**2``,
    deduplicated.

    Exact and free of MCMC burn-in: the right environments marginalise over the
    unfixed suffix whatever prefix has been drawn, so every sweep draws from the
    true distribution. Cost ``O(n_samples * n * chi**2)``.
    """
    tensors = mpo_to_pauli_mps(mpo)
    envs = _right_environments(tensors)
    total_weight = float(envs[0][0, 0].real)

    rng = np.random.default_rng(seed)
    labels = _draw_many_strings(tensors, envs, n_samples, rng)
    drawn = {"".join(PAULI_LABELS[label] for label in row) for row in labels}

    return _ensemble(tensors, tuple(sorted(drawn)), total_weight)


def top_k_pauli_strings(mpo, k: int) -> PauliEnsemble:
    """
    Deterministic top-``k`` strings by ``|c_P|**2``.

    Goes through the full ``4**n`` enumeration, so it exists to validate
    :func:`sample_pauli_strings` on small systems.
    """
    ranked = enumerate_pauli_coefficients(mpo)
    total_weight = float(sum(abs(coefficient) ** 2 for _, coefficient in ranked))
    coefficients = np.array([c for _, c in ranked[:k]], dtype=complex)
    return PauliEnsemble(
        strings=tuple(string for string, _ in ranked[:k]),
        coefficients=coefficients,
        retained_weight=_retained_fraction(coefficients, total_weight),
        total_weight=total_weight,
    )


def truncation_error_estimate(ensemble: PauliEnsemble):
    """
    Bound the systematic error from the Pauli strings ``ensemble`` left out.

    The discarded operator contributes at most ``sum |c_P|`` (L1, rigorous) and
    typically ``sqrt(sum |c_P|**2)`` (L2) over the discarded set. The L2 mass is
    exact from what :class:`PauliEnsemble` already carries. The L1 mass would
    need a sum over a discarded set that is in general exponentially large and
    never enumerated, so it is extrapolated from the sampled tail: the smallest
    retained ``|c_P|`` sets the coefficient scale at the sampling threshold, and
    dividing the exact discarded L2 mass by that scale gives an effective
    discarded-term count whose L1 mass is ``discarded_mass / c_min``.

    Returns:
        ``(l1_estimate, l2_exact)``.
    """
    discarded_mass = max(0.0, ensemble.total_weight * (1.0 - ensemble.retained_weight))
    l2 = float(np.sqrt(discarded_mass))
    if ensemble.coefficients.size == 0 or discarded_mass == 0.0:
        return l2, l2
    c_min = float(np.min(np.abs(ensemble.coefficients)))
    return (discarded_mass / c_min if c_min > 0 else l2), l2


def pool_pauli_terms(terms) -> dict:
    """
    Merge several Hamiltonian terms' sampled ensembles into one coefficient map.

    The head state does not depend on the observable, so once pooled every
    term's strings can be measured from the same shots.

    Args:
        terms: iterable of ``(term_coefficient, PauliEnsemble)`` pairs.

    Returns:
        ``{pauli_string: combined coefficient}``.
    """
    pooled: dict = {}
    for term_coefficient, ensemble in terms:
        for string, coefficient in zip(ensemble.strings, ensemble.coefficients):
            pooled[string] = pooled.get(string, 0.0) + term_coefficient * coefficient
    return pooled


def fold_pool_through_tableau(pool: dict, tail_tableau, stab_engine) -> dict:
    """
    Conjugate every pooled ``{pauli: coefficient}`` entry through ``tail_tableau``.

    A device running a resynthesised head prepares ``U(head)|psi_0>``, not the
    exact head state the tail-folded coefficients were derived against; the two
    differ by the resynthesis Clifford residual. Every sampled string therefore
    has to be conjugated through that residual *before* QWC grouping, since
    conjugation does not preserve qubit-wise commutativity and folding afterwards
    would silently invalidate the measurement settings. Strings that collide
    after folding are summed.

    Args:
        pool: ``{pauli: coefficient}``, already pooled across Hamiltonian terms.
        tail_tableau: the Clifford residual, a ``stim.Tableau``.
        stab_engine: a :class:`~mpstab.engines.StimEngine`.
    """
    folded: dict = {}
    for label, coefficient in pool.items():
        folded_label, sign = stab_engine.fold_pauli_through_tableau(
            label, tail_tableau, 1.0
        )
        folded[folded_label] = folded.get(folded_label, 0.0) + coefficient * sign
    return folded
