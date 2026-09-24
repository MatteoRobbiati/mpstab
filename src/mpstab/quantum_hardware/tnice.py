"""
TN-ICE: an MPS-optimized dual estimator for the ``"shadows"`` route's data.

Mangini & Cavalcanti, *Low variance estimations of many observables with
tensor networks and informationally-complete measurements* (Quantum 2025,
arXiv:2407.02923), replace the fixed *canonical-dual* post-processing of
classical shadows with a second MPS ``omega`` of reconstruction coefficients,
optimized to minimize an empirical-variance-plus-reconstruction-penalty cost
(their Eq. 16-20) against the very data classical shadows would otherwise
contract with the fixed canonical dual. This module implements that fit for
the measurement primitive already used by
:func:`~mpstab.quantum_hardware.plan.build_shadow_plan` -- random local
Pauli-basis measurements -- which is exactly the six-effect-per-qubit OC-POVM
of the paper's Eq. (24)/(32):

``Pi_k in {|0><0|/3, |1><1|/3, |+><+|/3, |-><-|/3, |+i><+i|/3, |-i><-i|/3}``

indexed here as ``k = 0..5`` in that same order (``outcome_index`` below).

Every object lives in the *plain Pauli-coefficient* convention already used by
:func:`~mpstab.quantum_hardware.pauli_expansion.mpo_to_pauli_mps`: an operator
``O`` is represented by the MPS of its exact expansion coefficients in
``O = sum_P c_P P`` (:data:`~mpstab.pauli.PAULI_LABELS` order, ``I, X, Y, Z``
per site). Because ``Pi_k = sum_p E[k, p] P_p`` for a *fixed*, site-independent
``(6, 4)`` matrix ``E`` (:data:`LOCAL_EFFECT_MAP`), the reconstruction map
``Pi`` of the paper's Eq. (12a) is a purely local (bond-dimension-1) map
applied site-by-site -- it never changes anyone's bond dimension. The same is
true of the canonical-dual seed (:data:`LOCAL_CANONICAL_DUAL_MAP`, from
Eq. 38/41/44): applying it to ``O``'s own Pauli-coefficient MPS gives the
*exact* canonical-dual reconstruction coefficients at zero cost and at ``O``'s
own bond dimension -- which is also, by construction, this module's optimizer
seed and the "zero sweeps" baseline that reproduces
:func:`~mpstab.quantum_hardware.estimate.estimate_shadows`'s value.

:func:`fit_omega_mps` then runs a one-site DMRG-like sweep, solving the exact
local quadratic problem of Eq. (19)-(20) at each site from environment
tensors accumulated over the (training) measured shots, alternating with a
left-environment update -- the same left-to-right batched-environment pattern
:func:`~mpstab.quantum_hardware.pauli_expansion._draw_many_strings` already
uses for Pauli-string sampling.
"""

from __future__ import annotations

import numpy as np

#: outcome index 0..5 -> (basis label, bit), Eq. (24)/(32) of the paper.
_BASIS_TO_OUTCOME_PAIR = {"Z": (0, 1), "X": (2, 3), "Y": (4, 5)}


def outcome_index(basis: str, bit: int) -> int:
    """The outcome index ``0..5`` for measuring ``basis`` and getting ``bit``."""
    return _BASIS_TO_OUTCOME_PAIR[basis][bit]


def _local_effect_map() -> np.ndarray:
    """
    ``(6, 4)`` real matrix ``E`` with ``Pi_k = sum_p E[k, p] P_p``, the six
    sub-normalised Pauli-eigenstate POVM effects of Eq. (24), expanded in the
    ``(I, X, Y, Z)`` basis. Every row is ``Pi_k = (I +- P_axis) / 6``.
    """
    axis_column = {"X": 1, "Y": 2, "Z": 3}
    effect_map = np.zeros((6, 4))
    effect_map[:, 0] = 1 / 6
    for row, (axis, sign) in enumerate(
        [("Z", 1), ("Z", -1), ("X", 1), ("X", -1), ("Y", 1), ("Y", -1)]
    ):
        effect_map[row, axis_column[axis]] = sign / 6
    return effect_map


#: ``(6, 4)``: ``Pi_k = sum_p LOCAL_EFFECT_MAP[k, p] P_p`` (Eq. 24/32).
LOCAL_EFFECT_MAP = _local_effect_map()

#: ``(6, 4)``: the canonical dual ``D_k = 9 Pi_k - I`` (Eq. 38/41), scaled so
#: that applying it site-wise to a Pauli-coefficient MPS gives the exact
#: canonical-dual reconstruction-coefficient MPS in this module's convention
#: -- see :func:`canonical_omega_seed`.
LOCAL_CANONICAL_DUAL_MAP = 6 * LOCAL_EFFECT_MAP @ np.diag([1.0, 3.0, 3.0, 3.0])

#: ``(6, 6)``: the local physical-index Gram matrix ``E @ E.T`` of the effect
#: map, the "local operator" the reconstruction penalty's quadratic term
#: sandwiches between two copies of ``omega`` at every untouched site.
_LOCAL_GRAM = LOCAL_EFFECT_MAP @ LOCAL_EFFECT_MAP.T


def canonical_omega_seed(pauli_mps: list) -> list:
    """
    The canonical-dual reconstruction-coefficient MPS for ``pauli_mps``.

    Applies :data:`LOCAL_CANONICAL_DUAL_MAP` to every site of ``pauli_mps``
    (an operator's exact Pauli-coefficient MPS, as
    :func:`~mpstab.quantum_hardware.pauli_expansion.mpo_to_pauli_mps`
    returns it -- site tensors shaped ``(4, left, right)``), giving the
    ``omega``-MPS (site tensors shaped ``(6, left, right)``, same bond
    dimension) that reproduces the classical-shadow canonical-dual estimator
    exactly, at zero optimization cost. The default and recommended seed for
    :func:`fit_omega_mps`.
    """
    return [
        np.einsum("kp,plr->klr", LOCAL_CANONICAL_DUAL_MAP, site.real)
        for site in pauli_mps
    ]


def _resize_bond_dimension(omega_mps: list, bond_dimension: int, seed=None) -> list:
    """
    Pad or truncate every bond of ``omega_mps`` to ``bond_dimension``.

    A naive resize (slice to truncate, zero-pad plus tiny noise to grow), not
    an SVD-optimal one: good enough to let :func:`fit_omega_mps` explore a
    bond dimension different from the seed's own, at the cost of not being
    the best possible resize at that dimension.
    """
    rng = np.random.default_rng(seed)
    n = len(omega_mps)
    resized = []
    for site, tensor in enumerate(omega_mps):
        target_l = 1 if site == 0 else bond_dimension
        target_r = 1 if site == n - 1 else bond_dimension
        k, left, right = tensor.shape
        new_tensor = np.zeros((k, target_l, target_r))
        keep_l, keep_r = min(left, target_l), min(right, target_r)
        new_tensor[:, :keep_l, :keep_r] = tensor[:, :keep_l, :keep_r]
        if target_l > left or target_r > right:
            pad_mask = np.ones_like(new_tensor, dtype=bool)
            pad_mask[:, :keep_l, :keep_r] = False
            new_tensor[pad_mask] += (
                1e-6 * rng.standard_normal(new_tensor.shape)[pad_mask]
            )
        resized.append(new_tensor)
    return resized


def shots_to_outcomes(bases: tuple, frequencies: list) -> tuple:
    """
    Flatten a ``"shadows"``-shaped plan's ``(bases, frequencies)`` into
    ``(outcomes, counts)``: one row of per-qubit outcome indices per distinct
    ``(circuit, bitstring)`` pair, and its measured shot count.

    Args:
        bases: one basis string per circuit, as
            :attr:`~mpstab.quantum_hardware.plan.MeasurementPlan.recombination`
            carries it for the ``"shadows"`` route.
        frequencies: one ``{bitstring: count}`` dict per circuit, in the same
            order.

    Returns:
        ``(outcomes, counts)``: ``outcomes`` is an ``(M, n)`` int array,
        ``counts`` an ``(M,)`` float array.
    """
    nqubits = len(bases[0])
    outcomes, counts = [], []
    for basis, freq in zip(bases, frequencies):
        pairs = [_BASIS_TO_OUTCOME_PAIR[label] for label in basis]
        for bitstring, count in freq.items():
            outcomes.append([pairs[q][int(bitstring[q])] for q in range(nqubits)])
            counts.append(count)
    return (
        np.asarray(outcomes, dtype=np.int64).reshape(-1, nqubits),
        np.asarray(counts, dtype=np.float64),
    )


def split_train_test(
    outcomes: np.ndarray, counts: np.ndarray, test_fraction: float, seed=None
) -> tuple:
    """
    Split every ``(outcome row, count)`` pair's count into a train and a test
    count via a binomial draw, so :func:`fit_omega_mps` and the final
    estimate never see the same shots -- the cross-validation the paper's
    Sec. 7.3 recommends to avoid overfitting ``omega`` to the very shots used
    to report its value.

    Returns:
        ``((train_outcomes, train_weights), (test_outcomes, test_weights))``,
        rows with a zero count dropped from each side.
    """
    rng = np.random.default_rng(seed)
    test_counts = rng.binomial(counts.astype(np.int64), test_fraction).astype(
        np.float64
    )
    train_counts = counts - test_counts

    keep_train = train_counts > 0
    keep_test = test_counts > 0
    return (
        (outcomes[keep_train], train_counts[keep_train]),
        (outcomes[keep_test], test_counts[keep_test]),
    )


def evaluate_omega(omega_mps: list, outcomes: np.ndarray) -> np.ndarray:
    """
    ``omega[k_1, ..., k_n]`` at every row of ``outcomes``, batched.

    A chained bond-matrix lookup -- no re-contraction against the target
    operator is needed at estimation time, only during :func:`fit_omega_mps`.
    Cost ``O(n_shots * n * chi**2)``, the same batched-left-environment
    pattern
    :func:`~mpstab.quantum_hardware.pauli_expansion._draw_many_strings` uses.
    """
    n_rows = outcomes.shape[0]
    left = np.ones((n_rows, 1))
    for site, tensor in enumerate(omega_mps):
        blocks = tensor[outcomes[:, site]]  # (n_rows, left, right)
        left = np.einsum("ml,mlr->mr", left, blocks)
    return left[:, 0]


def weighted_mean_and_variance(values: np.ndarray, weights: np.ndarray) -> tuple:
    """``(weighted mean, sample variance of a single unit-weight draw)``."""
    total = float(weights.sum())
    if total == 0:
        return 0.0, 0.0
    mean = float(np.sum(weights * values) / total)
    if total <= 1:
        return mean, 0.0
    variance = float(np.sum(weights * (values - mean) ** 2) / (total - 1))
    return mean, variance


def reconstruction_error(pauli_mps: list, omega_mps: list) -> float:
    """``||O - Pi(omega)||``, the penalty term of Eq. (17), exactly."""
    pauli_mps = [site.real for site in pauli_mps]
    psi_mps = [np.einsum("kp,klr->plr", LOCAL_EFFECT_MAP, site) for site in omega_mps]
    oo = _mps_overlap(pauli_mps, pauli_mps)
    o_psi = _mps_overlap(pauli_mps, psi_mps)
    psi_psi = _mps_overlap(psi_mps, psi_mps)
    return float(np.sqrt(max(oo - 2 * o_psi + psi_psi, 0.0)))


def _mps_overlap(a_mps: list, b_mps: list) -> float:
    """``sum_P a_P b_P`` between two same-length, real Pauli-coefficient MPS."""
    env = np.ones((1, 1))
    for a_site, b_site in zip(a_mps, b_mps):
        env = np.einsum("ab,pac,pbd->cd", env, a_site, b_site)
    return float(env[0, 0])


def _right_environments(
    pauli_mps: list, omega_mps: list, outcomes: np.ndarray
) -> tuple:
    """
    Every site's "everything strictly to its right" environment, in one
    backward pass: ``data_env[site]`` batched per training row,
    ``recon_env[site]`` and ``self_env[site]`` as plain overlap matrices.
    Index ``n`` (one past the last site) holds the trivial boundary.
    """
    n = len(omega_mps)
    n_rows = outcomes.shape[0]
    data_env = [None] * (n + 1)
    recon_env = [None] * (n + 1)
    self_env = [None] * (n + 1)
    data_env[n] = np.ones((n_rows, 1))
    recon_env[n] = np.ones((1, 1))
    self_env[n] = np.ones((1, 1))

    for site in range(n - 1, -1, -1):
        omega_site = omega_mps[site]
        blocks = omega_site[outcomes[:, site]]  # (n_rows, left, right)
        data_env[site] = np.einsum("mlr,mr->ml", blocks, data_env[site + 1])

        psi_site = np.einsum("kp,klr->plr", LOCAL_EFFECT_MAP, omega_site)
        recon_env[site] = np.einsum(
            "pac,pbd,cd->ab", pauli_mps[site], psi_site, recon_env[site + 1]
        )
        self_env[site] = np.einsum(
            "kac,kq,qbd,cd->ab", omega_site, _LOCAL_GRAM, omega_site, self_env[site + 1]
        )
    return data_env, recon_env, self_env


def _solve_site(
    site: int,
    pauli_mps: list,
    omega_mps: list,
    outcomes: np.ndarray,
    weights: np.ndarray,
    left_env_data: np.ndarray,
    left_env_recon: np.ndarray,
    left_env_self: np.ndarray,
    right_env_data: np.ndarray,
    right_env_recon: np.ndarray,
    right_env_self: np.ndarray,
    lam: float,
    ridge: float,
) -> np.ndarray:
    """The local linear solve of Eq. (19)-(20) at one site, ``omega``-shaped."""
    _, l_dim, r_dim = omega_mps[site].shape
    block_size = l_dim * r_dim

    # Variance term A: block-diagonal in the outcome index k (Eq. 19's first
    # term touches only the k that a given shot actually realised).
    k_site = outcomes[:, site]
    u = np.einsum("ml,mr->mlr", left_env_data, right_env_data[site + 1]).reshape(
        outcomes.shape[0], block_size
    )
    a_full = np.zeros((6 * block_size, 6 * block_size))
    for k in range(6):
        mask = k_site == k
        if not np.any(mask):
            continue
        u_k, w_k = u[mask], weights[mask]
        # Some BLAS backends (Apple's Accelerate in particular) raise spurious
        # divide-by-zero/overflow FP warnings on certain (u_k * w_k[:, None]).T
        # @ u_k shapes with no actual non-finite value in either operand or
        # result -- verified by hand on this exact computation. Suppressed
        # locally rather than silencing FP warnings module-wide.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            block = (u_k * w_k[:, None]).T @ u_k
        a_full[
            k * block_size : (k + 1) * block_size, k * block_size : (k + 1) * block_size
        ] = block
    a_full /= max(float(weights.sum()), 1.0)

    # Reconstruction term B (quadratic, self-overlap of Pi(omega)) and v
    # (linear, cross-overlap with the target O).
    b_full = np.einsum(
        "kq,ls,rt->klrqst", _LOCAL_GRAM, left_env_self, right_env_self[site + 1]
    ).reshape(6 * block_size, 6 * block_size)

    cross = np.einsum(
        "ai,pac,cj->pij", left_env_recon, pauli_mps[site], right_env_recon[site + 1]
    )
    v_full = np.einsum("kp,pij->kij", LOCAL_EFFECT_MAP, cross).reshape(6 * block_size)

    # `b_full` is built from `_LOCAL_GRAM = E @ E.T`, which is rank <= 4 on a
    # (6, 6) matrix by construction (E maps the overcomplete 6-outcome space
    # down to the 4-dimensional Pauli space) -- the redundant degrees of
    # freedom the paper's Sec. 4 attributes to any overcomplete POVM. `system`
    # is thus genuinely rank-deficient along those directions, not just
    # ill-conditioned by floating-point noise, so a plain ``solve`` (even with
    # a ridge) amplifies noise there; ``lstsq`` instead returns the
    # minimum-norm local optimum, the least-squares fix the paper's
    # Appendix C recommends.
    system = (1 - lam) * a_full + lam * b_full + ridge * np.eye(6 * block_size)
    solution, *_ = np.linalg.lstsq(system, lam * v_full, rcond=None)
    return solution.reshape(6, l_dim, r_dim)


def fit_omega_mps(
    pauli_mps: list,
    train_outcomes: np.ndarray,
    train_weights: np.ndarray,
    lam: float = 0.999,
    n_sweeps: int = 4,
    bond_dimension: int | None = None,
    ridge: float = 1e-10,
    seed=None,
) -> list:
    """
    Fit the reconstruction-coefficient MPS ``omega`` by minimizing the
    penalty-regularized cost of Eq. (17)-(18) over ``n_sweeps`` left-to-right
    one-site DMRG-like passes, seeded from :func:`canonical_omega_seed`.

    Args:
        pauli_mps: the target observable's Pauli-coefficient MPS, from
            :func:`~mpstab.quantum_hardware.pauli_expansion.mpo_to_pauli_mps`.
        train_outcomes: ``(M, n)`` int array of per-qubit outcome indices,
            from the *training* half of :func:`split_train_test`.
        train_weights: ``(M,)`` float array, that half's shot counts.
        lam: Eq. (17)'s ``lambda``, trading the empirical-variance term
            against the reconstruction penalty. In this convention the two
            terms' *natural* magnitudes differ by orders of magnitude (the
            penalty term is built from raw Pauli-coefficient overlaps, the
            variance term from squared canonical-dual-scale values), so
            ``lambda`` has to sit close to 1 -- as the paper's own examples
            do (``0.999``-``0.9999``) -- for the reconstruction constraint to
            actually bind; the default ``0.999`` was tuned empirically against
            :func:`reconstruction_error` on this module's own convention (it
            needs to sit closer to 1 the shorter/more trivial the tail-folded
            observable already is, since the variance term then has fewer
            genuine degrees of freedom to compete against), not copied from
            the paper's. Lower it only after checking that
            :func:`reconstruction_error` stays small at the value chosen.
        n_sweeps: left-to-right passes. Each solves every site's exact local
            quadratic problem once, so unlike a gradient optimizer this has
            no step-size hyperparameter and the cost is expected to decrease
            (weakly) every site update.
        bond_dimension: resize the seed to this bond dimension first (see
            :func:`_resize_bond_dimension`); ``None`` keeps the target
            operator's own bond dimension, at zero resize cost.
        ridge: Tikhonov regularization added to each local system's
            diagonal, guarding against the ill-conditioning the paper's
            Appendix C flags.
        seed: RNG seed, used only by a non-``None`` ``bond_dimension``'s
            padding noise.

    Returns:
        The optimized ``omega`` MPS, site tensors shaped ``(6, left, right)``.
    """
    pauli_mps = [site.real for site in pauli_mps]
    omega = canonical_omega_seed(pauli_mps)
    if bond_dimension is not None:
        omega = _resize_bond_dimension(omega, bond_dimension, seed)

    n = len(omega)
    if train_weights.sum() == 0 or n_sweeps <= 0:
        return omega

    for _ in range(n_sweeps):
        right_env_data, right_env_recon, right_env_self = _right_environments(
            pauli_mps, omega, train_outcomes
        )
        left_env_data = np.ones((train_outcomes.shape[0], 1))
        left_env_recon = np.ones((1, 1))
        left_env_self = np.ones((1, 1))

        for site in range(n):
            omega[site] = _solve_site(
                site,
                pauli_mps,
                omega,
                train_outcomes,
                train_weights,
                left_env_data,
                left_env_recon,
                left_env_self,
                right_env_data,
                right_env_recon,
                right_env_self,
                lam,
                ridge,
            )

            blocks = omega[site][train_outcomes[:, site]]
            left_env_data = np.einsum("ml,mlr->mr", left_env_data, blocks)
            psi_site = np.einsum("kp,klr->plr", LOCAL_EFFECT_MAP, omega[site])
            left_env_recon = np.einsum(
                "ab,pac,pbd->cd", left_env_recon, pauli_mps[site], psi_site
            )
            left_env_self = np.einsum(
                "ab,kac,kq,qbd->cd",
                left_env_self,
                omega[site],
                _LOCAL_GRAM,
                omega[site],
            )

    return omega
