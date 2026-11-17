"""
TN-ICE: an MPS-optimized dual estimator for the ``"shadows"`` route's data.

Mangini & Cavalcanti, *Low variance estimations of many observables with
tensor networks and informationally-complete measurements* (Quantum 2025,
arXiv:2407.02923), replace the fixed *canonical-dual* post-processing of
classical shadows with a second MPS ``omega`` of reconstruction coefficients,
optimized to minimize an empirical-variance-plus-reconstruction-penalty cost
(their Eq. 16-20) against the very data classical shadows would otherwise
contract with the fixed canonical dual. The measurement is the shadows one,
random local Pauli bases, which is exactly the six-effect OC-POVM of the
paper's Eq. (24)/(32); the effect map, the canonical dual and the batched
evaluation of ``omega`` live in :mod:`~mpstab.qpu.measurement.shadows`.

:func:`fit_omega_mps` starts from the canonical dual
(:func:`~mpstab.qpu.measurement.shadows.canonical_omega_seed`), so zero sweeps
reproduce the ``"shadows"`` estimate, and runs one-site DMRG-like sweeps,
solving the exact local quadratic problem of Eq. (19)-(20) at each site from
environment tensors accumulated over the (training) measured shots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from mpstab.qpu.measurement.abstract import ExpectationResult
from mpstab.qpu.measurement.shadows import (
    LOCAL_EFFECT_MAP,
    ClassicalShadows,
    canonical_omega_seed,
    estimate_with_omega,
    real_pauli_mps,
    shots_to_outcomes,
)

#: ``(6, 6)``: the local physical-index Gram matrix ``E @ E.T`` of the effect
#: map, the "local operator" the reconstruction penalty's quadratic term
#: sandwiches between two copies of ``omega`` at every untouched site.
_LOCAL_GRAM = LOCAL_EFFECT_MAP @ LOCAL_EFFECT_MAP.T


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


def reconstruction_error(pauli_mps: list, omega_mps: list) -> float:
    """``||O - Pi(omega)||``, the penalty term of Eq. (17), exactly."""
    pauli_mps = real_pauli_mps(pauli_mps)
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


def _fit_right_environments(
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
            :func:`~mpstab.qpu.measurement.pauli_expansion.mpo_to_pauli_mps`.
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
    pauli_mps = real_pauli_mps(pauli_mps)
    omega = canonical_omega_seed(pauli_mps)
    if bond_dimension is not None:
        omega = _resize_bond_dimension(omega, bond_dimension, seed)

    n = len(omega)
    if train_weights.sum() == 0 or n_sweeps <= 0:
        return omega

    for _ in range(n_sweeps):
        right_env_data, right_env_recon, right_env_self = _fit_right_environments(
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


@dataclass(frozen=True)
class TNICE(ClassicalShadows):
    """
    The ``"tnice"`` method: the same circuits, bases and shots as
    :class:`~mpstab.qpu.measurement.shadows.ClassicalShadows`, estimated with
    the MPS-optimized dual of :func:`fit_omega_mps` instead of the fixed
    canonical dual, so the two can be compared on identical measurement data.

    With an ``epsilon`` target the shot budget is sized by the
    canonical-dual predictor of ``"shadows"``, a conservative upper bound on
    the (unknown ahead of fitting) optimized variance.

    Args:
        shots_per_setting: as in ``ClassicalShadows``.
        tail_handling: as in ``ClassicalShadows``.
        lam: forwarded to :func:`fit_omega_mps`.
        n_sweeps: forwarded to :func:`fit_omega_mps`.
        bond_dimension: forwarded to :func:`fit_omega_mps`; ``None`` keeps
            the tail MPO's own.
        test_fraction: fraction of each circuit's shots held out for the final
            estimate; the rest trains ``omega``.
    """

    lam: float = 0.999
    n_sweeps: int = 4
    bond_dimension: int | None = None
    test_fraction: float = 0.5

    name: ClassVar[str] = "tnice"

    def estimate(self, plan, frequencies, seed=None) -> ExpectationResult:
        """
        Fit ``omega`` per term on a training share of the shots and report the
        estimate on the held-out rest, avoiding the overfitting the paper's
        Sec. 7.3 warns an in-sample estimate would show.

        Args:
            plan, frequencies: a shadows-shaped plan and its frequencies.
            seed: RNG seed for the train/test split.
        """
        _, bases = plan.recombination
        outcomes, counts = shots_to_outcomes(bases, frequencies)
        (train, train_weights), (test, test_weights) = split_train_test(
            outcomes, counts, self.test_fraction, seed
        )

        def fitted_omega(pauli_mps):
            return fit_omega_mps(
                pauli_mps,
                train,
                train_weights,
                lam=self.lam,
                n_sweeps=self.n_sweeps,
                bond_dimension=self.bond_dimension,
                seed=seed,
            )

        return estimate_with_omega(
            plan, frequencies, test, test_weights, fitted_omega, counts.sum()
        )
