"""
An SVD for jax whose gradient stays finite on rank-deficient matrices.

jax's own SVD derivative divides by ``s_i**2 - s_j**2`` and by ``s``, so repeated
zero singular values give ``inf`` and then ``nan``. An MPS that starts as a
product state and is evolved one rotation at a time produces exactly such
matrices at its first compressions, and whether the zeros are exact or a few ulps
off depends on the platform: ``jax.grad`` through ``HSMPO`` returned ``nan`` for
the first-layer parameters on one CI runner and was fine on others.

The rule below is jax's (``jax._src.lax.linalg._svd_jvp_rule``) with every
division made safe: where the denominator is zero the term is dropped. Those terms
only rotate within a degenerate (or null) singular subspace, where the singular
vectors are arbitrary and no output of a truncated decomposition depends on them.
"""

import jax
import jax.numpy as jnp


def _safe_inverse(x):
    nonzero = x != 0
    return jnp.where(nonzero, 1.0 / jnp.where(nonzero, x, 1.0), 0.0)


def _hermitian(x):
    return jnp.swapaxes(x.conj(), -1, -2)


@jax.custom_jvp
def _reduced_svd(a):
    return jnp.linalg.svd(a, full_matrices=False)


@_reduced_svd.defjvp
def _reduced_svd_jvp(primals, tangents):
    (a,) = primals
    (da,) = tangents
    u, s, vh = _reduced_svd(a)
    ut, v = _hermitian(u), _hermitian(vh)

    s_row = s[..., None, :]
    d_s = ut @ da @ v
    ds = jnp.real(jnp.diagonal(d_s, axis1=-2, axis2=-1))

    f = _safe_inverse(
        (s_row + jnp.swapaxes(s_row, -1, -2)) * (s_row - jnp.swapaxes(s_row, -1, -2))
    )
    s_inv = _safe_inverse(s)
    s_inv_diag = s_inv[..., None, :] * jnp.eye(s.shape[-1], dtype=s.dtype)

    d_ss = s_row.astype(a.dtype) * d_s
    s_ds = jnp.swapaxes(s_row, -1, -2).astype(a.dtype) * d_s
    du_dv_diag = 0.5 * (d_s - _hermitian(d_s)) * s_inv_diag.astype(a.dtype)
    du = u @ (f.astype(a.dtype) * (d_ss + _hermitian(d_ss)) + du_dv_diag)
    dv = v @ (f.astype(a.dtype) * (s_ds + _hermitian(s_ds)))

    m, n = a.shape[-2:]
    s_inv_row = s_inv[..., None, :].astype(a.dtype)
    if m > n:
        dav = da @ v
        du = du + (dav - u @ (ut @ dav)) * s_inv_row
    if n > m:
        dahu = _hermitian(da) @ u
        dv = dv + (dahu - v @ (vh @ dahu)) * s_inv_row

    return (u, s, vh), (du, ds, _hermitian(dv))


def safe_svd(a, full_matrices=False, compute_uv=True, **kwargs):
    """Drop-in for ``jnp.linalg.svd`` with a derivative that never produces ``nan``
    from repeated or zero singular values (reduced decompositions only)."""
    if full_matrices or not compute_uv or kwargs:
        return jnp.linalg.svd(
            a, full_matrices=full_matrices, compute_uv=compute_uv, **kwargs
        )
    return _reduced_svd(a)


def install():
    """Make ``autoray`` (and so quimb) use :func:`safe_svd` on the jax backend."""
    import autoray

    autoray.register_function("jax", "linalg.svd", safe_svd)
