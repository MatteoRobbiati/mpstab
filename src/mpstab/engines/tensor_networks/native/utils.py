"""Numerical helpers for the pure-Python tensor network."""

import networkx as nx
import numpy as np

#: Singular values below this are treated as zero when cutting a bond.
SVD_CUT = 1e-10


def multi_trace(tensor, directions_in: list[int], directions_out: list[int]):
    """
    Trace out each ``(directions_in[i], directions_out[i])`` pair of axes.

    Traces one pair at a time with ``np.trace``, shifting the remaining axis
    indices down after each contraction so they keep pointing at the right axes.
    """
    while directions_in:
        d_in, d_out = directions_in[0], directions_out[0]
        tensor = np.trace(tensor, axis1=d_in, axis2=d_out)
        directions_in = [d - (d > d_in) - (d > d_out) for d in directions_in[1:]]
        directions_out = [d - (d > d_in) - (d > d_out) for d in directions_out[1:]]
    return tensor


def _complex_conjugate(tensornet):
    """Conjugate every tensor in place, renaming each node ``name -> name_dg``."""
    for node in list(tensornet.nodes):
        tensornet.nodes[node]["tensor"] = np.conj(tensornet.nodes[node]["tensor"])
        nx.relabel_nodes(tensornet, {node: f"{node}_dg"}, copy=False)


def _bond_dimension_cut(U, D, V, max_bond_dimension):
    """
    Truncate an SVD to ``max_bond_dimension`` singular values, renormalising ``D``.

    Values below :data:`SVD_CUT` are dropped regardless;
    ``max_bond_dimension=None`` keeps the full numerical rank.
    """
    rank = np.count_nonzero(D > SVD_CUT)
    bond_dimension = (
        rank if max_bond_dimension is None else min(max_bond_dimension, rank)
    )
    retained = D[:bond_dimension]
    return (
        U[:, :bond_dimension],
        retained / np.sqrt(np.sum(retained**2)),
        V[:bond_dimension, :],
    )
