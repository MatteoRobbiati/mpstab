from dataclasses import dataclass
from typing import Optional, Union

import networkx as nx
import numpy as np
import scipy

from mpstab.engines.tensor_networks.native.utils import (
    _bond_dimension_cut,
    _complex_conjugate,
    multi_trace,
)
from mpstab.pauli import PAULI_MATRICES


@dataclass
class TensorNetwork:
    """
    A tensor network as a graph, supporting contraction and SVD splitting.

    Tensors are numpy arrays held on the nodes of a NetworkX ``MultiDiGraph``;
    each edge records which axis of each endpoint it joins, so an edge is a
    contraction waiting to happen.
    """

    def __post_init__(self):
        self.tensornet = nx.MultiDiGraph()

    @property
    def n_tensors(self):
        return len(self.tensornet.nodes)

    def add_tensor(self, id: str, tensor: np.ndarray):
        self.tensornet.add_node(
            id,
            tensor=tensor,
            shape=tensor.shape,
            free_directions=[True] * len(tensor.shape),
        )

    def add_measurement(self, id: str, alpha: float = 1.0, beta: float = 0.0):
        self.add_tensor(id=id, tensor=np.array([alpha, beta]))

    def add_pauli_pair(self, id: str, p0: str, p1: str):
        """Add a rank-3 tensor stacking the Pauli matrices ``p0`` and ``p1``."""
        self.add_tensor(
            id=id, tensor=np.array([PAULI_MATRICES[p0], PAULI_MATRICES[p1]])
        )

    def add_copy_tensor(self, id: str, n: int):
        tensor = np.zeros(shape=(n, n, n))
        np.fill_diagonal(tensor, 1)
        self.add_tensor(id=id, tensor=tensor)

    def complex_conjugate(self):
        """
        Conjugate every tensor in place, renaming each node ``name -> name_dg``.
        Edges are left unchanged.
        """
        return _complex_conjugate(self.tensornet)

    def add_edge(
        self,
        node_in: str,
        node_out: str,
        edge_id: str,
        directions: tuple[int, int],
        **edge_metadata,
    ):
        """
        Add an edge between two tensors, marking a contraction to perform later.

        Edges are *directed*: the same ``(node_in, node_out)`` order must be used
        when the edge is contracted.

        Args:
            node_in: first tensor to connect.
            node_out: second tensor to connect.
            edge_id: name of the edge.
            directions: which axis of each tensor this edge joins.

        Raises:
            ValueError: if the joined axes have different dimensions, or either
                axis is already taken by another edge.
        """

        d_in, d_out = directions
        if (
            self.tensornet.nodes[node_in]["shape"][d_in]
            != self.tensornet.nodes[node_out]["shape"][d_out]
        ):
            raise ValueError(
                f"Incompatible connected tensor directions {directions}: dim(T_{node_in}[{d_in}]) != dim(T_{node_out}[{d_out}]) ({self.tensornet.nodes[node_in]['shape'][d_in]} != {self.tensornet.nodes[node_out]['shape'][d_out]})"
            )

        if not (
            self.tensornet.nodes[node_in]["free_directions"][d_in]
            and self.tensornet.nodes[node_out]["free_directions"][d_out]
        ):
            raise ValueError("Node directions already in use.")

        self.tensornet.add_edge(
            node_in,
            node_out,
            key=edge_id,
            directions=directions,
            **edge_metadata,
        )

        self.tensornet.nodes[node_in]["free_directions"][d_in] = False
        self.tensornet.nodes[node_out]["free_directions"][d_out] = False

    def remove_edge(self, node_in, node_out, edge_id):
        """Remove an edge, freeing the axes it occupied on both endpoints."""
        d_in, d_out = self.tensornet.edges[node_in, node_out, edge_id]["directions"]

        self.tensornet.remove_edge(node_in, node_out, edge_id)

        self.tensornet.nodes[node_in]["free_directions"][d_in] = True
        self.tensornet.nodes[node_out]["free_directions"][d_out] = True

    def contract(
        self,
        node_in: str,
        node_out: str,
        edge_ids: Union[str, list[str]],
        new_node_id: str,
    ):
        """
        Contract ``node_in`` and ``node_out`` over ``edge_ids`` into one node.

        A self-loop (``node_in == node_out``) becomes a partial trace instead of
        a tensordot. Surviving edges are re-pointed at the merged node.
        """
        if type(edge_ids) is str:
            edge_ids = [edge_ids]

        # The axes each edge occupies on its input and its output node.
        directions_in, directions_out = [], []
        for edge_id in edge_ids:
            directions = self.tensornet.edges[node_in, node_out, edge_id]["directions"]
            directions_in.append(directions[0])
            directions_out.append(directions[1])

        for edge_id in edge_ids:
            self.remove_edge(node_in=node_in, node_out=node_out, edge_id=edge_id)

        if node_in == node_out:
            self._partial_trace(
                node=node_in,
                new_node_id=0,
                directions_in=directions_in,
                directions_out=directions_out,
            )
        else:
            self._contract_separate_nodes(
                node_in=node_in,
                node_out=node_out,
                new_node_id=0,
                directions_in=directions_in,
                directions_out=directions_out,
            )

        nx.relabel_nodes(self.tensornet, {0: new_node_id}, copy=False)

    def svd_decomposition(
        self,
        node: str,
        left_node_id: str,
        left_node_edges: Union[str, list[str]],
        right_node_id: str,
        right_node_edges: Union[str, list[str]],
        middle_node_id: str = "Lambda",
        middle_edge_left: str = "chi",
        middle_edge_right: str = "chi",
        max_bond_dimension: Optional[int] = None,
    ):
        """
        Split a tensor by SVD, ``T = U L V*``, into three connected nodes.

        Each edge of ``node`` is handed to either the ``U`` or the ``V*`` child,
        and the two are joined through the diagonal ``L``. Edges elsewhere in the
        network are re-pointed so the network stays valid. Every axis of ``node``
        must already be assigned to an edge.

        Args:
            node: the tensor to split.
            left_node_id: name for the left unitary ``U``.
            left_node_edges: edges to hand to ``U``.
            right_node_id: name for the right unitary ``V*``.
            right_node_edges: edges to hand to ``V*``.
            middle_node_id: name for the diagonal ``L``.
            middle_edge_left: name for the ``U``-``L`` edge.
            middle_edge_right: name for the ``L``-``V*`` edge.
            max_bond_dimension: keep at most this many singular values. ``None``
                keeps the full numerical rank, discarding only zeros.

        Raises:
            ValueError: if an edge of ``node`` is assigned to neither child, or
                if more edges are assigned than ``node`` has axes.
        """

        if type(left_node_edges) is str:
            left_node_edges = [left_node_edges]

        if type(right_node_edges) is str:
            right_node_edges = [right_node_edges]

        tensor = self.tensornet.nodes[node]["tensor"]

        # Transpose and reshape into a matrix
        transposition_vector = self._svd_transposition_vector(
            node, left_node_edges, right_node_edges
        )
        tensor = np.ascontiguousarray(np.transpose(tensor, transposition_vector))
        matrix_shape = (
            np.prod(tensor.shape[: len(left_node_edges)]),
            np.prod(tensor.shape[len(left_node_edges) :]),
        )
        new_l_shape = *tensor.shape[: len(left_node_edges)], -1
        new_r_shape = -1, *tensor.shape[len(left_node_edges) :]
        tensor = np.reshape(tensor, matrix_shape)

        # Perform SVD
        svd_result = scipy.linalg.svd(
            tensor, full_matrices=False, lapack_driver="gesvd"
        )
        left_tensor, middle_tensor, right_tensor = _bond_dimension_cut(
            *svd_result, max_bond_dimension
        )

        # Reshape into the original tensor dimensions
        left_tensor = np.reshape(left_tensor, new_l_shape)
        right_tensor = np.reshape(right_tensor, new_r_shape)
        middle_tensor = np.diag(middle_tensor)

        # Node is relabled at the end to avoid name collisions with left_node and right_node, which can
        # cause unwanted behavior
        nx.relabel_nodes(self.tensornet, {node: 0}, copy=False)

        # Create the new tensors and connect them
        self.add_tensor(id=left_node_id, tensor=left_tensor)
        self.add_tensor(id=right_node_id, tensor=right_tensor)
        self.add_tensor(id=middle_node_id, tensor=middle_tensor)

        self.add_edge(
            node_in=left_node_id,
            node_out=middle_node_id,
            edge_id=middle_edge_left,
            directions=(len(left_node_edges), 0),
        )
        self.add_edge(
            node_in=right_node_id,
            node_out=middle_node_id,
            edge_id=middle_edge_right,
            directions=(0, 1),
        )

        # Re-establish the old connections
        self._reconnect_edges(
            node=0,
            new_node_id=left_node_id,
            survived_directions=transposition_vector,
            allowed_edges=left_node_edges,
        )
        self._reconnect_edges(
            node=0,
            new_node_id=right_node_id,
            survived_directions=transposition_vector,
            shift=1 - len(left_node_edges),
            allowed_edges=right_node_edges,
        )

        self.tensornet.remove_node(0)

    def _svd_transposition_vector(
        self,
        node: str,
        left_node_edges: Union[str, list[str]],
        right_node_edges: Union[str, list[str]],
    ):
        """
        The axis permutation putting the left child's edges before the right's.

        Returns:
            A list whose position ``i`` holds the axis of ``node``, before the
            split, carrying the ``i``-th edge in that order.

        Raises:
            ValueError: if an edge belongs to neither child, or if more edges are
                assigned than ``node`` has axes.
        """
        transposition_vector = [-1] * (len(left_node_edges) + len(right_node_edges))

        def _update_tv(tv, edge_id, dir):

            if edge_id in left_node_edges:
                tv[left_node_edges.index(edge_id)] = dir
                return

            if edge_id in right_node_edges:
                tv[len(left_node_edges) + right_node_edges.index(edge_id)] = dir
                return

            raise ValueError(
                f"Edge {edge_id!r} is assigned to neither SVD child; every edge "
                "must go to the left or the right one."
            )

        for *_, edge_id, metadata in list(
            self.tensornet.out_edges(nbunch=node, keys=True, data=True)
        ):
            _update_tv(transposition_vector, edge_id, metadata["directions"][0])

        for *_, edge_id, metadata in list(
            self.tensornet.in_edges(nbunch=node, keys=True, data=True)
        ):
            _update_tv(transposition_vector, edge_id, metadata["directions"][1])

        if any(axis < 0 for axis in transposition_vector):
            raise ValueError(
                f"More edges were assigned to the SVD children than tensor {node} "
                "has axes."
            )

        return transposition_vector

    def _contract_separate_nodes(
        self,
        node_in: str,
        node_out: str,
        new_node_id: str,
        directions_in: list,
        directions_out: list,
    ):
        """Contract two distinct nodes with ``np.tensordot``, then remove them."""
        # Axes not consumed by the contraction, which the merged node inherits.
        non_contracted_index_in = [
            i
            for i in range(len(self.tensornet.nodes[node_in]["shape"]))
            if i not in directions_in
        ]
        non_contracted_index_out = [
            i
            for i in range(len(self.tensornet.nodes[node_out]["shape"]))
            if i not in directions_out
        ]

        # Construct the contracted tensor (the future new node in the graph)
        new_tensor = np.tensordot(
            a=self.tensornet.nodes[node_in]["tensor"],
            b=self.tensornet.nodes[node_out]["tensor"],
            axes=(directions_in, directions_out),
        )

        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)

        # Transfer the edge connections from the old to the new node and delete it
        self._reconnect_edges(
            node=node_in,
            new_node_id=new_node_id,
            survived_directions=non_contracted_index_in,
        )
        self._reconnect_edges(
            node=node_out,
            new_node_id=new_node_id,
            survived_directions=non_contracted_index_out,
            shift=len(
                non_contracted_index_in
            ),  # Comply with indexing convention of numpy tensordot
        )

        self.tensornet.remove_node(node_in)
        self.tensornet.remove_node(node_out)

    def _partial_trace(
        self, node: str, new_node_id: str, directions_in: list, directions_out: list
    ):
        """Contract a node's self-loops with :func:`multi_trace`, then remove it."""
        non_contracted_index = [
            i
            for i in range(len(self.tensornet.nodes[node]["shape"]))
            if i not in (directions_in + directions_out)
        ]

        new_tensor = multi_trace(
            tensor=self.tensornet.nodes[node]["tensor"],
            directions_in=directions_in,
            directions_out=directions_out,
        )

        # Add new node, containing contracted tensors
        self.add_tensor(id=new_node_id, tensor=new_tensor)

        # Transfer the edge connections from the old to the new node and delete it
        self._reconnect_edges(
            node=node,
            new_node_id=new_node_id,
            survived_directions=non_contracted_index,
        )

        self.tensornet.remove_node(node)

    def _reconnect_edges(
        self,
        node: str,
        new_node_id: str,
        survived_directions: list,
        shift: int = 0,
        allowed_edges: Optional[list[str]] = None,
    ):
        """
        Move ``node``'s edges onto ``new_node_id``, remapping the axis they point at.

        ``survived_directions`` lists the old axes the new node kept, in its own
        axis order, and ``shift`` offsets that order when the new node is a merge
        of two tensors. ``allowed_edges``, when given, restricts which edges move.
        """
        for u, v, edge_id, metadata in list(
            self.tensornet.out_edges(nbunch=node, keys=True, data=True)
        ):

            if allowed_edges is not None and edge_id not in allowed_edges:
                continue

            directions = (
                # remapped onto the new node
                survived_directions.index(metadata["directions"][0]) + shift,
                # unchanged on the far endpoint
                metadata["directions"][1],
            )

            self.remove_edge(node_in=u, node_out=v, edge_id=edge_id)
            self.add_edge(
                node_in=new_node_id, node_out=v, edge_id=edge_id, directions=directions
            )

        for u, v, edge_id, metadata in list(
            self.tensornet.in_edges(nbunch=node, keys=True, data=True)
        ):

            if allowed_edges is not None and edge_id not in allowed_edges:
                continue

            directions = (
                # unchanged on the far endpoint
                metadata["directions"][0],
                # remapped onto the new node
                survived_directions.index(metadata["directions"][1]) + shift,
            )

            self.remove_edge(node_in=u, node_out=v, edge_id=edge_id)
            self.add_edge(
                node_in=u, node_out=new_node_id, edge_id=edge_id, directions=directions
            )
