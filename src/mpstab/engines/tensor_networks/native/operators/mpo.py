"""The matrix product operator, used for both gates and observables."""

from typing import Optional

import numpy as np

from mpstab.engines.tensor_networks.native.tensor_network import TensorNetwork


class MPO(TensorNetwork):
    """A chain of tensors joined along one bond, representing a gate or observable."""

    def __init__(
        self,
        tensors: list[np.ndarray],
        link_directions: Optional[list[tuple[int, int]]] = None,
        physical_directions: Optional[list[tuple[int, int]]] = None,
        tensor_prefix: str = "O",
        link_name: str = "h_link",
    ):
        """
        Args:
            tensors: one array per site, in order.
            link_directions: which axes each neighbouring pair joins. Defaults to
                axes ``(2, 3)`` inside the chain and ``(2, 2)`` at its right end.
            physical_directions: the (output, input) axes of each site. Defaults
                to ``(1, 0)`` everywhere.
            tensor_prefix: node names are this plus the site index.
            link_name: name shared by every bond edge.

        Raises:
            ValueError: if the number of physical or link directions does not
                match the number of tensors.
        """
        n_tensors = len(tensors)

        if link_directions is None:
            link_directions = (
                [(2, 3) if i < n_tensors - 2 else (2, 2) for i in range(n_tensors - 1)]
                if n_tensors > 1
                else []
            )

        if physical_directions is None:
            physical_directions = [(1, 0) for i in range(n_tensors)]

        if n_tensors != len(physical_directions):
            raise ValueError(
                f"{n_tensors} tensors but {len(physical_directions)} physical "
                "directions; there must be one per tensor."
            )
        if n_tensors != len(link_directions) + 1:
            raise ValueError(
                f"{n_tensors} tensors but {len(link_directions)} link directions; "
                f"there must be {n_tensors - 1}, one per bond."
            )

        self.physical_directions = physical_directions
        self.prefix = tensor_prefix
        self.link = link_name

        super().__init__()

        self.add_tensor(self.prefix + "0", tensor=tensors[0])
        for q, (t, link_dir) in enumerate(zip(tensors[1:], link_directions), start=1):
            self.add_tensor(self.prefix + f"{q}", tensor=t)
            self.add_edge(
                self.prefix + f"{q-1}", self.prefix + f"{q}", link_name, link_dir
            )
