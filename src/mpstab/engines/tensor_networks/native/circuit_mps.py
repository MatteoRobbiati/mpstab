"""An MPS circuit simulator built on the pure-Python tensor network."""

from copy import deepcopy
from typing import Optional

import networkx as nx
import numpy as np

from mpstab.engines.tensor_networks.native.operators import MPO
from mpstab.engines.tensor_networks.native.operators.gates import (
    CNOT,
    CZ,
    SWAP,
    CNOT_inv,
    H,
    PauliExp,
    S,
    T,
    X,
    Y,
    Z,
)
from mpstab.engines.tensor_networks.native.operators.utils import basis
from mpstab.engines.tensor_networks.native.tensor_network import TensorNetwork


class CircuitMPS(TensorNetwork):
    """
    A quantum circuit simulated as a matrix product state.

    The state is kept in Vidal canonical form -- site tensors ``T{q}`` separated
    by Schmidt-coefficient tensors ``L{q}`` -- and evolved by applying MPOs and
    re-splitting by SVD, which preserves that form. Gates are applied through
    :meth:`apply`, observables read out through :meth:`expval`.
    """

    def __init__(
        self,
        n: int,
        initial_state: Optional[str | np.ndarray] = None,
        max_bond_dimension: Optional[int] = None,
    ):
        """
        Args:
            n: number of qubits, at least 2.
            initial_state: per-site amplitudes, or a string of ``0``/``1``/``+``/
                ``-`` characters. Defaults to all zeros.
            max_bond_dimension: truncation cap applied at every SVD, or ``None``
                for no cap.

        Raises:
            ValueError: on fewer than 2 qubits, or an initial state of the wrong
                length.
        """
        self.n_qubits = n
        self.max_bond_dimension = max_bond_dimension

        if initial_state is None:
            initial_state = n * "0"
        if type(initial_state) is str:
            initial_state = [basis(bit) for bit in initial_state]
        if n < 2:
            raise ValueError(f"CircuitMPS needs at least 2 qubits, got {n}.")
        if n != len(initial_state):
            raise ValueError(
                f"Initial state covers {len(initial_state)} qubits but the circuit "
                f"has {n}."
            )

        super().__init__()

        # One site tensor T{q} plus a measurement stub D{q} per qubit, with
        # Schmidt tensors L{q} on the bonds between them.
        self.add_tensor("T0", tensor=np.reshape(initial_state[0], (2, 1)))
        self.add_measurement("D0")
        self.add_edge("T0", "D0", "phyisical0", (0, 0))

        for q, s in enumerate(initial_state[1:], 1):
            self.add_tensor(f"T{q}", tensor=np.reshape(s, (2, 1, 1)))
            self.add_tensor(f"L{q-1}", tensor=np.reshape(np.array([1]), (1, 1)))
            self.add_measurement(f"D{q}")

            self.add_edge(f"T{q}", f"D{q}", f"phyisical{q}", (0, 0))
            self.add_edge(f"T{q}", f"L{q-1}", f"chi{q-1}_r", (2, 0))
            self.add_edge(f"T{q-1}", f"L{q-1}", f"chi{q-1}_l", (1, 1))

        # Close the last site's dangling right bond.
        self.add_tensor("tmp", tensor=np.array([1]))
        self.add_edge(f"T{n-1}", "tmp", "link", (1, 0))
        self.contract(f"T{n-1}", "tmp", "link", f"T{n-1}")

    def bipartite_entanglement_entropy(self, cut: int):
        """
        Von Neumann entanglement entropy across the bond after the first ``cut``
        sites, read straight off the Schmidt coefficients the canonical form keeps.
        """
        if not 1 <= cut <= self.n_qubits - 1:
            raise ValueError(
                f"Cut {cut} leaves an empty partition; it must be between 1 and "
                f"{self.n_qubits - 1}."
            )
        spectrum = np.diagonal(self.tensornet.nodes[f"L{cut-1}"]["tensor"]) ** 2
        return np.sum(-spectrum * np.log(spectrum))

    def amplitude(self, basis_element: str):
        """
        The amplitude of a given basis element.

        Args:
            basis_element: one of ``0``, ``1``, ``+`` or ``-`` per qubit.
        """
        if len(basis_element) != self.n_qubits:
            raise ValueError(
                f"Basis element covers {len(basis_element)} qubits but the state "
                f"has {self.n_qubits}."
            )

        mps = deepcopy(self)
        for q, state in enumerate(basis_element):
            mps.add_tensor("measurement", tensor=basis(state))
            mps._link_to_dummy(f"D{q}", "measurement", 0)
            mps.contract(f"T{q}", "measurement", "v_link", f"T{q}")

        mps.contract("T0", "L0", "chi0_l", "F")
        for q, q_next in zip(range(mps.n_qubits), range(1, mps.n_qubits - 1)):
            mps.contract(f"T{q_next}", "F", f"chi{q}_r", "F")
            mps.contract("F", f"L{q_next}", f"chi{q_next}_l", "F")

        mps.contract(f"T{self.n_qubits-1}", "F", f"chi{self.n_qubits-2}_r", "F")
        return mps.tensornet.nodes["F"]["tensor"].item()

    def cnot(self, control, target):
        """Apply a CNOT gate."""
        gate = CNOT if control < target else CNOT_inv
        return self.apply(gate, sorted([control, target]))

    def cz(self, control, target):
        """Apply a CZ gate."""
        return self.apply(CZ, sorted([control, target]))

    def swap(self, control, target):
        """Apply a SWAP gate."""
        return self.apply(SWAP, sorted([control, target]))

    def h(self, qubit):
        """Apply a Hadamard gate."""
        self.apply(H, [qubit])

    def x(self, qubit):
        """Apply an X gate."""
        self.apply(X, [qubit])

    def y(self, qubit):
        """Apply a Y gate."""
        self.apply(Y, [qubit])

    def z(self, qubit):
        """Apply a Z gate."""
        self.apply(Z, [qubit])

    def s(self, qubit):
        """Apply an S gate."""
        self.apply(S, [qubit])

    def t(self, qubit):
        """Apply a T gate."""
        self.apply(T, [qubit])

    def pauli_rot(self, pauli_generator, theta, qubits=None):
        """Apply ``exp(-i theta/2 P)`` for the Pauli string ``pauli_generator``."""
        self.apply(PauliExp(pauli_generator, theta), qubits)

    def expval(self, obs: MPO, sites: Optional[list[int]] = None):
        """
        Expectation value of an MPO observable.

        Sites outside the observable's support are traced out before contracting
        the bra, the observable and the ket layer by layer.

        Args:
            obs: the observable.
            sites: the qubits it acts on, which must be adjacent and ascending.
                Defaults to every qubit.
        """
        sites = self._check_sites(sites)

        # "Ket" MPS
        tn = deepcopy(self)

        # Cut unnecessary sites, and contract the remaning central nodes
        if sites[0] > 0:
            tn.remove_edge(f"T{sites[0]-1}", f"L{sites[0]-1}", f"chi{sites[0]-1}_l")
            tn.contract(
                f"T{sites[0]}", f"L{sites[0]-1}", f"chi{sites[0]-1}_r", f"T{sites[0]}"
            )

        if sites[-1] < self.n_qubits - 1:
            tn.remove_edge(f"T{sites[-1]+1}", f"L{sites[-1]}", f"chi{sites[-1]}_r")
            tn.contract(
                f"T{sites[-1]}", f"L{sites[-1]}", f"chi{sites[-1]}_l", f"T{sites[-1]}"
            )

        # Contract middle node
        for s in sites[1:]:
            tn.contract(f"T{s}", f"L{s-1}", f"chi{s-1}_r", f"T{s}")

        # "Bra" MPS
        bra = deepcopy(tn)
        bra.complex_conjugate()

        # Whole TensorNetework
        tn.tensornet = nx.union(tn.tensornet, obs.tensornet)
        tn.tensornet = nx.union(tn.tensornet, bra.tensornet)

        # Connect to the observable and remove dummy tensor
        for i, s in enumerate(sites):
            tn._link_to_dummy(
                f"D{s}", obs.prefix + f"{i}", obs.physical_directions[i][0]
            )
            tn._link_to_dummy(
                f"D{s}_dg", obs.prefix + f"{i}", obs.physical_directions[i][1]
            )

        # If necessary, reattach the extremal nodes
        if sites[0] > 0:
            free_d = tn.tensornet.nodes[f"T{sites[0]}"]["free_directions"].index(True)
            tn.add_edge(
                f"T{sites[0]}", f"T{sites[0]}_dg", f"left_link", (free_d, free_d)
            )

        if sites[-1] < self.n_qubits - 1:
            free_d = tn.tensornet.nodes[f"T{sites[-1]}"]["free_directions"].index(True)
            tn.add_edge(
                f"T{sites[-1]}", f"T{sites[-1]}_dg", f"right_link", (free_d, free_d)
            )

        # Contract the first layer
        tn.contract(f"T{sites[0]}", obs.prefix + f"0", "v_link", "F")
        tn.contract(f"T{sites[0]}_dg", "F", "v_link", "F")

        if sites[0] > 0:
            tn.contract(f"F", "F", "left_link", "F")

        for i, s in enumerate(sites[1:], start=1):
            tn.contract(f"T{s}", obs.prefix + f"{i}", "v_link", "tmp")
            tn.contract("F", f"tmp", [f"chi{s-1}_l", obs.link], "F")
            tn.contract("F", f"T{s}_dg", f"chi{s-1}_l", "F")
            tn.contract("F", "F", "v_link", "F")

        if sites[-1] < self.n_qubits - 1:
            tn.contract(f"F", "F", "right_link", "F")

        res = tn.tensornet.nodes["F"]["tensor"].item()
        return np.real(res)

    def _check_sites(self, sites: Optional[list[int]]) -> list[int]:
        """Default ``sites`` to the whole register, and require them contiguous."""
        if sites is None:
            return list(range(self.n_qubits))
        for site, next_site in zip(sites, sites[1:]):
            if next_site - site != 1:
                raise ValueError(
                    "An MPO's qubits must be adjacent and ascending; got the jump "
                    f"{site} -> {next_site}."
                )
        return sites

    def apply(self, mpo: MPO, sites: Optional[list[int]] = None):
        """
        Apply an MPO to the state, keeping the canonical form.

        Contracts the MPO in pairwise, re-splitting by SVD at every bond and
        reinserting the Schmidt coefficients that the contraction absorbed.

        Args:
            mpo: the operator to apply.
            sites: the qubits it acts on, which must be adjacent and ascending.
                Defaults to every qubit.
        """
        sites = self._check_sites(sites)

        # Link the MPS and MPO along the physical direction
        self.tensornet = nx.union(self.tensornet, mpo.tensornet)
        for i, s in enumerate(sites):
            self._move_dummy(
                f"D{s}",
                mpo.prefix + f"{i}",
                mpo.physical_directions[i],
                out_edge_id=f"physical{s}",
            )

        # If only one site is affected, apply simple matrix multiplication
        if len(sites) == 1:
            return self.contract(
                f"T{sites[0]}", mpo.prefix + f"0", "v_link", f"T{sites[0]}"
            )

        # Otherwise, perform pairwise contraction
        # -- Step 1: Save the Schmidt coefficients of the right extreme, if applicable
        if sites[-1] < self.n_qubits - 1:
            L_last = self.tensornet.nodes[f"L{sites[-1]}"]["tensor"]

        # -- Step 2: Prepare the leftmost node to be contracted
        self.contract(
            f"T{sites[0]}", f"L{sites[0]}", f"chi{sites[0]}_l", f"T{sites[0]}"
        )
        self.contract(f"T{sites[0]}", mpo.prefix + f"0", "v_link", f"T{sites[0]}")

        # -- Step 3: Sequentially contract the MPO
        for i, (s, s_next) in enumerate(zip(sites, sites[1:]), start=1):

            # -- Step 3.1: Contract the nodes involving two consecutive sites
            if s > 0:
                L = self.tensornet.nodes[f"L{s-1}"]["tensor"]
                self.contract(f"T{s}", f"L{s-1}", f"chi{s-1}_r", f"T{s}")

            self.contract(f"T{s_next}", f"T{s}", f"chi{s}_r", f"T{s}")
            self.contract(f"T{s}", mpo.prefix + f"{i}", [mpo.link, "v_link"], f"T{s}")

            if s_next < self.n_qubits - 1:
                self.contract(f"T{s}", f"L{s_next}", f"chi{s_next}_l", f"T{s}")

            # -- Step 3.2: SVD
            self.svd_decomposition(
                node=f"T{s}",
                left_node_id=f"T{s}",
                left_node_edges=[f"physical{s}"] + ([f"chi{s-1}_l"] if s > 0 else []),
                right_node_id=f"T{s_next}",
                right_node_edges=(
                    [f"chi{s_next}_r"] if s_next < self.n_qubits - 1 else []
                )
                + ([mpo.link] if s_next < sites[-1] else [])
                + [f"physical{s_next}"],
                middle_node_id=f"L{s}",
                middle_edge_left=f"chi{s}_l",
                middle_edge_right=f"chi{s}_r",
                max_bond_dimension=self.max_bond_dimension,
            )

            # -- Step 3.3: Insert the Schmidt coefficients back, to restore the normal form for the MPS
            if s > 0:
                self._insert_square_matrix(
                    f"T{s-1}",
                    f"T{s}",
                    f"chi{s-1}_l",
                    L,
                    f"L{s-1}",
                    right_edge_name=f"chi{s-1}_r",
                )
                T = self.tensornet.nodes[f"T{s}"]["tensor"]
                self.tensornet.nodes[f"T{s}"]["tensor"] = np.linalg.solve(L, T)

        # -- Step 4: Complete restoring the normal form of the MPS
        if sites[-1] < self.n_qubits - 1:
            self._insert_square_matrix(
                f"T{sites[-1]+1}",
                f"T{sites[-1]}",
                f"chi{sites[-1]}_r",
                L_last,
                f"L{sites[-1]}",
                right_edge_name=f"chi{sites[-1]}_l",
            )
            T = self.tensornet.nodes[f"T{sites[-1]}"]["tensor"]
            self.tensornet.nodes[f"T{sites[-1]}"]["tensor"] = np.linalg.solve(L_last, T)

    def _insert_square_matrix(
        self,
        left_node: str,
        right_node: str,
        edge: str,
        matrix: np.ndarray,
        matrix_name: str,
        right_edge_name: Optional[str] = None,
        left_edge_name: Optional[str] = None,
    ):
        """
        Insert a square matrix into an edge, splitting it in two.

        Args:
            left_node: node on the left of the edge.
            right_node: node on the right of the edge.
            edge: the edge to split.
            matrix: the square matrix to insert.
            matrix_name: node name for the inserted matrix.
            right_edge_name: name for the new edge on the right; ``None`` reuses
                the old edge name.
            left_edge_name: name for the new edge on the left; ``None`` reuses the
                old edge name.
        """

        dleft, dright = self.tensornet.edges[left_node, right_node, edge]["directions"]
        self.add_tensor(matrix_name, tensor=matrix)
        self.remove_edge(left_node, right_node, edge)
        self.add_edge(
            left_node,
            matrix_name,
            left_edge_name if left_edge_name is not None else edge,
            (dleft, 1),
        )
        self.add_edge(
            right_node,
            matrix_name,
            right_edge_name if right_edge_name is not None else edge,
            (dright, 0),
        )

    def _move_dummy(
        self,
        dummy: str,
        tensor: str,
        tensor_directions: tuple[int, int],
        in_edge_id: str = "v_link",
        out_edge_id: str = "physical",
    ):
        """
        Splice a newly added tensor in where a dummy node hangs::

            (M)-(dummy)  -(T)-   =>   (M)-(T)-(dummy)

        The dummy takes the tensor's output axis, so it stays available as the
        chain's free end.

        Args:
            dummy: the dummy node to shift along.
            tensor: the tensor to splice in.
            tensor_directions: its (input, output) axes.
            in_edge_id: name for the incoming edge.
            out_edge_id: name for the outgoing edge.
        """
        self._link_to_dummy(dummy, tensor, tensor_directions[0], in_edge_id)
        self.add_edge(tensor, dummy, out_edge_id, (tensor_directions[1], 0))

    def _link_to_dummy(
        self, dummy: str, tensor: str, tensor_direction: int, edge_id: str = "v_link"
    ):
        """Move the edge feeding ``dummy`` onto a free axis of ``tensor``."""
        source, target, edge, data = list(
            self.tensornet.in_edges(dummy, data=True, keys=True)
        )[0]
        dummy_direction = data["directions"][0]
        self.remove_edge(source, target, edge)
        self.add_edge(source, tensor, edge_id, (dummy_direction, tensor_direction))
