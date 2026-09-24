"""Tensor-network engine backed by quimb, with cotengra contraction paths."""

from __future__ import annotations

from typing import Any

import cotengra as ctg
from qibo.gates.abstract import ParametrizedGate
from quimb.gates import I, X, Y, Z
from quimb.tensor import (
    CircuitMPS,
    MatrixProductOperator,
    MatrixProductState,
    MPO_identity,
    MPO_product_operator,
)

from mpstab.engines.tensor_networks.abstract import TensorNetworkEngine

#: qibo gate names mapped to their quimb equivalents.
GATE_MAP = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "t": "T",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "u3": "U3",
    "cx": "CX",
    "cnot": "CNOT",
    "cy": "CY",
    "cz": "CZ",
    "iswap": "ISWAP",
    "swap": "SWAP",
    "ccx": "CCX",
    "ccy": "CCY",
    "ccz": "CCZ",
    "toffoli": "TOFFOLI",
    "cswap": "CSWAP",
    "fredkin": "FREDKIN",
    "fsim": "fsim",
    "measure": "measure",
}

#: Pauli labels mapped to quimb's own single-qubit Pauli arrays.
PAULI_GATES = {"I": I, "X": X, "Y": Y, "Z": Z}


def _qibo_circuit_to_quimb(nqubits: int, qibo_circ, **circuit_kwargs) -> CircuitMPS:
    """
    Convert a qibo circuit into a quimb ``CircuitMPS``. Measurement gates are
    skipped.

    Args:
        nqubits: circuit width.
        qibo_circ: the qibo circuit to convert.
        circuit_kwargs: forwarded to the ``CircuitMPS`` constructor, e.g.
            ``max_bond`` and ``to_backend``.

    Raises:
        ValueError: on a gate with no quimb equivalent in :data:`GATE_MAP`.
    """
    circuit = CircuitMPS(nqubits, **circuit_kwargs)

    for gate in qibo_circ.queue:
        name = getattr(gate, "name", None)
        quimb_name = GATE_MAP.get(name)
        if quimb_name == "measure":
            continue
        if quimb_name is None:
            raise ValueError(f"Gate {name} is not supported by the quimb engine.")

        parameters = getattr(gate, "parameters", ())
        qubits = getattr(gate, "qubits", ())
        parametrized = isinstance(gate, ParametrizedGate) and getattr(
            gate, "trainable", True
        )
        if parametrized:
            circuit.apply_gate(quimb_name, *parameters, *qubits, parametrized=True)
        else:
            circuit.apply_gate(quimb_name, *parameters, *qubits)

    return circuit


class QuimbEngine(TensorNetworkEngine):
    """
    MPS evolution and MPO expectation values via quimb.

    The only engine supporting operator-side conjugation, and so the only one
    that can run :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`'s head/tail
    split or DMRG optimization.
    """

    def __init__(
        self,
        backend: str = "numpy",
        cache: bool = False,
        cache_directory: str | None = "contractions_cache",
    ):
        """
        Args:
            backend: array backend, ``"numpy"``, ``"jax"`` or ``"torch"``.
            cache: reuse contraction paths across calls via cotengra's
                hyper-optimizer instead of contracting greedily.
            cache_directory: where cotengra persists those paths. Created if
                missing.
        """
        if backend == "numpy":
            import numpy

            self.np = numpy
        elif backend == "jax":
            import jax.numpy

            self.np = jax.numpy
        elif backend == "torch":
            import torch

            self.np = torch
        else:
            raise ValueError(
                f"Unsupported quimb backend {backend!r}, expected 'numpy', 'jax' "
                "or 'torch'."
            )

        self.backend = backend
        self.optimizer = (
            ctg.ReusableHyperOptimizer(
                directory=cache_directory,
                minimize="flops",
                max_repeats=128,
                progbar=False,
            )
            if cache
            else "greedy"
        )

    def _to_backend(self, mpo):
        """Move an MPO's arrays onto the engine's backend, when it needs it."""
        if self.backend == "torch":
            mpo.apply_to_arrays(self.np.as_tensor)
        return mpo

    def PauliExp(self, pauli_string: str, theta: float):
        """
        The MPO for ``exp(-i theta/2 P)``, from Euler's formula
        ``cos(theta/2) I - i sin(theta/2) P``.
        """
        identity = MPO_identity(len(pauli_string), phys_dim=2)
        pauli = MPO_product_operator(
            [PAULI_GATES[label.upper()] for label in pauli_string]
        )

        if self.backend == "torch":
            self._to_backend(pauli)
            self._to_backend(identity)
            theta = self.np.as_tensor(theta)

        return (self.np.cos(theta / 2) * identity).add_MPO(
            -1j * self.np.sin(theta / 2) * pauli
        )

    def build_circuit_mps(
        self,
        n: int,
        initial_state_amplitudes: Any,
        initial_state_circuit: Any,
        max_bond_dimension: int | None = None,
    ):
        """Build the MPS by running ``initial_state_circuit`` through quimb."""
        if initial_state_circuit is None:
            raise NotImplementedError(
                "QuimbEngine builds its MPS from a qibo circuit; pass "
                "initial_state_circuit rather than initial_state_amplitudes."
            )
        return _qibo_circuit_to_quimb(
            nqubits=n,
            qibo_circ=initial_state_circuit,
            max_bond=max_bond_dimension,
            to_backend=self.np.asarray,
        ).psi

    def pauli_mpo(self, pauli_string: str | object):
        """Build the MPO for a Pauli string."""
        mpo = MPO_product_operator(
            [PAULI_GATES[label.upper()] for label in pauli_string]
        )
        mpo.add_tag("MPO")
        return self._to_backend(mpo)

    def expval(
        self, state_circuit: MatrixProductState, operator: MatrixProductOperator
    ):
        """
        Expectation value of ``operator`` on ``state_circuit``, normalised.

        Truncation costs the state its unit norm, so the contraction is divided
        by it; the norm is also kept on ``self.norm`` for callers that want it.

        The operator MPO has upper (output) index ``k{i}`` and lower (input)
        index ``b{i}``. The ket must feed the operator's input, so it is
        reindexed ``k -> b`` while the conjugated bra keeps ``k{i}`` to meet the
        output. Reindexing the bra instead would evaluate ``<psi|O^T|psi>``,
        which differs for any operator with an odd number of Y's.
        """
        self.norm = state_circuit.norm(squared=True).real
        ket = state_circuit.reindex({f"k{i}": f"b{i}" for i in range(state_circuit.L)})
        bra = state_circuit.H
        return (bra & operator & ket).contract(
            backend=self.backend, optimize=self.optimizer
        ).real / self.norm

    def pauli_rot(
        self,
        state_circuit: MatrixProductState,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """Apply ``exp(-i angle/2 generator)`` to the state in place, then compress."""
        rotation = self._to_backend(self.PauliExp(generator, angle))
        state_circuit.gate_with_mpo(rotation, inplace=True, max_bond=max_bond_dimension)

    def conjugate_operator(
        self,
        operator: MatrixProductOperator,
        generator: str,
        angle: float,
        max_bond_dimension: int,
    ):
        """
        Heisenberg-conjugate ``operator`` by ``R = exp(-i angle/2 generator)``,
        returning ``R^dag . operator . R``.

        Folds the rotation into the observable so that
        ``<R psi| O |R psi> = <psi| R^dag O R |psi>``. The generators are Pauli
        strings, so ``R(angle)^dag == R(-angle)``, and each MPO-MPO product is
        compressed to ``max_bond_dimension``.

        Even at ``max_bond_dimension=None`` quimb applies its default SVD cutoff,
        and the exact operator rank of a scrambled observable grows
        exponentially, so folding a long tail accrues truncation error. Keeping
        this exact is only affordable for modest tail lengths; see
        :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.tail_truncation` to
        quantify it.
        """
        rotation = self._to_backend(self.PauliExp(generator, angle))
        rotation_dag = self._to_backend(self.PauliExp(generator, -angle))

        operator = rotation_dag.apply(
            operator, compress=True, max_bond=max_bond_dimension
        )
        return operator.apply(rotation, compress=True, max_bond=max_bond_dimension)
