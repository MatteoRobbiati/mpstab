"""Resynthesis of Pauli rotations into native gates via Qiskit's Rustiq backend.

This module isolates the optional ``qiskit`` dependency (imported lazily inside
the functions) exactly as :mod:`mpstab.evolutors.optimization` isolates the DMRG
machinery. It is only needed by :class:`mpstab.evolutors.hsynthsmpo.HSynthSMPO`.

The pipeline mirrors the sibling project ``hsmpo4transpilation``: a list of
(Pauli-string, angle) rotations is expressed as Qiskit ``PauliEvolutionGate``s and
synthesized with the ``"rustiq"`` high-level-synthesis plugin (bundled inside
qiskit, no separate ``rustiq`` package required). The resulting circuit is then
translated back into a qibo :class:`~qibo.Circuit` so it can be fed to mpstab's
tensor-network engines.
"""

from __future__ import annotations

from typing import List, Tuple

from qibo import Circuit, gates

# Inverse of the qibo -> quimb GATE_MAP in
# mpstab.engines.tensor_networks.quimb. Maps the (small) gate vocabulary that
# Rustiq emits back onto qibo gate classes. Parametrized single-qubit rotations
# carry a single angle; everything else is a fixed Clifford/Pauli gate.
_PARAMETRIZED_1Q = {
    "rx": gates.RX,
    "ry": gates.RY,
    "rz": gates.RZ,
}

_FIXED_1Q = {
    "h": gates.H,
    "x": gates.X,
    "y": gates.Y,
    "z": gates.Z,
    "s": gates.S,
    "sdg": gates.SDG,
    "sx": gates.SX,
    "sxdg": gates.SXDG,
    "t": gates.T,
    "tdg": gates.TDG,
    "id": gates.I,
}

_FIXED_2Q = {
    "cx": gates.CNOT,
    "cy": gates.CY,
    "cz": gates.CZ,
    "swap": gates.SWAP,
}


def synthesize_pauli_rotations(
    rotations: List[Tuple[str, float]],
    nqubits: int,
    preserve_order: bool = True,
    upto_clifford: bool = False,
):
    """
    Synthesize a sequence of Pauli rotations using Qiskit's Rustiq backend.

    Args:
        rotations: List of ``(pauli_string, angle)`` tuples. ``pauli_string`` is
            written qubit-0-leftmost (qibo convention).
        nqubits: Number of qubits.
        preserve_order: Whether Rustiq must preserve the rotation order. Required
            for correctness whenever the rotations do not mutually commute.
        upto_clifford: Whether to synthesize only up to a trailing Clifford. For
            exact expectation values this must stay ``False`` (the default here).

    Returns:
        A Qiskit ``QuantumCircuit`` with the synthesized rotations.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.passes import HighLevelSynthesis
    from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig

    qc = QuantumCircuit(nqubits)
    for pauli, angle in rotations:
        # Qiskit reads Pauli strings right-to-left, qibo writes left-to-right.
        op = SparsePauliOp.from_list([(pauli[::-1], 1.0)])
        qc.append(PauliEvolutionGate(op, time=angle / 2), range(nqubits))

    hls = HLSConfig(
        PauliEvolution=[
            (
                "rustiq",
                {
                    "preserve_order": preserve_order,
                    "upto_clifford": upto_clifford,
                },
            )
        ]
    )
    return HighLevelSynthesis(hls_config=hls)(qc)


def qiskit_circuit_to_qibo(qiskit_circuit, nqubits: int) -> Circuit:
    """
    Convert a (Rustiq-synthesized) Qiskit circuit into a qibo Circuit.

    Only the gate vocabulary Rustiq emits is supported; any other gate raises a
    ``ValueError`` (mirroring the boundary check in ``_qibo_circuit_to_quimb``).
    No qubit routing is introduced during synthesis (no coupling map is passed to
    ``HighLevelSynthesis``), so qubit indices carry over unchanged.
    """
    circ = Circuit(nqubits)

    for instruction in qiskit_circuit.data:
        operation = instruction.operation
        name = operation.name

        if name in ("barrier", "measure"):
            continue

        qubits = [qiskit_circuit.find_bit(q).index for q in instruction.qubits]
        params = [float(p) for p in operation.params]

        if name in _PARAMETRIZED_1Q:
            circ.add(_PARAMETRIZED_1Q[name](qubits[0], theta=params[0]))
        elif name in _FIXED_1Q:
            circ.add(_FIXED_1Q[name](qubits[0]))
        elif name in _FIXED_2Q:
            circ.add(_FIXED_2Q[name](*qubits))
        else:
            raise ValueError(
                f"Gate '{name}' produced by synthesis is not supported by the "
                "qiskit -> qibo converter."
            )

    return circ


def count_two_qubit_gates_qibo(circuit: Circuit) -> int:
    """Count the 2-qubit gates in a qibo circuit."""
    return sum(
        1
        for gate in circuit.queue
        if len(set(gate.target_qubits) | set(gate.control_qubits)) == 2
    )


def count_two_qubit_gates_qiskit(qiskit_circuit) -> int:
    """Count the 2-qubit gates in a Qiskit circuit."""
    return sum(
        1 for instruction in qiskit_circuit.data if instruction.operation.num_qubits == 2
    )
