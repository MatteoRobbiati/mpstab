"""
Compare the head syntheses on a device, the numbers of the guide
``docs/guides/edge_restricted_synthesis.rst``.

    uv run python examples/compare_head_syntheses.py [--nqubits 16] [--steps 8] [--cut 0.8]

For each ansatz, the head holding the first ``cut`` of the dressed rotations
is built by every synthesis and by the original circuit up to the cut, then
compiled by Qiskit (optimization level 3, basis R + CZ, the IQM natives) on a
line and on a grid. The grid is placed as a snake (boustrophedon), so that
consecutive qubits of the chain are neighbours, and the coupling graph is given
to the syntheses in that numbering. The coupler-restricted heads keep their
placement; the others are laid out and routed by Qiskit.

Columns: two-qubit gates (CZ) / depth counting two-qubit gates only, after
compilation; ``raw`` is the CNOT count of the synthesised head before
compilation, ``edge`` whether all of them are on couplers. Needs ``qiskit``.
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

from mpstab.evolutors.hsynthsmpo import HSynthSMPO  # noqa: E402
from mpstab.models.ansatze import (  # noqa: E402
    HardwareEfficientBrickwork,
    KickedIsing,
    ParametricKickedIsing,
    TrotterIsing,
)
from mpstab.qpu.synthesis import (  # noqa: E402
    EdgeGreedySynthesis,
    ExtractionSynthesis,
    RustiqSynthesis,
    SteinerSynthesis,
)


def snake_grid(nqubits):
    """Edges of the most square grid, numbered along a snake path."""
    rows = max(1, int(nqubits**0.5))
    cols = -(-nqubits // rows)
    label = {}
    for r in range(rows):
        cells = [(r, c) for c in range(cols)]
        for cell in cells if r % 2 == 0 else cells[::-1]:
            label[cell] = len(label)
    edges = set()
    for (r, c), i in label.items():
        for other in ((r + 1, c), (r, c + 1)):
            j = label.get(other)
            if j is not None and i < nqubits and j < nqubits:
                edges.add(tuple(sorted((i, j))))
    return sorted(edges)


def original_prefix(hs, cut_index):
    """The original circuit up to its ``cut_index``-th magic gate: the head with
    no resynthesis, the reference a synthesis has to beat (only its gates are
    counted here, so no residual is needed)."""
    from copy import deepcopy

    from qibo import Circuit

    magic = sorted(hs.magic_gates, key=lambda item: item[0])[:cut_index]
    cliffords = [g for g in hs.clifford_circuit.queue if type(g).__name__ != "M"]
    circuit, placed = Circuit(hs.nqubits), 0
    for position in range(len(cliffords) + 1):
        while placed < len(magic) and magic[placed][0] == position:
            circuit.add(deepcopy(magic[placed][1]))
            placed += 1
        if placed == len(magic) or position == len(cliffords):
            break
        circuit.add(deepcopy(cliffords[position]))
    return circuit


def qiskit_cost(circuit, edges, keep_layout, seed=0):
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap

    names = {"cx": "cx", "cz": "cz", "h": "h", "s": "s", "sdg": "sdg", "sx": "sx", "sxdg": "sxdg",
             "x": "x", "y": "y", "z": "z", "rx": "rx", "ry": "ry", "rz": "rz"}
    qc = QuantumCircuit(circuit.nqubits)
    for gate in circuit.queue:
        name = gate.name.lower()
        if name == "gpi2":
            qc.r(np.pi / 2, float(gate.parameters[0]), gate.qubits[0])
        else:
            getattr(qc, names[name])(*[float(p) for p in gate.parameters], *gate.qubits)
    both = [list(e) for e in edges] + [[b, a] for a, b in edges]
    out = transpile(qc, coupling_map=CouplingMap(both), basis_gates=["r", "cz"], optimization_level=3,
                    seed_transpiler=seed, initial_layout=list(range(circuit.nqubits)) if keep_layout else None)
    return out.count_ops().get("cz", 0), out.depth(lambda i: i.operation.num_qubits == 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nqubits", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cut", type=float, default=0.8)
    args = parser.parse_args()
    n, steps = args.nqubits, args.steps

    import numpy as np

    np.random.seed(0)
    ansatze = {
        "kicked Ising": KickedIsing(nqubits=n, n_steps=steps),
        "parametric kicked Ising": ParametricKickedIsing(nqubits=n, n_steps=steps, delta_h=0.1),
        "Trotter Ising": TrotterIsing(nqubits=n, n_steps=steps),
        "brickwork": HardwareEfficientBrickwork(nqubits=n, nlayers=max(1, steps * 3 // 4)),
    }
    devices = {"line": [(q, q + 1) for q in range(n - 1)], "grid": snake_grid(n)}
    for device, edges in devices.items():
        syntheses = {
            "original": "original",
            "rustiq": RustiqSynthesis(),
            "extraction": ExtractionSynthesis(connectivity=edges),
            "steiner": SteinerSynthesis(connectivity=edges),
            "edge_greedy": EdgeGreedySynthesis(connectivity=edges),
        }
        print(f"\n{device}, {n} qubits, head = {args.cut:.0%} of the rotations: CZ / 2q depth (raw CNOTs, on couplers?)")
        print(f"{'':26s}" + "".join(f"{name:>22s}" for name in syntheses))
        for label, ansatz in ansatze.items():
            hs = HSynthSMPO.rotations_only(ansatz)
            cut_index = int(round(args.cut * len(hs.magic_gates)))
            row = f"{label:26s}"
            for name, synthesis in syntheses.items():
                if name == "original":
                    circuit = original_prefix(hs, cut_index)
                else:
                    circuit = hs.resynthesize_head(cut_index, synthesis=synthesis).circuit
                aware = name in ("steiner", "edge_greedy")
                cz, depth = qiskit_cost(circuit, edges, keep_layout=aware)
                two_qubit = [g for g in circuit.queue if len(g.qubits) == 2]
                on_edges = all(tuple(sorted(g.qubits)) in set(edges) for g in two_qubit)
                row += f"{cz:>9d} / {depth:<3d}({len(two_qubit):>3d},{'y' if on_edges else 'n'})"
            print(row)


if __name__ == "__main__":
    main()
