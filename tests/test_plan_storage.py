"""
Device-facing helpers of `mpstab.quantum_hardware`: the OpenQASM export of a
plan's circuits (`circuit_to_qasm`, `plan_to_qasm`) and the output folder the
circuits and shots are saved to (`HSynthSMPO.plan`,
`mpstab.quantum_hardware.storage`), which is all a device run needs.
"""

import json

import numpy as np
import pytest
from qibo import Circuit, gates, get_backend, set_backend

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.quantum_hardware import (
    circuit_to_qasm,
    estimate,
    plan_to_qasm,
)
from mpstab.quantum_hardware.backend import QELIB1_GATES

set_backend("numpy")


def _hs(nqubits=4, nlayers=2, seed=0):
    np.random.seed(seed)
    return HSynthSMPO(ansatz=HardwareEfficient(nqubits=nqubits, nlayers=nlayers))


def _probabilities(circuit):
    bare = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        if not isinstance(gate, gates.M):
            bare.add(gate)
    return np.abs(np.asarray(bare().state())) ** 2


def _assert_same_distribution(circuit, text):
    # Probabilities rather than amplitudes, since u3 drops a global phase.
    np.testing.assert_allclose(
        _probabilities(Circuit.from_qasm(text)), _probabilities(circuit), atol=1e-9
    )


def _run_folder(folder):
    # What a device backend does: only the QASM files and shots.json.
    shots = json.loads((folder / "shots.json").read_text())
    return [
        dict(
            Circuit.from_qasm((folder / "planned_circuits" / name).read_text())(
                nshots=count
            ).frequencies()
        )
        for name, count in shots.items()
    ]


# ---------------------------------------------------------------------------
# OpenQASM export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pauli", "shadows"])
def test_plan_to_qasm_matches_every_circuit(method):
    hs = _hs()
    plan = hs.plan("XYZI", 4, method=method, n_shots=200, seed=0)
    texts = plan_to_qasm(plan)
    assert len(texts) == len(plan.circuits)
    for circuit, text in zip(plan.circuits, texts):
        _assert_same_distribution(circuit, text)


def test_circuit_to_qasm_handles_every_measurement_basis():
    # The basis changes qibo adds for M(basis=...) are Unitary gates, which
    # qibo's own to_qasm() rejects.
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RY(2, theta=0.4))
    circuit.add(gates.M(0, 1, 2, basis=[gates.X, gates.Y, gates.Z]))
    text = circuit_to_qasm(circuit)
    assert "u3(" in text
    assert text.count("measure") == 3
    _assert_same_distribution(circuit, text)


def test_circuit_to_qasm_writes_only_qelib1_gates():
    # rustiq heads contain SX/SXDG, whose names qiskit's default qelib1.inc
    # lacks: they must come out as u3, and the text must still match.
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.SX(0))
    circuit.add(gates.SXDG(1))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.SDG(1))
    circuit.add(gates.M(0, 1, basis=[gates.X, gates.Y]))
    text = circuit_to_qasm(circuit)
    names = {line.split("(")[0].split(" ")[0] for line in text.splitlines()[4:]}
    assert names <= QELIB1_GATES | {"measure"}, names
    assert "sx" not in names
    _assert_same_distribution(circuit, text)


@pytest.mark.parametrize(
    "gate", [gates.Unitary(np.eye(4), 0, 1), gates.SWAP(0, 1)], ids=["unitary", "swap"]
)
def test_circuit_to_qasm_rejects_a_non_qelib1_two_qubit_gate(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    with pytest.raises(ValueError, match="one-qubit"):
        circuit_to_qasm(circuit)


# ---------------------------------------------------------------------------
# The output folder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["pauli", "shadows", "tnice"])
def test_plan_writes_the_output_folder(tmp_path, method):
    hs = _hs()
    plan = hs.plan("XZIZ", 3, method=method, n_shots=200, seed=0, out_folder=tmp_path)

    shots = json.loads((tmp_path / "shots.json").read_text())
    assert list(shots.values()) == list(plan.shots)
    texts = [(tmp_path / "planned_circuits" / name).read_text() for name in shots]
    assert texts == plan_to_qasm(plan)


def test_saved_file_names_sort_in_plan_order(tmp_path):
    hs = _hs()
    hs.plan("ZZZZ", 2, method="shadows", n_shots=12, seed=0, out_folder=tmp_path)
    names = list(json.loads((tmp_path / "shots.json").read_text()))
    assert len(names) == 12
    assert names == sorted(names)
    assert names == sorted(p.name for p in (tmp_path / "planned_circuits").iterdir())


def test_plan_refuses_to_overwrite_saved_circuits(tmp_path):
    # A rebuilt plan can have other circuits: counts already measured from the
    # saved ones would no longer match.
    hs = _hs()
    hs.plan("ZZZZ", 2, method="pauli", n_shots=100, seed=0, out_folder=tmp_path)
    with pytest.raises(FileExistsError):
        hs.plan("ZZZZ", 2, method="pauli", n_shots=100, seed=0, out_folder=tmp_path)


def test_plan_needs_a_seed_to_save(tmp_path):
    # The plan is not saved: without a seed it could not be rebuilt for estimate.
    hs = _hs()
    with pytest.raises(ValueError, match="seed"):
        hs.plan("ZZZZ", 2, method="shadows", n_shots=10, out_folder=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("method", ["pauli", "shadows", "tnice"])
def test_saved_circuits_give_the_expectation_value(tmp_path, method):
    # The counts come from the saved files only, as on a device. An asymmetric
    # observable, so a wrong bit order or circuit order would show.
    hs = _hs()
    exact = hs.expectation_at_cut("XZIZ", 3, method="simulation").value
    plan = hs.plan("XZIZ", 3, method=method, n_shots=8000, seed=0, out_folder=tmp_path)

    get_backend().set_seed(1)
    kwargs = {"seed": 0} if method == "tnice" else {}
    result = estimate(plan, _run_folder(tmp_path), **kwargs)
    assert abs(result.value - exact) < 5 * max(result.stderr, 1e-3)
