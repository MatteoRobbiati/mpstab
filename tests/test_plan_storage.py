"""
Device-facing helpers of `mpstab.qpu`: the OpenQASM export of a
plan's circuits (`circuit_to_qasm`, `plan_to_qasm`) and the output folder the
circuits and shots are saved to (`HSynthSMPO.plan`,
`mpstab.qpu.execution.storage`), which is all a device run needs.
"""

import json
from contextlib import chdir

import numpy as np
import pytest
from qibo import Circuit, gates, get_backend, set_backend

from mpstab.evolutors.hsynthsmpo import HSynthSMPO
from mpstab.models.ansatze import HardwareEfficient
from mpstab.qpu import circuit_to_qasm, estimate, plan_to_qasm
from mpstab.qpu.execution import QELIB1_GATES

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
    plan = hs.plan("XYZI", 4, method=method, n_shots=200, seed=0, out_folder=False)
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
    # The default folder needs one too.
    hs = _hs()
    with chdir(tmp_path):
        with pytest.raises(ValueError, match="seed"):
            hs.plan("ZZZZ", 2, method="shadows", n_shots=10, out_folder=tmp_path)
        with pytest.raises(ValueError, match="seed"):
            hs.plan("ZZZZ", 2, method="shadows", n_shots=10)
    assert not list(tmp_path.iterdir())


def test_plan_saves_to_a_timestamped_folder_by_default(tmp_path, capsys):
    hs = _hs()
    with chdir(tmp_path):
        plan = hs.plan("ZZZZ", 2, method="shadows", n_shots=12, seed=0)
        again = hs.plan("ZZZZ", 2, method="shadows", n_shots=12, seed=0)

    assert [p.name for p in tmp_path.iterdir()] == ["planned_runs"]
    folders = sorted((tmp_path / "planned_runs").iterdir())
    assert len(folders) == 2  # a second call does not hit the first folder
    assert all(f.name.startswith("plan_shadows_cut2_") for f in folders)
    for folder, built in zip(folders, (plan, again)):
        shots = json.loads((folder / "shots.json").read_text())
        assert list(shots.values()) == list(built.shots)
    printed = capsys.readouterr().out
    assert all(folder.name in printed for folder in folders)


def test_plan_with_out_folder_false_saves_nothing(tmp_path):
    # No seed needed either: nothing has to be rebuilt.
    hs = _hs()
    with chdir(tmp_path):
        plan = hs.plan("ZZZZ", 2, method="pauli", n_shots=100, out_folder=False)
    assert sum(plan.shots) == 100
    assert not list(tmp_path.iterdir())


def test_expectation_at_cut_saves_nothing(tmp_path):
    hs = _hs()
    with chdir(tmp_path):
        hs.expectation_at_cut("ZZZZ", 2, method="shadows", n_shots=20, seed=0)
        hs.expectation_at_cut("ZZZZ", 2, method="pauli", n_shots=100)
    assert not list(tmp_path.iterdir())
