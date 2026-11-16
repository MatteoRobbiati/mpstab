"""Tests for mpstab.noise: Pauli and coherent noise in the HSMPO frame."""

from functools import reduce

import numpy as np
import pytest
import stim
from qibo import Circuit, gates
from qibo.noise import AmplitudeDampingError, DepolarizingError, NoiseModel, PauliError

from mpstab import HSMPO
from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.noise import (
    CoherentRotation,
    NoiseEvent,
    PauliChannel,
    compile_noise,
    from_qibo_noise_model,
    noisy_expectation,
)
from mpstab.pauli import PAULI_MATRICES

OBSERVABLE = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZI": 0.3}


def _matrix(pauli: str) -> np.ndarray:
    return reduce(np.kron, [PAULI_MATRICES[label] for label in pauli])


def _exact_expectation(circuit: Circuit, events, observable: dict) -> float:
    """``<observable>`` on a qibo density matrix with the events inserted."""
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)

    noisy = Circuit(circuit.nqubits, density_matrix=True)

    def add_events(index):
        for event in by_gate.get(index, ()):
            if isinstance(event, PauliChannel):
                noisy.add(
                    gates.PauliNoiseChannel(event.qubits, list(event.probs.items()))
                )
            else:
                support = [q for q, p in enumerate(event.generator) if p != "I"]
                local = "".join(event.generator[q] for q in support)
                unitary = np.cos(event.angle / 2) * np.eye(
                    2 ** len(support)
                ) - 1j * np.sin(event.angle / 2) * _matrix(local)
                noisy.add(gates.Unitary(unitary, *support))

    add_events(-1)
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        add_events(index)

    rho = noisy().state()
    return sum(c * np.real(np.trace(rho @ _matrix(p))) for p, c in observable.items())


def _hardware_efficient():
    np.random.seed(3)
    return HardwareEfficient(nqubits=5, nlayers=2)


def _kicked_ising():
    return KickedIsing(nqubits=5, n_steps=2, theta_h=0.8, theta_z=0.3, theta_j=0.4)


ANSATZE = [_hardware_efficient, _kicked_ising]


def _depolarizing(circuit, p):
    return [
        PauliChannel(index, (q,), {label: p / 3 for label in "XYZ"})
        for index, gate in enumerate(circuit.queue)
        for q in gate.qubits
    ]


def _correlated(circuit, rng):
    """A random correlated channel after every two-qubit gate, local noise elsewhere."""
    events = []
    for index, gate in enumerate(circuit.queue):
        if len(gate.qubits) == 2:
            probs = {"ZZ": rng.uniform(0, 0.02), "XY": rng.uniform(0, 0.01)}
            probs["IX"] = rng.uniform(0, 0.01)
        else:
            probs = {"Z": rng.uniform(0, 0.02), "X": rng.uniform(0, 0.005)}
        events.append(PauliChannel(index, gate.qubits, probs))
    return events


def _coherent(circuit, rng):
    """ZZ crosstalk after every two-qubit gate and an over-rotation after every RY/RX."""
    n = circuit.nqubits
    events = []
    for index, gate in enumerate(circuit.queue):
        labels = ["I"] * n
        if len(gate.qubits) == 2:
            for q in gate.qubits:
                labels[q] = "Z"
        elif gate.name in ("rx", "ry"):
            labels[gate.qubits[0]] = gate.name[1].upper()
        else:
            continue
        events.append(CoherentRotation(index, "".join(labels), rng.uniform(-0.1, 0.1)))
    return events


# -- Flip vectors ---------------------------------------------------------------


def _hand_circuit():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.RZ(0, theta=0.3))  # magic, dressed by H: generator XI
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(1, theta=0.5))  # magic, dressed by CNOT.H: generator IX
    return circuit


def test_flip_vectors_by_hand():
    events = [
        PauliChannel(0, (0,), {"X": 0.1}),  # dressed ZI: anticommutes with XI
        PauliChannel(2, (1,), {"Z": 0.1}),  # dressed XZ: after XI, flips IX
        PauliChannel(-1, (0, 1), {"ZZ": 0.1}),  # undressed: flips XI and IX
    ]
    compiled = compile_noise(HSMPO(_hand_circuit()), events)

    assert compiled.generators == ["XI", "IX"]
    # Faults are stored in circuit order, so the initial-state one comes first.
    assert compiled.fault_paulis == ["ZZ", "ZI", "XZ"]
    np.testing.assert_array_equal(
        compiled.flips, [[True, True], [True, False], [False, True]]
    )
    # Z on the control commutes with the CNOT, so ZI is dressed to XI by the H.
    assert compiled.dress_observable("ZI") == ("XI", 1)
    np.testing.assert_array_equal(
        compiled.observable_flips(["XI"]), [[True], [True], [False]]
    )


def test_flips_match_prefix_backpropagation():
    """The single tableau pass agrees with dressing every fault through its own prefix."""
    circuit = _kicked_ising().circuit
    hsmpo = HSMPO(circuit)
    events = _depolarizing(circuit, 0.03)
    compiled = compile_noise(hsmpo, events)
    generators = [stim.PauliString(g) for g in compiled.generators]

    n_cliffords = n_magic = 0
    row = 0
    for index, gate in enumerate(circuit.queue):
        if gate.clifford:
            n_cliffords += 1
        else:
            n_magic += 1
        prefix = hsmpo._clifford_subcircuit(hsmpo.clifford_circuit, n_cliffords)
        for q in gate.qubits:
            for label in "XYZ":
                fault = "".join(
                    label if i == q else "I" for i in range(circuit.nqubits)
                )
                dressed, _ = hsmpo.stab_engine.backpropagate(fault, prefix)
                expected = [
                    j >= n_magic and not stim.PauliString(dressed).commutes(g)
                    for j, g in enumerate(generators)
                ]
                assert compiled.fault_paulis[row] == dressed
                np.testing.assert_array_equal(compiled.flips[row], expected)
                row += 1
    assert row == len(compiled.fault_paulis)


def test_noiseless_compilation_matches_dressed_rotations():
    hsmpo = HSMPO(_hardware_efficient().circuit)
    compiled = compile_noise(hsmpo, [])
    generators, angles = zip(*hsmpo._dressed_rotations())
    assert compiled.generators == list(generators)
    np.testing.assert_array_equal(compiled.angles, angles)
    assert compiled.flips.shape == (0, len(generators))


# -- Estimator against exact simulation -----------------------------------------


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_zero_noise_reproduces_expectation(make_ansatz):
    hsmpo = HSMPO(make_ansatz())
    circuit = hsmpo.ansatz.circuit
    events = _depolarizing(circuit, 0.0)
    compiled = compile_noise(hsmpo, events)

    result = noisy_expectation(compiled, "ZZIII", n_samples=100, seed=0)
    assert result.mean == hsmpo.expectation("ZZIII")
    assert result.stratum_weights == (1.0, 0.0, 0.0)
    assert result.n_patterns == 1

    result = noisy_expectation(compiled, OBSERVABLE, n_samples=100, seed=0)
    assert result.mean == pytest.approx(hsmpo.expectation(OBSERVABLE), abs=1e-14)


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_uniform_depolarizing(make_ansatz):
    circuit = make_ansatz().circuit
    events = _depolarizing(circuit, 0.02)
    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=1500, seed=1
    )
    exact = _exact_expectation(circuit, events, OBSERVABLE)
    assert result.stratum_weights[2] > 0.05  # the sampled stratum matters
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
    np.testing.assert_allclose(result.fidelities, 1.0, atol=1e-10)


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_correlated_two_qubit_channel(make_ansatz):
    circuit = make_ansatz().circuit
    events = _correlated(circuit, np.random.default_rng(2))
    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=1500, seed=3
    )
    exact = _exact_expectation(circuit, events, OBSERVABLE)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_coherent_only_is_exact(make_ansatz):
    circuit = make_ansatz().circuit
    events = _coherent(circuit, np.random.default_rng(4))
    hsmpo = HSMPO(circuit)
    # quimb's default SVD cutoff drops the tiny Schmidt values many small
    # rotations leave behind, worth ~1e-7 here; disable it for an exact check.
    hsmpo.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
    result = noisy_expectation(compile_noise(hsmpo, events), OBSERVABLE, n_samples=0)
    exact = _exact_expectation(circuit, events, OBSERVABLE)
    assert result.stratum_weights == (1.0, 0.0, 0.0)
    assert result.stderr == 0.0
    assert result.mean == pytest.approx(exact, abs=1e-10)
    assert abs(result.mean - hsmpo.expectation(OBSERVABLE)) > 1e-4  # noise matters


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_coherent_and_pauli(make_ansatz):
    circuit = make_ansatz().circuit
    rng = np.random.default_rng(5)
    events = _coherent(circuit, rng) + _correlated(circuit, rng)
    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=1500, seed=6
    )
    exact = _exact_expectation(circuit, events, OBSERVABLE)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_six_qubits_noise_on_initial_state():
    circuit = KickedIsing(nqubits=6, n_steps=2, theta_h=0.6, theta_z=0.2).circuit
    events = [PauliChannel(-1, (q,), {"X": 0.05}) for q in range(6)]
    events += _depolarizing(circuit, 0.01)
    observable = {"ZIIIIZ": 1.0, "IIXXII": 0.5}
    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), observable, n_samples=1500, seed=7
    )
    exact = _exact_expectation(circuit, events, observable)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_truncated_strata_bias_is_bounded():
    circuit = _hardware_efficient().circuit
    events = _depolarizing(circuit, 0.02)
    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=0
    )
    exact = _exact_expectation(circuit, events, OBSERVABLE)
    bound = result.stratum_weights[2] * sum(abs(c) for c in OBSERVABLE.values())
    assert result.n_samples == 0
    assert abs(result.mean - exact) <= bound
    assert sum(result.stratum_weights) == pytest.approx(1.0, abs=1e-14)


def test_native_engine_matches_quimb():
    """The engine's default rotation hooks give the same estimate as quimb's MPO cache."""
    circuit = _kicked_ising().circuit
    events = _depolarizing(circuit, 0.02)
    quimb = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=200, seed=9
    )
    native = HSMPO(circuit)
    native.set_engines(tn_engine=NativeTensorNetworkEngine())
    result = noisy_expectation(
        compile_noise(native, events), OBSERVABLE, n_samples=200, seed=9
    )
    assert result.n_patterns == quimb.n_patterns
    assert result.mean == pytest.approx(quimb.mean, abs=1e-10)


# -- qibo noise models ----------------------------------------------------------


def test_from_qibo_noise_model():
    circuit = _kicked_ising().circuit
    model = NoiseModel()
    model.add(PauliError([("X", 0.01), ("Z", 0.02)]), gates.RX)
    model.add(DepolarizingError(0.03), gates.CNOT)
    events = from_qibo_noise_model(model, circuit)

    cnots = [i for i, g in enumerate(circuit.queue) if g.name == "cx"]
    assert [e.after_gate for e in events if len(e.qubits) == 2] == cnots
    assert all(len(e.probs) == 15 for e in events if len(e.qubits) == 2)

    density = Circuit(circuit.nqubits, density_matrix=True)
    for gate in circuit.queue:
        density.add(gate)
    rho = model.apply(density)().state()
    exact = sum(c * np.real(np.trace(rho @ _matrix(p))) for p, c in OBSERVABLE.items())

    result = noisy_expectation(
        compile_noise(HSMPO(circuit), events), OBSERVABLE, n_samples=1500, seed=8
    )
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_unsupported_qibo_channel():
    model = NoiseModel()
    model.add(AmplitudeDampingError(0.01), gates.RX)
    with pytest.raises(NotImplementedError, match="AmplitudeDampingChannel"):
        from_qibo_noise_model(model, _kicked_ising().circuit)


# -- Validation -----------------------------------------------------------------


def test_invalid_events():
    hsmpo = HSMPO(_hand_circuit())
    with pytest.raises(NotImplementedError, match="Kraus"):
        compile_noise(hsmpo, [NoiseEvent(0)])
    with pytest.raises(ValueError, match="gates 0 to 3"):
        compile_noise(hsmpo, [PauliChannel(4, (0,), {"X": 0.1})])
    with pytest.raises(ValueError):
        compile_noise(hsmpo, [CoherentRotation(0, "ZZZ", 0.1)])
    with pytest.raises(ValueError, match="sum"):
        PauliChannel(0, (0,), {"X": 0.6, "Z": 0.6})
    with pytest.raises(ValueError, match="identity"):
        PauliChannel(0, (0, 1), {"II": 0.1})


# -- Two-qubit Pauli rotations ---------------------------------------------------


def _rzz_circuit():
    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.RY(1, theta=0.4))
    circuit.add(gates.RZZ(0, 1, theta=0.7))
    circuit.add(gates.RXX(1, 2, theta=-0.3))
    circuit.add(gates.RYY(0, 2, theta=1.1))
    circuit.add(gates.CNOT(1, 2))
    return circuit


def test_two_qubit_rotations_as_magic_gates():
    circuit = _rzz_circuit()
    state = circuit().state()
    for pauli in ("ZZI", "XIY", "IYX"):
        exact = np.real(state.conj() @ _matrix(pauli) @ state)
        assert HSMPO(circuit).expectation(pauli) == pytest.approx(exact, abs=1e-12)


def test_two_qubit_rotations_replaced_by_cliffords():
    """Replaced RZZ/RXX/RYY land in the Clifford part, which stim must handle."""
    surrogate = HSMPO(_rzz_circuit())
    for seed in range(3):
        np.random.seed(seed)
        expval, partitions = surrogate.expectation_from_partition(
            "ZXY",
            replacement_probability=1.0,
            replacement_method="random",
            return_partitions=True,
        )
        assert not partitions["magic_gates"]
        state = partitions["full_circuit"]().state()
        exact = np.real(state.conj() @ _matrix("ZXY") @ state)
        assert expval == pytest.approx(exact, abs=1e-12)


# -- Documentation examples -----------------------------------------------------


def _docstring_examples():
    """Every ``::`` literal block in the docstrings of mpstab.noise."""
    import inspect
    import textwrap

    import mpstab.noise as noise
    from mpstab import pauli

    blocks = []
    objects = [getattr(noise, name) for name in noise.__all__]
    objects += [pauli.anticommutation, pauli.light_cone]
    for obj in objects:
        lines = (inspect.getdoc(obj) or "").splitlines()
        for i, line in enumerate(lines):
            if not line.rstrip().endswith("::"):
                continue
            body = []
            for following in lines[i + 1 :]:
                if following.strip() and not following.startswith(" "):
                    break
                body.append(following)
            blocks.append(
                pytest.param(textwrap.dedent("\n".join(body)), id=obj.__name__)
            )
    return blocks


@pytest.mark.parametrize("code", _docstring_examples())
def test_docstring_examples_run(code):
    exec(compile(code, "<docstring>", "exec"), {})
