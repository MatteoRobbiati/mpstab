"""Tests for mpstab.models.noise: Pauli, coherent and Kraus noise in the HSMPO frame."""

import inspect
import textwrap

import numpy as np
import pytest
import stim
from qibo import Circuit, gates

import mpstab.models.noise
from mpstab import HSMPO
from mpstab.engines import QuimbEngine
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.models.noise import (
    AmplitudeDampingError,
    CoherentError,
    DepolarizingError,
    KrausError,
    NoiseModel,
    PauliError,
    ReadoutError,
    StatePreparationError,
    ThermalRelaxationError,
)
from mpstab.models.noise.compile import compile_noise
from mpstab.models.noise.events import (
    CoherentRotation,
    KrausChannel,
    NoiseEvent,
    PauliChannel,
)
from mpstab.models.noise.mc_noise_estimator import noisy_expectation
from mpstab.models.noise.model import _qibo_gate
from mpstab.pauli import anticommutation, pauli_light_cone, pauli_matrix

OBSERVABLE = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZI": 0.3}


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
                ) - 1j * np.sin(event.angle / 2) * pauli_matrix(local)
                noisy.add(gates.Unitary(unitary, *support))

    add_events(-1)
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        add_events(index)

    rho = noisy().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


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


def test_restricted_to_by_hand():
    events = [
        PauliChannel(0, (0,), {"X": 0.1}),  # dressed ZI
        PauliChannel(2, (1,), {"Z": 0.2}),  # dressed XZ
        PauliChannel(-1, (0, 1), {"ZZ": 0.3}),  # undressed ZZ
    ]
    compiled = compile_noise(HSMPO(_hand_circuit()), events)

    # IZ: IX anticommutes with it and stays; XI commutes with IZ and IX and
    # drops out. ZI then flips nothing, so its location goes.
    restricted = compiled.restricted_to(["IZ"])
    assert restricted.generators == ["IX"]
    assert restricted.fault_paulis == ["ZZ", "XZ"]
    np.testing.assert_allclose(restricted.location_probs, [0.3, 0.2])
    np.testing.assert_array_equal(restricted.fault_location, [0, 1])
    np.testing.assert_array_equal(restricted.flips, [[True], [True]])

    # XI: neither rotation reaches it, and only ZZ and ZI flip its sign.
    restricted = compiled.restricted_to(["XI"])
    assert restricted.generators == []
    assert restricted.fault_paulis == ["ZZ", "ZI"]
    assert restricted.flips.shape == (2, 0)
    assert compiled.generators == ["XI", "IX"]  # the original is untouched


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_light_cone_matches_unrestricted(make_ansatz):
    """Coherent noise is deterministic, so the two computations must agree."""
    circuit = make_ansatz().circuit
    events = _coherent(circuit, np.random.default_rng(7))
    hsmpo = HSMPO(circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    compiled = compile_noise(hsmpo, events)
    for observable in (OBSERVABLE, "IIIIZ", "XIIII"):
        full = noisy_expectation(compiled, observable, n_samples=0, light_cone=False)
        cut = noisy_expectation(compiled, observable, n_samples=0)
        assert cut.mean == pytest.approx(full.mean, abs=1e-12)


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
    # Commuting rotations are reordered for the kernels, so equal up to rounding.
    unrestricted = noisy_expectation(compiled, "ZZIII", n_samples=0, light_cone=False)
    assert unrestricted.mean == pytest.approx(hsmpo.expectation("ZZIII"), abs=1e-14)
    assert result.mean == pytest.approx(hsmpo.expectation("ZZIII"), abs=1e-14)
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
        exact = np.real(state.conj() @ pauli_matrix(pauli) @ state)
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
        exact = np.real(state.conj() @ pauli_matrix("ZXY") @ state)
        assert expval == pytest.approx(exact, abs=1e-12)


# -- Documentation examples -----------------------------------------------------


def _docstring_examples():
    """Every ``::`` literal block in the docstrings of mpstab.models.noise."""
    blocks = []
    objects = [
        getattr(mpstab.models.noise, name) for name in mpstab.models.noise.__all__
    ]
    objects += [compile_noise, noisy_expectation]
    objects += [PauliChannel, CoherentRotation, KrausChannel]
    objects += [anticommutation, pauli_light_cone]
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


# -- NoiseModel and HSMPO.noisy_expectation -------------------------------------


def _model_expectation_exact(noise, circuit, observable):
    rho = noise.apply(circuit)().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_noise_model_matches_densitypauli_matrix(make_ansatz):

    ansatz = make_ansatz()
    noise = NoiseModel()
    noise.add(PauliError({"X": 0.004, "Y": 0.002, "Z": 0.006}))
    noise.add(DepolarizingError(0.01), gates.CZ)
    noise.add(DepolarizingError(0.01), gates.CNOT)
    noise.add(PauliError({"ZZ": 0.005}), (gates.CZ, gates.CNOT))
    noise.add(CoherentError("ZZ", 0.03), (gates.CZ, gates.CNOT))
    noise.add(CoherentError("X", -0.02), gates.RY, qubits=[0, 2])

    result = HSMPO(ansatz).noisy_expectation(OBSERVABLE, noise, n_samples=1500, seed=4)
    exact = _model_expectation_exact(noise, ansatz.circuit, OBSERVABLE)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_noise_model_matches_events():

    circuit = _kicked_ising().circuit
    p = 0.01
    noise = NoiseModel().add(PauliError({label: p / 3 for label in "XYZ"}))
    assert noise.events(circuit) == _depolarizing(circuit, p)

    hsmpo = HSMPO(circuit)
    new = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=300, seed=1)
    old = noisy_expectation(
        compile_noise(hsmpo, _depolarizing(circuit, p)),
        OBSERVABLE,
        n_samples=300,
        seed=1,
    )
    assert new.mean == pytest.approx(old.mean, abs=1e-12)


def test_noise_model_rules():

    circuit = _hand_circuit()  # H(0), RZ(0), CNOT(0,1), RX(1)
    events = NoiseModel().add(PauliError({"Z": 0.1}), gates.CNOT).events(circuit)
    assert [(e.after_gate, e.qubits) for e in events] == [(2, (0,)), (2, (1,))]

    events = (
        NoiseModel().add(PauliError({"Z": 0.1}), gates.CNOT, qubits=1).events(circuit)
    )
    assert [(e.after_gate, e.qubits) for e in events] == [(2, (1,))]

    # A joint error needs the whole gate inside the qubit filter.
    assert (
        NoiseModel().add(DepolarizingError(0.1), gates.CNOT, qubits=[1]).events(circuit)
        == []
    )
    (event,) = NoiseModel().add(DepolarizingError(0.1), gates.CNOT).events(circuit)
    assert len(event.probs) == 15 and event.probs["XY"] == pytest.approx(0.1 / 16)

    (event,) = NoiseModel().add(CoherentError("XZ", 0.2), gates.CNOT).events(circuit)
    assert (event.after_gate, event.generator, event.angle) == (2, "XZ", 0.2)

    with pytest.raises(ValueError, match="width 2"):
        NoiseModel().add(PauliError({"ZZ": 0.1}), gates.H).events(circuit)
    with pytest.raises(TypeError, match="mpstab error"):
        NoiseModel().add(PauliChannel(0, (0,), {"X": 0.1}))


def test_noise_model_scaled():

    circuit = _hand_circuit()
    noise = (
        NoiseModel().add(PauliError({"X": 0.01})).add(CoherentError("Z", 0.1), gates.RX)
    )
    doubled = noise.scaled(2.0).events(circuit)
    for event, base in zip(doubled, noise.events(circuit)):
        if isinstance(event, PauliChannel):
            assert event.probs["X"] == pytest.approx(2 * base.probs["X"])
        else:
            assert event.angle == pytest.approx(2 * base.angle)
    assert noise.scaled(0.0).events(circuit)[0].probs["X"] == 0.0


def test_noisy_expectation_without_noise():

    hsmpo = HSMPO(_kicked_ising())
    result = hsmpo.noisy_expectation(OBSERVABLE, NoiseModel())
    assert result.mean == pytest.approx(hsmpo.expectation(OBSERVABLE), abs=1e-10)
    assert result.stderr == 0.0


def test_readout_matches_bit_flips_before_measurement():

    circuit = _kicked_ising().circuit
    noise = NoiseModel().add(PauliError({"Z": 0.004, "X": 0.002}))
    noise.add(ReadoutError(0.03)).add(ReadoutError(0.05), qubits=[1])
    observable = {"ZZIII": 0.7, "IZIZI": -0.4, "IIIIZ": 0.3}

    # For Z-basis terms, readout bit flips are X flips right before measuring.
    noisy = noise.apply(circuit)
    for q in range(circuit.nqubits):
        noisy.add(gates.PauliNoiseChannel([q], [("X", 0.03)]))
    noisy.add(gates.PauliNoiseChannel([1], [("X", 0.05)]))
    rho = noisy().state()
    exact = sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )

    result = HSMPO(circuit).noisy_expectation(observable, noise, n_samples=1500, seed=2)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_readout_only_attenuates_terms():

    hsmpo = HSMPO(_hardware_efficient())
    noise = NoiseModel().add(ReadoutError(0.1), qubits=[0, 2])
    np.testing.assert_allclose(noise.readout_attenuation(5), [0.8, 1, 0.8, 1, 1])

    constant, terms = noise.readout_observable(OBSERVABLE, 5)
    assert constant == 0.0
    assert terms == pytest.approx(
        {"ZZIII": 0.7 * 0.8, "IXIXI": -0.4, "IIYZI": 0.3 * 0.8}
    )

    result = hsmpo.noisy_expectation(OBSERVABLE, noise)
    assert result.mean == pytest.approx(hsmpo.expectation(terms), abs=1e-10)
    assert noise.scaled(0.5).readout_attenuation(5)[0] == pytest.approx(0.9)

    with pytest.raises(ValueError, match="without a gate"):
        NoiseModel().add(ReadoutError(0.1), gates.M)


def test_parallel_patterns_match_serial():
    hsmpo = HSMPO(_kicked_ising(), max_bond_dimension=4)
    noise = NoiseModel().add(PauliError({"X": 0.01, "Z": 0.02}))
    serial = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=200, seed=5)
    parallel = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=200, seed=5, n_workers=3
    )
    assert parallel.mean == pytest.approx(serial.mean, abs=1e-12)
    assert parallel.stderr == pytest.approx(serial.stderr, abs=1e-12)
    np.testing.assert_allclose(parallel.fidelities, serial.fidelities, atol=1e-12)


# -- Kraus channels: amplitude damping, thermal relaxation ----------------------


def _events_exact(circuit, events, observable):
    """``<observable>`` on qibo's density matrix with ``events`` inserted."""
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)
    noisy = Circuit(circuit.nqubits, density_matrix=True)
    for event in by_gate.get(-1, ()):
        noisy.add(_qibo_gate(event))
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        for event in by_gate.get(index, ()):
            noisy.add(_qibo_gate(event))
    rho = noisy().state()
    return sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in observable.items()
    )


def _random_kraus(rng, nqubits=2, rank=3):
    """A random channel, from an isometry stacked into Kraus operators."""
    dim = 2**nqubits
    matrix = rng.normal(size=(rank * dim, dim)) + 1j * rng.normal(
        size=(rank * dim, dim)
    )
    isometry, _ = np.linalg.qr(matrix)
    return [isometry[k * dim : (k + 1) * dim] for k in range(rank)]


def test_amplitude_damping_process_matrix():
    gamma = 0.2
    chi = AmplitudeDampingError(gamma)._event(0, (0,), 1).chi
    a, b = (1 + np.sqrt(1 - gamma)) / 2, (1 - np.sqrt(1 - gamma)) / 2
    assert chi[("I", "I")] == pytest.approx(a**2)
    assert chi[("Z", "Z")] == pytest.approx(b**2)
    assert chi[("X", "X")] == pytest.approx(gamma / 4)
    assert chi[("Y", "Y")] == pytest.approx(gamma / 4)
    assert chi[("I", "Z")] == pytest.approx(gamma / 4)
    assert chi[("X", "Y")] == pytest.approx(-1j * gamma / 4)
    for (p, q), value in chi.items():
        assert chi[(q, p)] == pytest.approx(np.conj(value))
    assert sum(chi[(p, p)] for p in "IXYZ").real == pytest.approx(1.0)


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_single_kraus_location_is_exact(make_ansatz):
    """One channel leaves nothing to sample: the k <= 1 strata are the answer."""
    circuit = make_ansatz().circuit
    hsmpo = HSMPO(circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    observable = dict(OBSERVABLE, IIIII=0.2, IIIIZ=0.5)
    rng = np.random.default_rng(8)
    two_qubit = [i for i, g in enumerate(circuit.queue) if len(g.qubits) == 2]
    events = [
        AmplitudeDampingError(0.3)._event(-1, (2,), 5),
        AmplitudeDampingError(0.3)._event(3, (0,), 5),
        ThermalRelaxationError(1.0, 0.6, 0.4, 0.2)._event(
            len(circuit.queue) - 1, (4,), 5
        ),
        KrausChannel(
            two_qubit[0], circuit.queue[two_qubit[0]].qubits, _random_kraus(rng)
        ),
    ]
    for event in events:
        compiled = compile_noise(hsmpo, [event])
        result = noisy_expectation(compiled, observable)
        assert result.n_samples == 0
        if len(compiled.block_probs):
            # Relaxation with T2 <= T1: a Pauli and reset mixture, no coherence.
            assert len(compiled.coherence_weights) == 0
        else:
            assert result.n_transitions > 0
        exact = _events_exact(circuit, [event], observable)
        assert result.mean == pytest.approx(exact, abs=1e-10)


def test_kraus_light_cone_matches_unrestricted():
    circuit = _kicked_ising().circuit
    hsmpo = HSMPO(circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    events = _coherent(circuit, np.random.default_rng(9))
    events.append(AmplitudeDampingError(0.2)._event(4, (1,), 5))
    compiled = compile_noise(hsmpo, events)
    for observable in (OBSERVABLE, "IIIIZ", "ZIIII"):
        full = noisy_expectation(compiled, observable, light_cone=False)
        cut = noisy_expectation(compiled, observable)
        assert cut.mean == pytest.approx(full.mean, abs=1e-12)
        assert cut.mean == pytest.approx(
            _events_exact(circuit, events, pauli_terms_dict(observable)), abs=1e-10
        )


def pauli_terms_dict(observable):
    return {observable: 1.0} if isinstance(observable, str) else observable


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_relaxation_matches_density_matrix(make_ansatz):
    ansatz = make_ansatz()
    noise = NoiseModel()
    noise.add(
        ThermalRelaxationError(t1=1.0, t2=0.8, time=0.01, excited_population=0.05)
    )
    noise.add(AmplitudeDampingError(0.02), (gates.CZ, gates.CNOT))
    noise.add(PauliError({"X": 0.002, "Z": 0.004}))
    result = HSMPO(ansatz).noisy_expectation(OBSERVABLE, noise, n_samples=2000, seed=3)
    exact = _model_expectation_exact(noise, ansatz.circuit, OBSERVABLE)
    assert result.stratum_weights[2] > 0.05  # the sampled stratum matters
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
    noiseless = HSMPO(ansatz).expectation(OBSERVABLE)
    assert abs(exact - noiseless) > 1e-3  # and the noise matters


def test_random_two_qubit_kraus_matches_density_matrix():
    circuit = _hardware_efficient().circuit
    rng = np.random.default_rng(10)
    # A weak random channel: mostly the identity, a little of a random channel.
    noise = NoiseModel()
    weak = [np.sqrt(0.97) * np.eye(4)] + [np.sqrt(0.03) * k for k in _random_kraus(rng)]
    noise.add(KrausError(weak), gates.CZ)
    result = HSMPO(circuit).noisy_expectation(OBSERVABLE, noise, n_samples=2000, seed=4)
    exact = _model_expectation_exact(noise, circuit, OBSERVABLE)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_kraus_validation():
    with pytest.raises(ValueError, match="trace preserving"):
        KrausError([np.eye(2), np.eye(2)])
    with pytest.raises(ValueError, match="2\\*\\*k"):
        KrausError([np.eye(3)])
    with pytest.raises(ValueError, match="2 t1"):
        ThermalRelaxationError(t1=1.0, t2=3.0, time=0.1)
    with pytest.raises(NotImplementedError):
        NoiseModel().add(KrausError([np.eye(2)])).scaled(2.0)
    with pytest.raises(ValueError, match="width 2"):
        NoiseModel().add(KrausError([np.eye(4)]), gates.H).events(_hand_circuit())


def test_new_errors_scaled():
    noise = NoiseModel().add(AmplitudeDampingError(0.01))
    noise.add(ThermalRelaxationError(1.0, 1.0, 0.1)).add(ReadoutError(0.01, 0.03))
    noise.add(StatePreparationError(0.02))
    errors = [rule.error for rule in noise.scaled(2.0).rules]
    assert errors[0].gamma == pytest.approx(0.02)
    assert errors[1].time == pytest.approx(0.2)
    assert (errors[2].p0, errors[2].p1) == pytest.approx((0.02, 0.06))
    assert errors[3].p == pytest.approx(0.04)


def test_pauli_transitions_against_statevectors():
    rng = np.random.default_rng(11)
    engine = QuimbEngine(svd_cutoff=0.0)
    hsmpo = HSMPO(_hardware_efficient(), tn_engine=engine)
    states, vectors = [], []
    for _ in range(2):
        mps = hsmpo.initial_mps()
        vector = np.zeros(32, dtype=complex)
        vector[0] = 1
        for _ in range(6):
            generator = "".join(rng.choice(list("IXYZ"), 5))
            angle = rng.uniform(-1, 1)
            engine.pauli_rot(mps, generator, angle, None)
            matrix = pauli_matrix(generator)
            vector = (
                np.cos(angle / 2) * vector - 1j * np.sin(angle / 2) * matrix @ vector
            )
        states.append(mps)
        vectors.append(vector)
    paulis = ["IIIII", "ZXIYZ", "XIIIX"]
    values = engine.pauli_transitions(states[0], states[1], paulis)
    expected = [vectors[0].conj() @ pauli_matrix(p) @ vectors[1] for p in paulis]
    np.testing.assert_allclose(values, expected, atol=1e-12)
    same = engine.pauli_transitions(states[0], states[0], paulis)
    np.testing.assert_allclose(
        same,
        [vectors[0].conj() @ pauli_matrix(p) @ vectors[0] for p in paulis],
        atol=1e-12,
    )


# -- Crosstalk, idle noise and preparation errors --------------------------------


def _layered_circuit():
    """Five qubits on a line, queued layer by layer; qubit 4 never acts."""
    circuit = Circuit(5)
    circuit.add(gates.H(0))  # 0, moment 0
    circuit.add(gates.CNOT(1, 2))  # 1, moment 0
    circuit.add(gates.RX(3, theta=0.3))  # 2, moment 0
    circuit.add(gates.CNOT(0, 1))  # 3, moment 1
    circuit.add(gates.RY(2, theta=0.5))  # 4, moment 1
    circuit.add(gates.CZ(2, 3))  # 5, moment 2
    return circuit


LINE = [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_crosstalk_rule_events():
    circuit = _layered_circuit()
    events = NoiseModel().add_crosstalk(CoherentError("ZZ", 0.1), LINE, gates.CNOT)
    events = events.events(circuit)
    assert [(e.after_gate, e.generator) for e in events] == [
        (1, "ZZIII"),  # CNOT(1,2): spectators 0 (via 1) and 3 (via 2)
        (1, "IIZZI"),
        (3, "IZZII"),  # CNOT(0,1): spectator 2 (via 1)
    ]
    events = NoiseModel().add_crosstalk(PauliError({"Z": 0.1}), LINE, gates.CZ)
    assert [(e.after_gate, e.qubits) for e in events.events(circuit)] == [
        (5, (1,)),
        (5, (4,)),
    ]
    with pytest.raises(ValueError, match="width 3"):
        NoiseModel().add_crosstalk(PauliError({"ZZZ": 0.1}), LINE).events(circuit)


def test_idle_rule_events():
    circuit = _layered_circuit()
    events = NoiseModel().add_idle(PauliError({"Z": 0.1})).events(circuit)
    # Moment 0: 4 idle; moment 1: 3 and 4 idle; moment 2: 0, 1 and 4 idle.
    assert [(e.after_gate, e.qubits) for e in events] == [
        (-1, (4,)),
        (2, (3,)),
        (-1, (4,)),
        (3, (0,)),
        (3, (1,)),
        (-1, (4,)),
    ]
    events = NoiseModel().add_idle(CoherentError("ZZ", 0.1), coupling=LINE)
    assert [(e.after_gate, e.generator) for e in events.events(circuit)] == [
        (2, "IIIZZ"),  # moment 1: after RX(3)
        (3, "ZZIII"),  # moment 2: after CNOT(0,1)
    ]
    with pytest.raises(ValueError, match="coupling"):
        NoiseModel().add_idle(CoherentError("ZZ", 0.1)).events(circuit)


def test_idle_moments_follow_the_queue():
    """A gate never starts before the one queued ahead of it, so moments never go back."""
    circuit = Circuit(3)
    for _ in range(3):
        circuit.add(gates.H(2))  # 0-2: moments 0-2
    circuit.add(gates.H(0))  # 3: moment 2, not 0
    circuit.add(gates.CNOT(0, 2))  # 4: moment 3
    circuit.add(gates.H(1))  # 5: moment 3
    noise = NoiseModel().add_idle(PauliError({"ZZ": 0.1}), coupling=[(0, 1)])
    # Qubits 0 and 1 are both idle in moments 0 and 1, before any gate.
    assert [(e.after_gate, e.qubits) for e in noise.events(circuit)] == [
        (-1, (0, 1)),
        (-1, (0, 1)),
    ]
    events = NoiseModel().add_idle(PauliError({"Z": 0.1}), qubits=[1]).events(circuit)
    assert [e.after_gate for e in events] == [-1, -1, -1]


@pytest.mark.parametrize("make_ansatz", ANSATZE)
def test_crosstalk_and_idle_match_density_matrix(make_ansatz):
    ansatz = make_ansatz()
    line = [(q, q + 1) for q in range(4)]
    noise = NoiseModel()
    noise.add_crosstalk(CoherentError("ZZ", 0.04), line, (gates.CZ, gates.CNOT))
    noise.add_crosstalk(PauliError({"Z": 0.01}), line)
    noise.add_idle(CoherentError("ZZ", 0.03), coupling=line)
    noise.add_idle(AmplitudeDampingError(0.01))
    noise.add(StatePreparationError(0.01))
    hsmpo = HSMPO(ansatz, tn_engine=QuimbEngine(svd_cutoff=0.0))
    result = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=2000, seed=5)
    exact = _model_expectation_exact(noise, ansatz.circuit, OBSERVABLE)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_coherent_twirled():
    twirled = CoherentError("ZZ", 0.2).twirled()
    assert twirled.probs == {"ZZ": pytest.approx(np.sin(0.1) ** 2)}


def test_preparation_error_events():
    circuit = _hand_circuit()
    events = NoiseModel().add(StatePreparationError(0.1), qubits=[1]).events(circuit)
    assert events == [PauliChannel(-1, (1,), {"X": 0.1})]
    with pytest.raises(ValueError, match="without a gate"):
        NoiseModel().add(StatePreparationError(0.1), gates.H)


# -- Asymmetric readout -----------------------------------------------------------


def _readout_exact(rho, observable, confusion):
    """
    ``<observable>`` read out with per-qubit confusion matrices
    ``confusion[q][reported, true]``, each term measured in its own eigenbasis.
    """
    n = len(confusion)
    total = 0.0
    for pauli, coefficient in observable.items():
        rotation = np.array([[1.0]])
        for p in pauli:
            basis = {
                "X": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                "Y": np.array([[1, -1j], [1, 1j]]) / np.sqrt(2),
            }.get(p, np.eye(2))
            rotation = np.kron(rotation, basis)
        probs = np.real(np.diag(rotation @ rho @ rotation.conj().T))
        probs = probs.reshape([2] * n)
        for q in range(n):
            probs = np.moveaxis(np.tensordot(confusion[q], probs, axes=(1, q)), 0, q)
        signs = np.ones([2] * n)
        for q, p in enumerate(pauli):
            if p != "I":
                shape = [1] * n
                shape[q] = 2
                signs = signs * np.array([1, -1]).reshape(shape)
        total += coefficient * float(np.sum(probs * signs))
    return total


def test_asymmetric_readout_matches_confusion_matrices():
    circuit = _kicked_ising().circuit
    noise = NoiseModel().add(PauliError({"Z": 0.004, "X": 0.002}))
    noise.add(ReadoutError(0.01, 0.04)).add(ReadoutError(0.02, 0.0), qubits=[1])
    observable = {"ZZIII": 0.7, "IXIXI": -0.4, "IIYZZ": 0.3, "IIIII": 0.1}

    confusion = []
    for q in range(5):
        matrix = np.eye(2)
        errors = [(0.01, 0.04)] + ([(0.02, 0.0)] if q == 1 else [])
        for p0, p1 in errors:
            matrix = np.array([[1 - p0, p1], [p0, 1 - p1]]) @ matrix
        confusion.append(matrix)
    exact = _readout_exact(noise.apply(circuit)().state(), observable, confusion)

    constant, terms = noise.readout_observable(observable, 5)
    rho = noise.apply(circuit)().state()
    folded = constant + sum(
        c * np.real(np.trace(rho @ pauli_matrix(p))) for p, c in terms.items()
    )
    assert folded == pytest.approx(exact, abs=1e-12)

    result = HSMPO(circuit).noisy_expectation(observable, noise, n_samples=1500, seed=6)
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10


def test_causal_cone_drops_unreachable_kraus_locations():
    """Damping that cannot reach the observable is the identity, by trace preservation."""
    circuit = Circuit(4)
    circuit.add(gates.RY(0, theta=0.4))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RX(1, theta=0.3))
    circuit.add(gates.RY(3, theta=0.7))
    circuit.add(gates.CNOT(2, 3))
    circuit.add(gates.RZ(2, theta=0.2))
    hsmpo = HSMPO(circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    noise = NoiseModel().add(AmplitudeDampingError(0.1))
    compiled = compile_noise(hsmpo, noise.events(circuit))

    # Qubits 2 and 3 never meet 0 and 1: all of their damping drops out.
    restricted = compiled.restricted_to([compiled.dress_observable("ZZII")[0]])
    assert set(restricted.location_position) <= set(
        range(len(restricted.generators) + 1)
    )
    for pauli in restricted.coherence_ket + restricted.coherence_bra:
        assert pauli[2:] == "II"
    assert len(restricted.coherence_weights) < len(compiled.coherence_weights)

    # Damping after the last gate on qubits 2, 3 only: exactly noiseless for ZZII.
    late = NoiseModel().add(AmplitudeDampingError(0.3), gates.CNOT, qubits=[2, 3])
    result = hsmpo.noisy_expectation("ZZII", late)
    assert result.n_transitions == 0
    assert result.mean == pytest.approx(hsmpo.expectation("ZZII"), abs=1e-12)

    result = hsmpo.noisy_expectation("ZZII", noise, n_samples=3000, seed=12)
    exact = _model_expectation_exact(noise, circuit, {"ZZII": 1.0})
    assert abs(result.mean - exact) <= 4 * result.stderr + 1e-10
