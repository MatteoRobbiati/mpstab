"""Tests for mpstab.noise: Pauli and coherent noise in the HSMPO frame."""

import numpy as np
import pytest
import stim
from qibo import Circuit, gates
from utils import pauli_matrix

from mpstab import HSMPO
from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.noise import (
    CoherentError,
    DepolarizingError,
    NoiseModel,
    PauliError,
    ReadoutError,
)
from mpstab.noise.compile import compile_noise
from mpstab.noise.estimator import noisy_expectation
from mpstab.noise.events import CoherentRotation, NoiseEvent, PauliChannel

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
    # Bit-exact without the light cone; with it, fewer rotations round differently.
    unrestricted = noisy_expectation(compiled, "ZZIII", n_samples=0, light_cone=False)
    assert unrestricted.mean == hsmpo.expectation("ZZIII")
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
    """Every ``::`` literal block in the docstrings of mpstab.noise."""
    import inspect
    import textwrap

    import mpstab.noise as noise
    from mpstab import pauli
    from mpstab.noise import compile, estimator, events

    blocks = []
    objects = [getattr(noise, name) for name in noise.__all__]
    objects += [compile.compile_noise, estimator.noisy_expectation]
    objects += [
        events.PauliChannel,
        events.CoherentRotation,
    ]
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
    from mpstab.noise import NoiseModel, PauliError

    hsmpo = HSMPO(_kicked_ising(), max_bond_dimension=4)
    noise = NoiseModel().add(PauliError({"X": 0.01, "Z": 0.02}))
    serial = hsmpo.noisy_expectation(OBSERVABLE, noise, n_samples=200, seed=5)
    parallel = hsmpo.noisy_expectation(
        OBSERVABLE, noise, n_samples=200, seed=5, n_workers=3
    )
    assert parallel.mean == pytest.approx(serial.mean, abs=1e-12)
    assert parallel.stderr == pytest.approx(serial.stderr, abs=1e-12)
    np.testing.assert_allclose(parallel.fidelities, serial.fidelities, atol=1e-12)
