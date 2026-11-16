"""Noisy expectation values on the HSMPO surrogate, against exact simulation.

A kicked Ising circuit runs under depolarizing noise after every gate, with a
correlated ZZ dephasing term after every CNOT and coherent ZZ crosstalk. For
several noise strengths the script prints the noiseless value, the noisy value
from the hybrid surrogate (``mpstab.noise``) and the exact value from a qibo
density-matrix simulation, with timings.

At this width the exact density matrix is far cheaper: the point of the
comparison is accuracy. The hybrid cost is set by the number of distinct sign
patterns times the cost of a dressed rotation on the MPS, not by ``4**n``, so it
is the method that keeps working where density matrices no longer fit.

Run with ``uv run python examples/noisy_simulation.py``.
"""

import time

import numpy as np
from qibo import Circuit, gates, set_backend

from mpstab import HSMPO
from mpstab.hamiltonians import pauli_string_to_hamiltonian
from mpstab.models.ansatze import KickedIsing
from mpstab.noise import (
    CoherentRotation,
    PauliChannel,
    compile_noise,
    noisy_expectation,
)

NQUBITS = 8
OBSERVABLE = "ZIIIIIIZ"
NOISE_STRENGTHS = (0.001, 0.005, 0.01)
CROSSTALK_ANGLE = 0.02
N_SAMPLES = 1000


def noise_events(circuit, p):
    """Depolarizing noise everywhere, plus ZZ dephasing and crosstalk on CNOTs."""
    events = []
    for index, gate in enumerate(circuit.queue):
        for q in gate.qubits:
            events.append(PauliChannel(index, (q,), {l: p / 3 for l in "XYZ"}))
        if len(gate.qubits) == 2:
            events.append(PauliChannel(index, gate.qubits, {"ZZ": p / 2}))
            generator = "".join(
                "Z" if q in gate.qubits else "I" for q in range(circuit.nqubits)
            )
            events.append(CoherentRotation(index, generator, CROSSTALK_ANGLE))
    return events


def exact_noisy_expectation(circuit, events, observable):
    """The same noise in a qibo density-matrix simulation."""
    noisy = Circuit(circuit.nqubits, density_matrix=True)
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        for event in by_gate.get(index, ()):
            if isinstance(event, PauliChannel):
                noisy.add(
                    gates.PauliNoiseChannel(event.qubits, list(event.probs.items()))
                )
            else:
                q0, q1 = (q for q, label in enumerate(event.generator) if label == "Z")
                noisy.add(gates.RZZ(q0, q1, theta=event.angle))
    hamiltonian = pauli_string_to_hamiltonian(observable)
    return hamiltonian.expectation_from_state(noisy().state())


def main():
    set_backend("numpy")
    circuit = KickedIsing(
        nqubits=NQUBITS, n_steps=2, theta_h=0.7, theta_z=0.3, theta_j=-0.5 * np.pi
    ).circuit
    surrogate = HSMPO(circuit)
    print(
        f"{NQUBITS} qubits, {len(circuit.queue)} gates, "
        f"{surrogate.n_magic_gates} magic gates, observable {OBSERVABLE}"
    )
    print(f"noiseless: {surrogate.expectation(OBSERVABLE):+.5f}\n")

    header = (
        f"{'p':>6} | {'hybrid':>18} | {'exact':>9} | {'|diff|/err':>10} | "
        f"{'patterns':>8} | {'Pr(k>=2)':>8} | {'t compile':>9} | {'t hybrid':>8} | {'t exact':>7}"
    )
    print(header)
    print("-" * len(header))
    for p in NOISE_STRENGTHS:
        events = noise_events(circuit, p)

        start = time.perf_counter()
        compiled = compile_noise(surrogate, events)
        compile_time = time.perf_counter() - start
        result = noisy_expectation(compiled, OBSERVABLE, n_samples=N_SAMPLES, seed=0)
        hybrid_time = time.perf_counter() - start - compile_time

        start = time.perf_counter()
        exact = exact_noisy_expectation(circuit, events, OBSERVABLE)
        exact_time = time.perf_counter() - start

        deviation = (
            abs(result.mean - exact) / result.stderr if result.stderr else np.nan
        )
        print(
            f"{p:>6} | {result.mean:+.5f} +- {result.stderr:.5f} | {exact:+.5f} | "
            f"{deviation:>10.2f} | {result.n_patterns:>8} | "
            f"{result.stratum_weights[2]:>8.4f} | {compile_time:>8.3f}s | "
            f"{hybrid_time:>7.2f}s | {exact_time:>6.2f}s"
        )


if __name__ == "__main__":
    main()
