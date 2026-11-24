"""Noisy expectation values on the HSMPO surrogate, against exact simulation.

A kicked Ising circuit runs under depolarizing noise after every gate, with a
correlated ZZ dephasing term after every CNOT and coherent ZZ crosstalk. For
several noise strengths the script prints the noiseless value, the noisy value
from the hybrid surrogate (``mpstab.models.noise``) and the exact value from a qibo
density-matrix simulation, with timings.

At this width the exact density matrix is far cheaper: the point of the
comparison is accuracy. The hybrid cost is set by the number of distinct sign
patterns times the cost of a dressed rotation on the MPS, not by ``4**n``, so it
is the method that keeps working where density matrices no longer fit.

Run with ``uv run python examples/scripts/noise/noisy_simulation.py``.
"""

import time

import numpy as np
from qibo import gates

from mpstab import (
    HSMPO,
    CoherentError,
    NoiseModel,
    PauliError,
    pauli_string_to_hamiltonian,
)
from mpstab.models.ansatze import KickedIsing

NQUBITS = 8
OBSERVABLE = "ZIIIIIIZ"
NOISE_STRENGTHS = (0.001, 0.005, 0.01)
CROSSTALK_ANGLE = 0.02
N_SAMPLES = 1000


def noise_model(p):
    """Depolarizing noise everywhere, plus ZZ dephasing and crosstalk on CNOTs."""
    noise = NoiseModel()
    noise.add(PauliError({label: p / 3 for label in "XYZ"}))
    noise.add(PauliError({"ZZ": p / 2}), gates.CNOT)
    noise.add(CoherentError("ZZ", CROSSTALK_ANGLE), gates.CNOT)
    return noise


def exact_noisy_expectation(circuit, noise, observable):
    """The same noise in a qibo density-matrix simulation."""
    hamiltonian = pauli_string_to_hamiltonian(observable)
    return hamiltonian.expectation_from_state(noise.apply(circuit)().state())


def main():
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
        f"{'patterns':>8} | {'Pr(k>=2)':>8} | {'t hybrid':>8} | {'t exact':>7}"
    )
    print(header)
    print("-" * len(header))
    for p in NOISE_STRENGTHS:
        noise = noise_model(p)

        start = time.perf_counter()
        result = surrogate.noisy_expectation(
            OBSERVABLE, noise, n_samples=N_SAMPLES, seed=0
        )
        hybrid_time = time.perf_counter() - start

        start = time.perf_counter()
        exact = exact_noisy_expectation(circuit, noise, OBSERVABLE)
        exact_time = time.perf_counter() - start

        deviation = (
            abs(result.mean - exact) / result.stderr if result.stderr else np.nan
        )
        print(
            f"{p:>6} | {result.mean:+.5f} +- {result.stderr:.5f} | {exact:+.5f} | "
            f"{deviation:>10.2f} | {result.n_patterns:>8} | "
            f"{result.stratum_weights[2]:>8.4f} | "
            f"{hybrid_time:>7.2f}s | {exact_time:>6.2f}s"
        )


if __name__ == "__main__":
    main()
