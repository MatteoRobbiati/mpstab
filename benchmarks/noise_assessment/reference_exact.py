"""Exact noisy expectation values on qibo's density matrix, cached per configuration.

Usage:
  python reference_exact.py --n 10 --steps 2 --scale 1 [--observable ZZ_mid]
Prints the value and stores it, with its wall time, in results/reference.json.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from bench_estimator_run import build, observable
from common import RESULTS
from qibo import Circuit

from mpstab.models.noise.model import _qibo_gate
from mpstab.pauli import pauli_matrix

CACHE = RESULTS / "reference.json"


def reference_key(n, steps, scale, zz_khz, obs):
    return f"n={n},steps={steps},scale={scale:g},zz={zz_khz:g},obs={obs}"


def exact_value(n, steps, scale, zz_khz, obs_kind):
    circuit, events = build(n, steps, scale, zz_khz)
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)
    noisy = Circuit(n, density_matrix=True)
    for event in by_gate.get(-1, ()):
        noisy.add(_qibo_gate(event))
    for index, gate in enumerate(circuit.queue):
        noisy.add(gate)
        for event in by_gate.get(index, ()):
            noisy.add(_qibo_gate(event))
    rho = np.asarray(noisy().state())
    pauli = observable(n, obs_kind)
    # Z strings are diagonal: <P> = sum_i rho_ii (+-1).
    signs = np.ones(2**n)
    for q, p in enumerate(pauli):
        if p == "Z":
            bit = (np.arange(2**n) >> (n - 1 - q)) & 1
            signs *= 1 - 2 * bit
        elif p != "I":
            return float(np.real(np.trace(rho @ pauli_matrix(pauli))))
    return float(np.real(np.diagonal(rho) @ signs))


def cached(n, steps, scale, zz_khz=50.0, obs_kind="ZZ_mid"):
    table = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    k = reference_key(n, steps, scale, zz_khz, obs_kind)
    if k not in table:
        start = time.perf_counter()
        value = exact_value(n, steps, scale, zz_khz, obs_kind)
        table = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        table[k] = {"value": value, "wall_s": time.perf_counter() - start}
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(table, indent=1))
    return table[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zz-khz", type=float, default=50.0)
    parser.add_argument("--observable", default="ZZ_mid")
    args = parser.parse_args()
    print(
        json.dumps(cached(args.n, args.steps, args.scale, args.zz_khz, args.observable))
    )


if __name__ == "__main__":
    main()
