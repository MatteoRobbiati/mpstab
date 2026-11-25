"""Phase 2: noise-mass audit of realistic device noise on the mitigation circuits.

For KickedIsing (the error-mitigation tutorial circuit) at n in {10, 20} and
depth in {10, 30, 60} moments, with local (weight 1, 2) and global observables,
report L, L_cone, M, M_cone, (W0, W1, W>=2), the k=1 pattern and transition
counts, and which noise source dominates M_cone. No MPS is built.

Variants:
  physical     layer-ordered circuit, duration-aware idle (virtual RZ = 0 ns)
  physical_cz  same, CNOT compiled as H CZ H
  stock_layer  NoiseModel API (add / add_idle with one 150 ns idle per moment)
               on the layer-ordered circuit
  stock_queue  same NoiseModel API on mpstab's KickedIsing queue (qubit-by-qubit
               order, which add_idle's in-order layering turns into a staircase)

Usage:
  python phase2_noise_mass_audit.py [--nqubits 10 20] [--depths 10 30 60]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
from common import (
    RESULTS,
    Device,
    depth_stats,
    device_dict,
    device_events,
    environment,
    kicked_ising_layered,
    summarize,
    write_json,
)
from qibo import gates

from mpstab import HSMPO
from mpstab.hamiltonians import pauli_terms
from mpstab.models.ansatze import KickedIsing
from mpstab.models.noise import (
    CoherentError,
    DepolarizingError,
    NoiseModel,
    ThermalRelaxationError,
)
from mpstab.models.noise.compile import compile_noise
from mpstab.models.noise.mc_noise_estimator import noisy_expectation

ANGLES = dict(theta_h=0.7, theta_z=0.3, theta_j=-1.3)


def observables(n: int) -> dict[str, str]:
    mid = n // 2
    z1 = ["I"] * n
    z1[mid] = "Z"
    z2 = ["I"] * n
    z2[mid - 1] = z2[mid] = "Z"
    return {"Z_mid": "".join(z1), "ZZ_mid": "".join(z2), "Z_all": "Z" * n}


def steps_for_depth(n: int, depth: int, native: str) -> int:
    per_step = (
        depth_stats(kicked_ising_layered(n, 2, native=native, **ANGLES))["moments"] / 2
    )
    return max(1, int(round(depth / per_step)))


def stock_model(device: Device, n: int) -> NoiseModel:
    chain = [(q, q + 1) for q in range(n - 1)]
    zz = CoherentError("ZZ", np.pi * device.zz_hz * 150e-9)
    model = (
        NoiseModel()
        .add(DepolarizingError(device.p1q), [gates.RX, gates.H])
        .add(DepolarizingError(device.p2q), [gates.CNOT, gates.CZ])
        .add(ThermalRelaxationError(device.t1, device.t2, 30e-9), [gates.RX, gates.H])
        .add(
            ThermalRelaxationError(device.t1, device.t2, 150e-9), [gates.CNOT, gates.CZ]
        )
        .add_idle(ThermalRelaxationError(device.t1, device.t2, 150e-9))
    )
    if device.zz_hz > 0:
        model.add_idle(zz, coupling=chain)
    return model


def cone_summary(compiled, observable: str) -> dict:
    _, terms = pauli_terms(observable, compiled.nqubits)
    dressed = compiled.dress_observables(list(terms))
    return summarize(compiled.restricted_to([p for p, _ in dressed]))


def timed_compile(hsmpo, events, repeats: int):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled = compile_noise(hsmpo, events)
        times.append(time.perf_counter() - start)
    tracemalloc.start()
    compile_noise(hsmpo, events)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return compiled, statistics.median(times), peak / 2**20


def run_config(n, depth, variant, device, repeats):
    native = "cz" if variant == "physical_cz" else "cnot"
    steps = steps_for_depth(n, depth, native)
    if variant == "stock_queue":
        circuit = KickedIsing(nqubits=n, n_steps=steps, **ANGLES).circuit
    else:
        circuit = kicked_ising_layered(n, steps, native=native, **ANGLES)
    hsmpo = HSMPO.rotations_only(circuit)
    stats = depth_stats(circuit)

    if variant.startswith("stock"):
        by_source = {"all": stock_model(device, n).events(hsmpo.ansatz.circuit)}
    else:
        by_source = device_events(hsmpo.ansatz.circuit, device)
    all_events = [e for evs in by_source.values() for e in evs]
    compiled, t_compile, mem = timed_compile(hsmpo, all_events, repeats)
    full = summarize(compiled)

    rows = []
    for obs_name, obs in observables(n).items():
        cone = cone_summary(compiled, obs)
        row = {
            "variant": variant,
            "n": n,
            "depth_target": depth,
            "n_steps": steps,
            **{f"circuit_{k}": v for k, v in stats.items()},
            "observable": obs_name,
            "weight": obs.count("Z"),
            "L": full["locations"],
            "M": full["M"],
            "R": full["rotations"],
            "L_cone": cone["locations"],
            "M_cone": cone["M"],
            "M_cone_twirl": cone["M_twirl"],
            "M_cone_coherence": cone["M_coherence"],
            "R_cone": cone["rotations"],
            "faults_cone": cone["faults"],
            "W0": cone["W0"],
            "W1": cone["W1"],
            "W2plus": cone["W2plus"],
            "quasi_norm_cone": cone["quasi_norm"],
            "k1_patterns": cone["k1_patterns"],
            "k1_transitions": cone["k1_transitions"],
            "compile_s_median": t_compile,
            "compile_peak_MiB": mem,
        }
        if not variant.startswith("stock"):
            for source, evs in by_source.items():
                if not evs:
                    continue
                only = compile_noise(hsmpo, evs)
                rest = compile_noise(
                    hsmpo, [e for s, es in by_source.items() if s != source for e in es]
                )
                row[f"M_cone_only_{source}"] = cone_summary(only, obs)["M"]
                row[f"M_cone_without_{source}"] = cone_summary(rest, obs)["M"]
        rows.append(row)
        print(
            f"{variant:12s} n={n:2d} depth~{depth:2d} ({stats['moments']} moments, {steps} steps) "
            f"{obs_name:6s} L={row['L']:5d} L_cone={row['L_cone']:5d} M={row['M']:.3f} "
            f"M_cone={row['M_cone']:.3f} W=({row['W0']:.3f},{row['W1']:.3f},{row['W2plus']:.3f}) "
            f"pat={row['k1_patterns']} tr={row['k1_transitions']} R_cone={row['R_cone']}",
            flush=True,
        )
    return rows


def validate(device: Device) -> dict:
    """Check our strata formulas against noisy_expectation on a small case."""
    n = 6
    circuit = kicked_ising_layered(n, 1, **ANGLES)
    hsmpo = HSMPO(circuit, progress=False)
    events = [
        e for evs in device_events(hsmpo.ansatz.circuit, device).values() for e in evs
    ]
    compiled = compile_noise(hsmpo, events)
    obs = observables(n)["ZZ_mid"]
    ours = cone_summary(compiled, obs)
    theirs = noisy_expectation(compiled, obs, n_samples=0).stratum_weights
    diff = max(
        abs(a - b) for a, b in zip((ours["W0"], ours["W1"], ours["W2plus"]), theirs)
    )
    print(
        f"validation: ours {ours['W0']:.6g},{ours['W1']:.6g},{ours['W2plus']:.6g} "
        f"vs noisy_expectation {theirs}; max diff {diff:.2e}"
    )
    assert diff < 1e-12, "strata formulas disagree with noisy_expectation"
    return {
        "n": n,
        "ours": [ours["W0"], ours["W1"], ours["W2plus"]],
        "mpstab": list(theirs),
        "max_abs_diff": diff,
    }


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    markers = {"Z_mid": "o", "ZZ_mid": "s", "Z_all": "^"}
    colors = {
        "physical": "C0",
        "physical_cz": "C1",
        "stock_layer": "C2",
        "stock_queue": "C3",
    }
    for ax, n in zip(axes, sorted({r["n"] for r in rows})):
        for variant in colors:
            for obs in markers:
                pts = sorted(
                    (r["circuit_moments"], r["M_cone"])
                    for r in rows
                    if r["n"] == n
                    and r["variant"] == variant
                    and r["observable"] == obs
                )
                if pts:
                    x, y = zip(*pts)
                    ax.plot(
                        x,
                        y,
                        marker=markers[obs],
                        color=colors[variant],
                        label=f"{variant}, {obs}",
                        lw=1,
                        ms=5,
                    )
        ax.axhline(0.3, color="k", ls=":", lw=1)
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="circuit depth (moments)",
            title=f"n = {n}",
        )
    axes[0].set_ylabel(r"$M_{\rm cone}$ (quasi-probability mass in the light cone)")
    axes[1].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--nqubits", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--depths", type=int, nargs="+", default=[10, 30, 60])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["physical", "physical_cz", "stock_layer", "stock_queue"],
    )
    parser.add_argument("--zz-khz", type=float, default=50.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    device = Device(zz_hz=args.zz_khz * 1e3)
    validation = validate(device)
    rows = []
    for variant in args.variants:
        for n in args.nqubits:
            for depth in args.depths:
                rows += run_config(n, depth, variant, device, args.repeats)

    stem = f"phase2_mass_audit{args.tag}"
    write_json(
        RESULTS / f"{stem}.json",
        {
            "device": device_dict(device),
            "angles": ANGLES,
            "environment": environment(),
            "validation": validation,
            "rows": rows,
        },
    )
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(RESULTS / f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, RESULTS / f"{stem}.png")
    print(f"wrote {RESULTS / stem}.{{json,csv,png}}")


if __name__ == "__main__":
    main()
