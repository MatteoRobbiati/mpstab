"""Wall time of a local Pauli noise model plus readout error versus system size.

Kicked-Ising circuit (common.kicked_ising_layered), with the public NoiseModel:
DepolarizingError(p1q) after every RX, DepolarizingError(p2q) after every CNOT
and ReadoutError(p0, p1) on every measured qubit. RZ is virtual and noiseless.

For each (n, steps, observable): one run with n_samples=0 (the exact k=0, k=1
strata only: the fixed cost) and --repeats runs at --samples with seeds
0, 1, ...; from these, t(N) = a + b N and the time to stderr <= eps.
No exact reference here (n up to 50): the error bar is the sampler's stderr.

Usage:
  PYTHONPATH=<mpstab>/src python bench_pauli_readout.py \
      [--ns 10 20 30 40 50] [--steps 2 4 8] [--observables ZZ_mid Z_avg]
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from common import RESULTS, environment, kicked_ising_layered
from qibo import gates

from mpstab import HSMPO, DepolarizingError, NoiseModel, ReadoutError


def observable(n: int, kind: str):
    if kind == "ZZ_mid":
        labels = ["I"] * n
        labels[n // 2 - 1] = labels[n // 2] = "Z"
        return "".join(labels)
    if kind == "Z_avg":  # magnetization per site: n local terms
        return {"I" * q + "Z" + "I" * (n - q - 1): 1 / n for q in range(n)}
    raise ValueError(kind)


def noise_model(p1q, p2q, p0, p1):
    return (
        NoiseModel()
        .add(DepolarizingError(p1q), gates.RX)
        .add(DepolarizingError(p2q), gates.CNOT)
        .add(ReadoutError(p0, p1))
    )


def one_run(n, steps, kind, samples, seed, args):
    """Runs in a fresh process so the peak memory is per run."""
    code = (
        "import json, sys, time, resource\n"
        "sys.path.insert(0, '.')\n"
        "from bench_pauli_readout import observable, noise_model\n"
        "from common import kicked_ising_layered\n"
        "from mpstab import HSMPO\n"
        f"noise = noise_model({args.p1q}, {args.p2q}, {args.p0}, {args.p1})\n"
        "t = time.perf_counter()\n"
        f"h = HSMPO.rotations_only(kicked_ising_layered({n}, {steps}))\n"
        f"r = h.noisy_expectation(observable({n}, '{kind}'), noise, "
        f"n_samples={samples}, seed={seed})\n"
        "wall = time.perf_counter() - t\n"
        "print(json.dumps({'mean': r.mean, 'stderr': r.stderr, 'wall_s': wall,"
        " 'W': list(r.stratum_weights), 'n_patterns': r.n_patterns,"
        " 'peak_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not lines:
        raise RuntimeError(out.stderr[-800:])
    return json.loads(lines[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--steps", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--observables", nargs="+", default=["ZZ_mid", "Z_avg"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--p1q", type=float, default=1e-3)
    parser.add_argument("--p2q", type=float, default=5e-3)
    parser.add_argument("--p0", type=float, default=0.01)
    parser.add_argument("--p1", type=float, default=0.03)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=600,
        help="skip larger n for a (steps, observable) once a run exceeds this",
    )
    args = parser.parse_args()

    rows = []
    for kind in args.observables:
        for steps in args.steps:
            for n in args.ns:
                fixed = one_run(n, steps, kind, 0, 0, args)
                reps = [
                    one_run(n, steps, kind, args.samples, s, args)
                    for s in range(args.repeats)
                ]
                walls = np.array([r["wall_s"] for r in reps])
                a = fixed["wall_s"]
                b = max(walls.mean() - a, 0.0) / args.samples
                s = float(np.mean([r["stderr"] for r in reps]))
                row = {
                    "observable": kind,
                    "steps": steps,
                    "n": n,
                    "W0": fixed["W"][0],
                    "W1": fixed["W"][1],
                    "W2plus": fixed["W"][2],
                    "n_patterns": fixed["n_patterns"],
                    "mean": float(np.mean([r["mean"] for r in reps])),
                    "stderr_N": s,
                    "N": args.samples,
                    "wall_fixed_s": a,
                    "wall_N_s": float(walls.mean()),
                    "wall_N_std": float(walls.std(ddof=1)) if len(walls) > 1 else 0.0,
                    "per_sample_ms": 1e3 * b,
                    "peak_rss_MiB": max(r["peak_rss_MiB"] for r in reps + [fixed]),
                }
                for eps in args.eps:
                    n_eps = (
                        0
                        if s <= eps * 1e-12
                        else int(np.ceil(args.samples * (s / eps) ** 2))
                    )
                    row[f"time_eps_{eps:g}"] = a + b * n_eps
                rows.append(row)
                print(json.dumps(row), flush=True)
                if walls.max() > args.max_seconds:
                    break

    stem = RESULTS / "pauli_readout"
    RESULTS.mkdir(exist_ok=True)
    with open(f"{stem}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "pauli_readout.json").write_text(
        json.dumps(
            {"args": vars(args), "environment": environment(), "rows": rows}, indent=1
        )
    )

    fig, axes = plt.subplots(
        1,
        len(args.observables),
        figsize=(5.5 * len(args.observables), 4),
        sharey=True,
        squeeze=False,
    )
    for ax, kind in zip(axes[0], args.observables):
        for i, steps in enumerate(args.steps):
            sel = [r for r in rows if r["observable"] == kind and r["steps"] == steps]
            ns = [r["n"] for r in sel]
            ax.plot(
                ns,
                [r["wall_N_s"] for r in sel],
                "o-",
                color=f"C{i}",
                label=f"{steps} steps, N={args.samples}",
            )
            ax.plot(
                ns,
                [r[f"time_eps_{min(args.eps):g}"] for r in sel],
                "s:",
                color=f"C{i}",
                label=f"{steps} steps, to stderr {min(args.eps):g}",
            )
        ax.set(yscale="log", xlabel="qubits", title=kind)
    axes[0][0].set_ylabel("wall time [s], single core")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=150)


if __name__ == "__main__":
    main()
