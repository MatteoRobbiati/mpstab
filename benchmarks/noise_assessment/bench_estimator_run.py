"""One timed noisy-expectation run, printed as a JSON line.

Version-independent on purpose: it builds the noise as events (common.py) and
calls compile_noise + noisy_expectation, which exist in every mpstab version,
so the same script times the baseline (main) and the new estimator, whichever
mpstab is on PYTHONPATH. Options a version lacks are refused, not ignored.

Usage:
  PYTHONPATH=<mpstab>/src python bench_estimator_run.py --n 10 --steps 2 \
      --scale 1 --samples 1000 --seed 0 [--no-cv] [--no-mixtures] [--k1 N]
"""

from __future__ import annotations

import argparse
import inspect
import json
import resource
import time

import numpy as np
from common import Device, device_events, environment, kicked_ising_layered

from mpstab import HSMPO
from mpstab.models.noise.compile import compile_noise
from mpstab.models.noise.mc_noise_estimator import noisy_expectation

ANGLES = dict(theta_h=0.7, theta_z=0.3, theta_j=-1.3)


def observable(n: int, kind: str) -> str:
    mid = n // 2
    labels = ["I"] * n
    if kind == "Z_mid":
        labels[mid] = "Z"
    elif kind == "ZZ_mid":
        labels[mid - 1] = labels[mid] = "Z"
    else:
        labels = ["Z"] * n
    return "".join(labels)


def scaled_device(scale: float, zz_khz: float) -> Device:
    """Every rate times ``scale``: depolarizing, 1/T1, 1/T2 and the ZZ shift."""
    base = Device(zz_hz=zz_khz * 1e3)
    return Device(
        t1=base.t1 / scale,
        t2=base.t2 / scale,
        p1q=base.p1q * scale,
        p2q=base.p2q * scale,
        zz_hz=base.zz_hz * scale,
    )


def build(n, steps, scale, zz_khz):
    circuit = kicked_ising_layered(n, steps, **ANGLES)
    events = [
        e
        for evs in device_events(circuit, scaled_device(scale, zz_khz)).values()
        for e in evs
    ]
    return circuit, events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--zz-khz", type=float, default=50.0)
    parser.add_argument("--observable", default="ZZ_mid")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-mixtures", action="store_true")
    parser.add_argument("--k1", type=int, default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    circuit, events = build(args.n, args.steps, args.scale, args.zz_khz)
    obs = observable(args.n, args.observable)

    compile_options, estimate_options = {}, {}
    if "positive_mixtures" in inspect.signature(compile_noise).parameters:
        compile_options["positive_mixtures"] = not args.no_mixtures
    elif args.no_mixtures is False and args.label != "baseline":
        raise SystemExit("this mpstab has no positive mixtures")
    params = inspect.signature(noisy_expectation).parameters
    if "control_variates" in params:
        estimate_options["control_variates"] = not args.no_cv
        estimate_options["order_one_samples"] = args.k1
    elif args.k1 is not None or args.no_cv:
        raise SystemExit("this mpstab has no --k1 or --no-cv (removed after a68e03c)")

    start = time.perf_counter()
    hsmpo = HSMPO.rotations_only(circuit)
    compiled = compile_noise(hsmpo, events, **compile_options)
    t_compile = time.perf_counter() - start
    result = noisy_expectation(
        compiled, obs, n_samples=args.samples, seed=args.seed, **estimate_options
    )
    wall = time.perf_counter() - start

    print(
        json.dumps(
            {
                "label": args.label,
                "n": args.n,
                "steps": args.steps,
                "scale": args.scale,
                "zz_khz": args.zz_khz,
                "observable": args.observable,
                "samples": args.samples,
                "seed": args.seed,
                "no_cv": args.no_cv,
                "no_mixtures": args.no_mixtures,
                "k1": args.k1,
                "mean": result.mean,
                "stderr": result.stderr,
                "stratum_weights": list(result.stratum_weights),
                "n_patterns": result.n_patterns,
                "n_transitions": getattr(result, "n_transitions", None),
                "variance_reduction": getattr(result, "variance_reduction", None),
                "min_fidelity": (
                    float(np.nanmin(result.fidelities))
                    if len(result.fidelities)
                    else None
                ),
                "wall_s": wall,
                "compile_s": t_compile,
                # ru_maxrss is in bytes on macOS
                "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 2**20,
                "mpstab_commit": environment()["mpstab_commit"],
            }
        )
    )


if __name__ == "__main__":
    main()
