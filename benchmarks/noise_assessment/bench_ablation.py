"""Ablation of the estimator strategies against the baseline (main) estimator.

Runs bench_estimator_run.py in a subprocess per (config, variant, seed), with
the right mpstab on PYTHONPATH, appending JSON lines to results/ablation.jsonl
(resumable: finished runs are skipped). Exact references come from
reference_exact.py. Single core, fixed seeds; seeds 0-2 at --samples, one more
seed at 4x --samples for the cost per sample.

The variants switch options that exist only up to commit a68e03c of the
noise-estimator branch (control_variates, positive_mixtures,
order_one_samples); run --new against a checkout of that commit. The
results led to keeping only the control variates.

Usage:
  python bench_ablation.py --base <main checkout>/src --new <branch>/src \
      [--ns 8 12] [--steps 1 2 4] [--scales 0.3 1 3]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import RESULTS

HERE = Path(__file__).parent
VARIANTS = {
    "baseline": ("base", []),
    "new_off": ("new", ["--no-cv", "--no-mixtures"]),
    "new_cv": ("new", ["--no-mixtures"]),
    "new_mix": ("new", ["--no-cv"]),
    "new_all": ("new", []),
    "new_cv_k1": ("new", ["--no-mixtures", "--k1", "100"]),
}


def free_memory_percent() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--ns", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--scales", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--out", default=str(RESULTS / "ablation.jsonl"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add(
                (r["label"], r["n"], r["steps"], r["scale"], r["seed"], r["samples"])
            )
    paths = {"base": args.base, "new": args.new}
    python = sys.executable

    for n in args.ns:
        for steps in args.steps:
            for scale in args.scales:
                ref = subprocess.run(
                    [
                        python,
                        "reference_exact.py",
                        "--n",
                        str(n),
                        "--steps",
                        str(steps),
                        "--scale",
                        str(scale),
                    ],
                    cwd=HERE,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "PYTHONPATH": args.new},
                )
                print(
                    f"reference n={n} steps={steps} scale={scale}: "
                    f"{ref.stdout.strip().splitlines()[-1] if ref.stdout else ref.stderr[-300:]}",
                    flush=True,
                )
                for variant in args.variants:
                    version, flags = VARIANTS[variant]
                    for seed, samples in [
                        (0, args.samples),
                        (1, args.samples),
                        (2, args.samples),
                        (3, 4 * args.samples),
                    ]:
                        if (variant, n, steps, scale, seed, samples) in done:
                            continue
                        if free_memory_percent() < 25:
                            sys.exit("memory pressure: stopping")
                        run = subprocess.run(
                            [
                                python,
                                "bench_estimator_run.py",
                                "--n",
                                str(n),
                                "--steps",
                                str(steps),
                                "--scale",
                                str(scale),
                                "--samples",
                                str(samples),
                                "--seed",
                                str(seed),
                                "--label",
                                variant,
                                *flags,
                            ],
                            cwd=HERE,
                            capture_output=True,
                            text=True,
                            env={**os.environ, "PYTHONPATH": paths[version]},
                        )
                        line = [l for l in run.stdout.splitlines() if l.startswith("{")]
                        if not line:
                            print(
                                f"FAILED {variant} n={n} steps={steps} scale={scale} "
                                f"seed={seed}: {run.stderr[-500:]}",
                                flush=True,
                            )
                            continue
                        with out.open("a") as f:
                            f.write(line[-1] + "\n")
                        r = json.loads(line[-1])
                        print(
                            f"{variant:10s} n={n} steps={steps} scale={scale:g} seed={seed} "
                            f"N={samples}: {r['mean']:+.5f} +- {r['stderr']:.5f} "
                            f"in {r['wall_s']:.1f}s",
                            flush=True,
                        )


if __name__ == "__main__":
    main()
