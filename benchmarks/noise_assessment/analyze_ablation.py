"""Time to accuracy epsilon for each estimator variant, from results/ablation.jsonl.

For each (config, variant): the wall time t(N) = a + b N is fitted from the
three seeds at N and the one at 4N; the standard error scales as 1/sqrt(N)
from its mean over the three seeds at N; the time to reach stderr <= eps is
then a + b N_eps with N_eps = N (stderr / eps)^2 (at least one sample).
Bias is checked as z = (mean - exact) / stderr over every run.

M_cone is the entrywise (quasi-probability) mass in the observable's light
cone, the same for every variant. Writes results/ablation_summary.{csv,md,png}.

Usage:
  PYTHONPATH=<branch>/src python analyze_ablation.py [--eps 1e-2 1e-3]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from bench_estimator_run import build, observable
from common import RESULTS, summarize
from reference_exact import reference_key

from mpstab import HSMPO
from mpstab.hamiltonians import pauli_terms
from mpstab.models.noise.compile import compile_noise

ORDER = ["baseline", "new_off", "new_cv", "new_mix", "new_all", "new_cv_k1"]
COLORS = dict(zip(ORDER, ["k", "C0", "C1", "C2", "C3", "C4"]))


def cone_mass(n, steps, scale):
    circuit, events = build(n, steps, scale, 50.0)
    compiled = compile_noise(
        HSMPO.rotations_only(circuit), events, positive_mixtures=False
    )
    _, terms = pauli_terms(observable(n, "ZZ_mid"), n)
    dressed = compiled.dress_observables(list(terms))
    return summarize(compiled.restricted_to([p for p, _ in dressed]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-2, 1e-3])
    args = parser.parse_args()

    runs = [
        json.loads(l) for l in (RESULTS / "ablation.jsonl").read_text().splitlines()
    ]
    refs = json.loads((RESULTS / "reference.json").read_text())
    groups = defaultdict(list)
    for r in runs:
        groups[(r["n"], r["steps"], r["scale"], r["label"])].append(r)

    masses = {}
    rows = []
    for (n, steps, scale, label), rs in sorted(groups.items()):
        if (n, steps, scale) not in masses:
            masses[n, steps, scale] = cone_mass(n, steps, scale)
        cone = masses[n, steps, scale]
        exact = refs[reference_key(n, steps, scale, 50.0, "ZZ_mid")]
        small = [r for r in rs if r["seed"] < 3]
        large = [r for r in rs if r["seed"] == 3]
        if not small:
            continue
        N = small[0]["samples"]
        t_small = np.array([r["wall_s"] for r in small])
        s_small = np.array([r["stderr"] for r in small])
        if large:
            b = max(
                (large[0]["wall_s"] - t_small.mean()) / (large[0]["samples"] - N), 0.0
            )
        else:
            b = 0.0
        a = max(t_small.mean() - b * N, 0.0)
        s = s_small.mean()
        z = [(r["mean"] - exact["value"]) / r["stderr"] for r in rs if r["stderr"] > 0]
        row = {
            "n": n,
            "steps": steps,
            "scale": scale,
            "variant": label,
            "M_cone": cone["M"],
            "W2_cone": cone["W2plus"],
            "L_cone": cone["locations"],
            "R_cone": cone["rotations"],
            "exact": exact["value"],
            "exact_dm_s": exact["wall_s"],
            "wall_s_N": t_small.mean(),
            "wall_s_N_std": t_small.std(ddof=1) if len(small) > 1 else 0,
            "fixed_s": a,
            "per_sample_ms": 1e3 * b,
            "stderr_N": s,
            "N": N,
            "max_abs_z": float(np.max(np.abs(z))) if z else 0.0,
            "mean_z": float(np.mean(z)) if z else 0.0,
            "error_N4": abs(large[0]["mean"] - exact["value"]) if large else np.nan,
            "stderr_N4": large[0]["stderr"] if large else np.nan,
            "peak_rss_MiB": max(r["peak_rss_MiB"] for r in rs),
        }
        for eps in args.eps:
            n_eps = 1 if s == 0 else max(1, int(np.ceil(N * (s / eps) ** 2)))
            row[f"N_eps_{eps:g}"] = n_eps
            row[f"time_eps_{eps:g}"] = a + b * n_eps
        rows.append(row)

    by_config = defaultdict(dict)
    for row in rows:
        by_config[row["n"], row["steps"], row["scale"]][row["variant"]] = row
    for config, variants in by_config.items():
        base = variants.get("baseline")
        for row in variants.values():
            for eps in args.eps:
                k = f"time_eps_{eps:g}"
                row[f"speedup_{eps:g}"] = (
                    base[k] / row[k] if base and row[k] > 0 else np.nan
                )

    with open(RESULTS / "ablation_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| n | steps | scale | M_cone | W>=2 | "
        + " | ".join(f"{v} t(1e-2) / t(1e-3) [s]" for v in ORDER)
        + " |",
        "|" + "---|" * (5 + len(ORDER)),
    ]
    for config in sorted(by_config):
        variants = by_config[config]
        any_row = next(iter(variants.values()))
        cells = []
        for v in ORDER:
            r = variants.get(v)
            cells.append(
                "—"
                if r is None
                else f"{r['time_eps_0.01']:.3g} / {r['time_eps_0.001']:.3g}"
            )
        lines.append(
            f"| {config[0]} | {config[1]} | {config[2]:g} | {any_row['M_cone']:.2f} | "
            f"{any_row['W2_cone']:.3f} | " + " | ".join(cells) + " |"
        )
    (RESULTS / "ablation_summary.md").write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(
        1, len(args.eps), figsize=(6 * len(args.eps), 4.5), sharey=True
    )
    for ax, eps in zip(np.atleast_1d(axes), args.eps):
        for v in ORDER:
            pts = sorted(
                (r["M_cone"], r[f"time_eps_{eps:g}"], r["n"])
                for r in rows
                if r["variant"] == v
            )
            for n, marker in ((8, "o"), (12, "s")):
                sel = [(m, t) for m, t, nn in pts if nn == n]
                if sel:
                    x, y = zip(*sel)
                    ax.plot(
                        x,
                        y,
                        marker=marker,
                        ls="-" if n == 12 else ":",
                        color=COLORS[v],
                        label=f"{v} (n={n})",
                        ms=4,
                        lw=1,
                    )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"$M_{\rm cone}$",
            title=rf"time to stderr $\leq$ {eps:g}",
        )
    np.atleast_1d(axes)[0].set_ylabel("wall time [s], single core")
    np.atleast_1d(axes)[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "ablation_summary.png", dpi=150)
    print((RESULTS / "ablation_summary.md").read_text())
    for v in ORDER:
        zs = [r["max_abs_z"] for r in rows if r["variant"] == v]
        if zs:
            print(f"{v:10s} configs={len(zs)} worst |z| over runs={max(zs):.2f}")


if __name__ == "__main__":
    main()
