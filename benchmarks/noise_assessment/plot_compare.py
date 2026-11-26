"""Figures and summary table of bench_compare.py (results/compare.jsonl).

Time to |error| <= eps, per method:
- dm: its wall time (exact).
- mpstab: a + b N_eps, with a the n_samples=0 run, b from the runs at N samples
  and N_eps = N (stderr / eps)^2.
- pp: the fastest grid setting with |value - reference| <= eps; missing if none.
- traj: mean time per trajectory times (std / eps)^2.

Reference: dm where it ran; else the tightest PP setting if it moved by less
than 1e-4 from the previous one; else the most precise mpstab mean (flagged).

Writes results/compare_summary.csv and results/compare_{n,steps,scale}.png.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent / "results"
EPS = (1e-2, 1e-3)
METHODS = ["mpstab", "pp", "traj", "dm"]
NAMES = {
    "mpstab": "mpstab",
    "pp": "PauliPropagation.jl",
    "traj": "MPS trajectories (quimb)",
    "dm": "density matrix (qibo)",
}
# Fixed categorical slots (dataviz reference palette, light mode), by method.
COLORS = {"mpstab": "#2a78d6", "pp": "#eb6834", "traj": "#1baf7a", "dm": "#52514e"}
MARKERS = {"mpstab": "o", "pp": "s", "traj": "^", "dm": "D"}


def load():
    records = defaultdict(dict)
    for line in (RESULTS / "compare.jsonl").read_text().splitlines():
        r = json.loads(line)
        records[r["n"], r["steps"], r["scale"], r["obs"]][r["method"]] = r
    return records


def reference(methods):
    dm = methods.get("dm")
    if dm and "value" in dm:
        return dm["value"], "dm"
    pp = methods.get("pp")
    if pp and pp.get("grid"):
        full = [g for g in pp["grid"] if g["max_weight"] == "Inf"]
        if len(full) >= 2 and abs(full[-1]["value"] - full[-2]["value"]) < 1e-4:
            return full[-1]["value"], "pp"
    mp = methods.get("mpstab")
    if mp and "runs" in mp:
        best = min(
            (r for r in mp["runs"] if r["samples"] > 0), key=lambda r: r["stderr"]
        )
        return best["mean"], "mpstab"
    return None, None


def summarize(cfg, methods):
    ref, source = reference(methods)
    rows = []
    for method, r in methods.items():
        row = dict(
            zip(("n", "steps", "scale", "obs"), cfg),
            method=method,
            reference=ref,
            ref_source=source,
            peak_rss_MiB=r.get("peak_rss_MiB"),
            error=r.get("error"),
        )
        if "error" in r:
            rows.append(row)
            continue
        if method == "dm":
            for eps in EPS:
                row[f"t_{eps:g}"] = r["wall_s"]
            row["value"] = r["value"]
        elif method == "mpstab":
            fixed = next(x for x in r["runs"] if x["samples"] == 0)
            sampled = [x for x in r["runs"] if x["samples"] > 0]
            n = sampled[0]["samples"]
            a = fixed["wall_s"]
            b = max(np.mean([x["wall_s"] for x in sampled]) - a, 0) / n
            s = float(np.mean([x["stderr"] for x in sampled]))
            row.update(
                value=float(np.mean([x["mean"] for x in sampled])),
                stderr=s,
                W2=fixed["W"][2],
                wall_N=float(np.mean([x["wall_s"] for x in sampled])),
                wall_fixed=a,
            )
            for eps in EPS:
                row[f"t_{eps:g}"] = a + b * n * (s / eps) ** 2
        elif method == "pp":
            grid = r["grid"]
            full = [g for g in grid if g["max_weight"] == "Inf"]
            row["value"] = full[-1]["value"] if full else grid[-1]["value"]
            row["pp_tightest_coeff"] = full[-1]["min_abs_coeff"] if full else None
            for eps in EPS:
                ok = [
                    g["wall_s"]
                    for g in grid
                    if ref is not None
                    and abs(g["value"] - ref) <= eps
                    and not (source == "pp" and g is full[-1])
                ]
                if source == "pp" and not ok and full:
                    ok = [full[-1]["wall_s"]]
                row[f"t_{eps:g}"] = min(ok) if ok else np.nan
        elif method == "traj":
            v = np.array(r["values"])
            t = float(np.mean(r["walls"]))
            sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
            row.update(
                value=float(v.mean()),
                stderr=sd / np.sqrt(len(v)),
                n_traj=len(v),
                t_traj=t,
                max_bond=r["max_bond"],
                bond_capped=r["bond_capped"],
            )
            for eps in EPS:
                row[f"t_{eps:g}"] = t * max(1.0, (sd / eps) ** 2)
        rows.append(row)
    return rows


def panel(ax, rows, x, eps, title):
    for method in METHODS:
        pts = sorted(
            (r[x], r.get(f"t_{eps:g}", np.nan)) for r in rows if r["method"] == method
        )
        pts = [(a, b) for a, b in pts if np.isfinite(b)]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(
            xs,
            ys,
            marker=MARKERS[method],
            color=COLORS[method],
            lw=2,
            ms=6,
            label=NAMES[method],
        )
        ax.annotate(
            NAMES[method].split(" ")[0],
            (xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=7,
            color="#52514e",
            va="center",
        )
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure(rows, x, groups, fname, xlabel, xlog=False):
    fig, axes = plt.subplots(
        len(groups),
        len(EPS),
        figsize=(5.2 * len(EPS), 3.6 * len(groups)),
        squeeze=False,
        sharey="row",
    )
    for i, (label, sel) in enumerate(groups):
        for j, eps in enumerate(EPS):
            ax = axes[i][j]
            panel(
                ax, [r for r in rows if sel(r)], x, eps, f"{label}, |error| <= {eps:g}"
            )
            ax.set_xlabel(xlabel)
            if xlog:
                ax.set_xscale("log")
        axes[i][0].set_ylabel("time to accuracy [s], 1 core")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / fname, dpi=150)
    plt.close(fig)


def main():
    rows = [
        row
        for cfg, methods in sorted(load().items())
        for row in summarize(cfg, methods)
    ]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    with open(RESULTS / "compare_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    groups = [
        (
            f"{o}, {s} steps",
            (
                lambda r, o=o, s=s: r["obs"] == o
                and r["steps"] == s
                and r["scale"] == 1.0
            ),
        )
        for o in ("ZZ_mid", "Z_avg")
        for s in (4, 8)
    ]
    figure(rows, "n", groups, "compare_n.png", "qubits n")
    figure(
        rows,
        "steps",
        [
            (
                "n = 20, ZZ_mid",
                lambda r: r["n"] == 20 and r["scale"] == 1.0 and r["obs"] == "ZZ_mid",
            )
        ],
        "compare_steps.png",
        "Trotter steps",
    )
    figure(
        rows,
        "scale",
        [
            (
                "n = 20, 6 steps, ZZ_mid",
                lambda r: r["n"] == 20 and r["steps"] == 6 and r["obs"] == "ZZ_mid",
            )
        ],
        "compare_scale.png",
        "noise rate / device rate",
        xlog=True,
    )

    for r in rows:
        print(
            f"{r['method']:6s} n={r['n']:2d} steps={r['steps']:2d} scale={r['scale']:<4g} "
            f"{r['obs']:6s} t(1e-2)={r.get('t_0.01', np.nan):9.3g} "
            f"t(1e-3)={r.get('t_0.001', np.nan):9.3g} value={r.get('value', np.nan):+.5f} "
            f"ref={r['reference'] if r['reference'] is not None else np.nan:+.5f}({r['ref_source']})"
            + (f" W2={r['W2']:.3f}" if "W2" in r else "")
            + (
                f" bond={r['max_bond']}{'!' if r['bond_capped'] else ''}"
                if "max_bond" in r
                else ""
            )
            + (f" ERROR" if r.get("error") else "")
        )


if __name__ == "__main__":
    main()
