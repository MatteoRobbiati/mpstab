"""mpstab against other simulators on local Pauli noise plus readout error.

Model (common to every method): kicked-Ising circuit (common.kicked_ising_layered,
CNOT native, RZ virtual and noiseless); single-qubit depolarizing
PauliError({X, Y, Z: p/3}) on every qubit of every RX (p = p1q * scale) and of
every CNOT (p = p2q * scale); ReadoutError(p0, p1) on every qubit. Readout acts on
the measured bits, so every method gets the same readout-folded observable
(NoiseModel.readout_observable): a constant plus Pauli terms.

Methods, each run in its own subprocess (one core, timed after imports and,
for Julia, after a compilation run):

- ``mpstab``: HSMPO.noisy_expectation (k=0, 1 exact, k>=2 sampled with control
  variates). One run at n_samples=0 (fixed cost a) and --repeats seeds at
  --samples; time to stderr <= eps is a + b N_eps, N_eps = N (stderr/eps)^2.
- ``pp``: PauliPropagation.jl 0.4.1 through juliacall, unmodified. Grid over
  min_abs_coeff (1e-2 ... 1e-9, loose to tight) and max_weight (Inf, 6, 10);
  the time to eps is the fastest setting whose error is <= eps. A setting
  slower than --budget stops the tighter ones.
- ``traj``: Pauli-fault trajectories on quimb's CircuitMPS (cutoff 1e-10,
  --traj-bond max bond): each trajectory samples the faults and returns the
  observable exactly; time to eps is t_traj (std/eps)^2 from --traj trajectories.
- ``dm``: qibo density matrix of NoiseModel.apply (n <= --dm-max).

Reference: the density matrix where it runs; otherwise the tightest PP setting,
accepted only when it moved by less than 1e-4 from the previous one (flagged
in the output as ``ref_source``).

Usage:
  PYTHONPATH=<mpstab>/src python bench_compare.py --scan n|steps|scale|all \
      [--methods mpstab pp traj dm]
Results: results/compare.jsonl (resumable), then --plot writes
results/compare_*.png and results/compare_summary.csv.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = RESULTS / "compare.jsonl"

SCANS = {
    "n": [
        dict(n=n, steps=s, scale=1.0, obs=o)
        for o in ("ZZ_mid", "Z_avg")
        for s in (4, 8)
        for n in (10, 12, 20, 30, 40, 50)
    ],
    "steps": [
        dict(n=20, steps=s, scale=1.0, obs="ZZ_mid") for s in (1, 2, 4, 6, 8, 10, 12)
    ],
    "scale": [dict(n=20, steps=6, scale=x, obs="ZZ_mid") for x in (0.1, 0.3, 1.0, 3.0)],
}


# -- The model -------------------------------------------------------------------


def build(n, steps, scale, obs, p1q=1e-3, p2q=5e-3, p0=0.01, p1=0.03):
    from common import kicked_ising_layered
    from qibo import gates

    from mpstab import NoiseModel, PauliError, ReadoutError

    def local(p):
        return PauliError({"X": p / 3, "Y": p / 3, "Z": p / 3})

    circuit = kicked_ising_layered(n, steps)
    noise = (
        NoiseModel()
        .add(local(p1q * scale), gates.RX)
        .add(local(p2q * scale), gates.CNOT)
        .add(ReadoutError(p0, p1))
    )
    if obs == "ZZ_mid":
        labels = ["I"] * n
        labels[n // 2 - 1] = labels[n // 2] = "Z"
        observable = "".join(labels)
    elif obs == "Z_avg":
        observable = {"I" * q + "Z" + "I" * (n - q - 1): 1 / n for q in range(n)}
    else:
        raise ValueError(obs)
    constant, terms = noise.readout_observable(observable, n)
    return circuit, noise, observable, constant, terms


def peak_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


# -- Methods (run inside the worker process) --------------------------------------


def run_mpstab(cfg, args):
    from mpstab import HSMPO

    circuit, noise, observable, _, _ = build(**cfg)
    runs = []
    for samples, seed in [(0, 0)] + [(args.samples, s) for s in range(args.repeats)]:
        start = time.perf_counter()
        hsmpo = HSMPO.rotations_only(circuit)
        r = hsmpo.noisy_expectation(observable, noise, n_samples=samples, seed=seed)
        runs.append(
            dict(
                samples=samples,
                seed=seed,
                mean=r.mean,
                stderr=r.stderr,
                wall_s=time.perf_counter() - start,
                W=list(r.stratum_weights),
                n_patterns=r.n_patterns,
            )
        )
    return dict(runs=runs, peak_rss_MiB=peak_mib())


def _pp_gate(gate):
    q = [x + 1 for x in gate.qubits]
    name = type(gate).__name__
    if name in ("RX", "RY", "RZ"):
        return [f"PauliRotation(:{name[1]}, {q[0]}, {float(gate.parameters[0])!r})"]
    if name == "CNOT":
        return [f"CliffordGate(:CNOT, {q})"]
    if name == "H":
        return [f"CliffordGate(:H, {q})"]
    raise NotImplementedError(name)


def _pp_noise(event):
    # Single-qubit depolarizing {X,Y,Z: p/3}: Pauli eigenvalue 1 - 4p/3, which is
    # PauliPropagation's DepolarizingNoise(q, 4p/3).
    (q,) = event.qubits
    probs = [event.probs.get(k, 0.0) for k in "XYZ"]
    if not np.allclose(probs, probs[0]):
        raise NotImplementedError("only symmetric single-qubit depolarizing")
    return [f"DepolarizingNoise({q + 1}, {4 * probs[0]!r})"]


def run_pp(cfg, args):
    from juliacall import Main as jl

    jl.seval("using PauliPropagation")
    circuit, noise, _, constant, terms = build(**cfg)
    by_gate = {}
    for event in noise.events(circuit):
        by_gate.setdefault(event.after_gate, []).append(event)
    items = [g for e in by_gate.get(-1, ()) for g in _pp_noise(e)]
    for index, gate in enumerate(circuit.queue):
        items += _pp_gate(gate)
        items += [g for e in by_gate.get(index, ()) for g in _pp_noise(e)]
    jl.seval("circ = Gate[" + ", ".join(items) + "]")
    lines = [f"let H = PauliSum({circuit.nqubits})"]
    for pauli, c in terms.items():
        support = [q for q, p in enumerate(pauli) if p != "I"]
        lines.append(
            f"add!(H, [{', '.join(':' + pauli[q] for q in support)}], "
            f"[{', '.join(str(q + 1) for q in support)}], {float(c)!r})"
        )
    jl.seval("obs = " + "; ".join(lines) + "; H end")
    # compilation run, not timed
    jl.seval("overlapwithzero(propagate(circ, obs; min_abs_coeff=1e-1))")
    grid = []
    for max_weight in ("Inf", "10", "6"):
        for exponent in range(2, 10):
            coeff = 10.0**-exponent
            start = time.perf_counter()
            value = float(
                jl.seval(
                    f"begin out = propagate(circ, obs; max_weight={max_weight}, "
                    f"min_abs_coeff={coeff!r}); overlapwithzero(out) end"
                )
            )
            wall = time.perf_counter() - start
            grid.append(
                dict(
                    max_weight=max_weight,
                    min_abs_coeff=coeff,
                    value=value + constant,
                    wall_s=wall,
                    n_strings=int(jl.seval("length(out)")),
                    peak_rss_MiB=peak_mib(),
                )
            )
            if wall > args.budget or peak_mib() > args.max_rss_mib:
                break
    return dict(grid=grid, peak_rss_MiB=peak_mib())


_QUIMB = {"RX": "RX", "RY": "RY", "RZ": "RZ", "CNOT": "CNOT", "H": "H"}
_PAULI = {
    "X": np.array([[0, 1], [1, 0]], complex),
    "Y": np.array([[0, -1j], [1j, 0]]),
    "Z": np.array([[1, 0], [0, -1]], complex),
}


def run_traj(cfg, args):
    import quimb.tensor as qtn

    circuit, noise, _, constant, terms = build(**cfg)
    events = noise.events(circuit)
    by_gate = {}
    for event in events:
        by_gate.setdefault(event.after_gate, []).append(event)
    ops = []
    for t in terms:
        support = [q for q, p in enumerate(t) if p != "I"]
        if len(support) == 1:
            op = _PAULI[t[support[0]]]
        elif len(support) == 2 and support[1] == support[0] + 1:
            op = np.kron(_PAULI[t[support[0]]], _PAULI[t[support[1]]])
        else:
            raise NotImplementedError(t)
        ops.append((op, tuple(support), terms[t]))
    rng = np.random.default_rng(0)

    def sample_faults(index):
        out = []
        for event in by_gate.get(index, ()):
            labels, probs = zip(*event.probs.items())
            u = rng.random()
            acc = 0.0
            for label, p in zip(labels, probs):
                acc += p
                if u < acc:
                    out.append((label, event.qubits[0]))
                    break
        return out

    values, walls, bonds = [], [], []
    deadline = time.perf_counter() + args.budget
    for _ in range(args.traj):
        start = time.perf_counter()
        psi = qtn.CircuitMPS(circuit.nqubits, max_bond=args.traj_bond, cutoff=1e-10)
        for label, q in sample_faults(-1):
            psi.apply_gate(label, q)
        for index, gate in enumerate(circuit.queue):
            name = _QUIMB[type(gate).__name__]
            params = [float(x) for x in gate.parameters]
            psi.apply_gate(name, *params, *gate.qubits)
            for label, q in sample_faults(index):
                psi.apply_gate(label, q)
        value = constant + sum(
            c * float(np.real(psi.local_expectation(op, where))) for op, where, c in ops
        )
        walls.append(time.perf_counter() - start)
        values.append(value)
        bonds.append(int(psi.psi.max_bond()))
        if time.perf_counter() > deadline and len(values) >= 3:
            break
    return dict(
        values=values,
        walls=walls,
        max_bond=max(bonds),
        bond_capped=max(bonds) >= args.traj_bond,
        peak_rss_MiB=peak_mib(),
    )


def run_dm(cfg, args):
    circuit, noise, _, constant, terms = build(**cfg)
    start = time.perf_counter()
    rho = np.asarray(noise.apply(circuit)().state())
    n = circuit.nqubits
    idx = np.arange(2**n)
    value = constant
    for pauli, c in terms.items():
        if set(pauli) - {"I", "Z"}:
            raise NotImplementedError(pauli)
        signs = np.ones(2**n)
        for q, p in enumerate(pauli):
            if p == "Z":
                signs *= 1 - 2 * ((idx >> (n - 1 - q)) & 1)
        value += c * float(np.real(np.diagonal(rho) @ signs))
    return dict(
        value=value, wall_s=time.perf_counter() - start, peak_rss_MiB=peak_mib()
    )


METHODS = {"mpstab": run_mpstab, "pp": run_pp, "traj": run_traj, "dm": run_dm}


# -- Driver -----------------------------------------------------------------------


def free_memory_percent():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return 100


def drive(args):
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            r = json.loads(line)
            done.add((r["method"], r["n"], r["steps"], r["scale"], r["obs"]))
    scans = list(SCANS) if args.scan == "all" else [args.scan]
    configs = []
    for scan in scans:
        for cfg in SCANS[scan]:
            if cfg not in configs:
                configs.append(cfg)
    for cfg in configs:
        for method in args.methods:
            if method == "dm" and cfg["n"] > args.dm_max:
                continue
            if method == "traj" and cfg["obs"] != "ZZ_mid":
                continue
            key = (method, cfg["n"], cfg["steps"], cfg["scale"], cfg["obs"])
            if key in done:
                continue
            if free_memory_percent() < 25:
                sys.exit("memory pressure: stopping")
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                method,
                "--cfg",
                json.dumps(cfg),
                "--samples",
                str(args.samples),
                "--repeats",
                str(args.repeats),
                "--budget",
                str(args.budget),
                "--traj",
                str(args.traj),
                "--traj-bond",
                str(args.traj_bond),
                "--max-rss-mib",
                str(args.max_rss_mib),
            ]
            start = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd, cwd=HERE, capture_output=True, text=True, timeout=args.timeout
                )
                lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
                payload = (
                    json.loads(lines[-1]) if lines else dict(error=proc.stderr[-600:])
                )
            except subprocess.TimeoutExpired:
                payload = dict(error=f"timeout {args.timeout}s")
            record = dict(
                method=method, **cfg, total_s=time.perf_counter() - start, **payload
            )
            with OUT.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{method:6s} {cfg} {record['total_s']:.0f}s"
                + (f" ERROR {payload['error'][-200:]}" if "error" in payload else ""),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", default="all", choices=["all", *SCANS])
    parser.add_argument("--methods", nargs="+", default=["dm", "mpstab", "pp", "traj"])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--budget",
        type=float,
        default=120.0,
        help="PP: stop tightening after a setting this slow; traj: time cap",
    )
    parser.add_argument("--traj", type=int, default=50)
    parser.add_argument("--traj-bond", type=int, default=256)
    parser.add_argument("--max-rss-mib", type=float, default=6000)
    parser.add_argument("--dm-max", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--worker")
    parser.add_argument("--cfg")
    args = parser.parse_args()
    if args.worker:
        sys.path.insert(0, str(HERE))
        print(json.dumps(METHODS[args.worker](json.loads(args.cfg), args)))
        return
    drive(args)


if __name__ == "__main__":
    main()
