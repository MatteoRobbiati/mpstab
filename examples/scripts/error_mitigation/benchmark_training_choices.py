"""Benchmark replacement_probability and replacement_method of the training set.

The defaults disagree today: 0.9 / "closest" in :func:`cdr`, :func:`vncdr` and
:func:`training_circuits`, 0.7 / "closest" in :class:`VnCDRMitigation`. This
script sweeps both choices on small models, with every noisy value sampled from
a :class:`DensityMatrixNoiseSimulator` under depolarizing noise and asymmetric
readout errors, so the defaults can be chosen from the table it prints:

- ``cdr`` and ``vncdr``: the noise is a callable running each training circuit
  on the simulator, the noiseless values come from HSMPO;
- ``plan``: :class:`VnCDRMitigation` on the head of an HQSMPO plan, run with
  :func:`execute_plan` on the simulator.

Columns, per model, method and configuration, over ``--n-seeds`` seeds:

- ``err_mean``, ``err_std``: ``|mitigated - exact|``;
- ``distinct``: distinct training circuits (``cdr``, ``vncdr``) or distinct
  candidates the plan picks its training set from (``plan``), out of ``draws``;
- ``failed``: seeds where the plan raised the "distinct training circuits"
  error (always 0 for ``cdr`` and ``vncdr``, which keep duplicates);
- ``exact_min``, ``exact_max``: range of the noiseless training values;
- ``zero_frac``: fraction of training values that are zero;
- ``hsmpo_s``: classical time of the training set, in seconds: HSMPO on the
  training circuits for ``cdr`` and ``vncdr``, the whole ``build_plan`` (HSMPO
  on every candidate and the tail fold) for ``plan``.

Usage::

    python examples/scripts/error_mitigation/benchmark_training_choices.py
    python examples/scripts/error_mitigation/benchmark_training_choices.py \\
        --n-seeds 5 --shots 8000 --csv results.csv
"""

import argparse
import csv
import pathlib
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from qibo import gates, set_backend

from mpstab import (
    HQSMPO,
    HSMPO,
    DensityMatrixNoiseSimulator,
    DepolarizingError,
    NoiseModel,
    ReadoutError,
    VnCDRMitigation,
    cdr,
    execute_plan,
    fold_circuit,
    vncdr,
)
from mpstab.models.ansatze import CircuitAnsatz, HardwareEfficient
from mpstab.qpu.measurement.abstract import basis_measurement

HERE = pathlib.Path(__file__).parent
PROBABILITIES = (0.6, 0.7, 0.8, 0.9)
METHODS = ("closest", "random")
LEVELS = (1, 3, 5)
ZERO = 1e-12


def noise_model():
    """Depolarizing noise after every gate, stronger on CZ, asymmetric readout."""
    return (
        NoiseModel()
        .add(DepolarizingError(5e-3))
        .add(DepolarizingError(2e-2), gates.CZ)
        .add(ReadoutError(0.02, 0.06))
    )


def circuit_models():
    """``(name, ansatz, observable)`` of the models mitigated with cdr and vncdr."""
    np.random.seed(1)
    small = HardwareEfficient(nqubits=3, nlayers=2)
    np.random.seed(2)
    large = HardwareEfficient(nqubits=4, nlayers=3)
    return [("HE-3q-2l", small, "ZZI"), ("HE-4q-3l", large, "IZZI")]


def plan_model():
    """``(name, HQSMPO, observable, cut)`` of the model mitigated with a plan."""
    np.random.seed(3)
    model = HQSMPO(
        HardwareEfficient(nqubits=4, nlayers=3),
        head_synthesis="extraction",
        progress=False,
    )
    return "HQSMPO-4q-3l", model, "IZZI", int(model.n_magic_gates * 0.6)


class Device:
    """
    ``f(circuit, level=1)``: the noisy value of a Z-string observable, from
    shots of the circuit folded to ``level`` on a density-matrix simulator.
    """

    def __init__(self, observable, noise, shots, seed):
        self.support = [q for q, label in enumerate(observable) if label == "Z"]
        self.nqubits = len(observable)
        self.simulator = DensityMatrixNoiseSimulator(noise, seed=seed)
        self.shots = shots

    def __call__(self, circuit, level=1):
        folded = fold_circuit(circuit, level) if level != 1 else circuit
        measured = folded + basis_measurement("Z" * self.nqubits, self.nqubits)
        (frequencies,) = self.simulator.execute_circuits([measured], self.shots)
        parity = sum(
            count * (-1) ** sum(int(bits[q]) for q in self.support)
            for bits, count in frequencies.items()
        )
        return parity / self.shots


def distinct(circuits) -> int:
    """Number of distinct circuits, by gate names, qubits and angles."""
    return len(
        {
            tuple(
                (gate.name, gate.qubits, tuple(np.round(np.ravel(gate.parameters), 12)))
                for gate in circuit.queue
            )
            for circuit in circuits
        }
    )


def hsmpo_time(circuits, observable) -> float:
    """Seconds to get the noiseless values of ``circuits`` with HSMPO."""
    start = time.perf_counter()
    for circuit in circuits:
        HSMPO(circuit, progress=False).expectation(observable)
    return time.perf_counter() - start


def run_circuit_method(method, ansatz, observable, p, rm, seed, args):
    """One cdr or vncdr run: the row of the raw results."""
    noise = noise_model()
    device = Device(observable, noise, args.shots, seed)
    exact = HSMPO(ansatz, progress=False).expectation(observable)
    options = dict(
        n_training_circuits=args.n_training,
        replacement_probability=p,
        replacement_method=rm,
        seed=seed,
    )
    if method == "cdr":
        noise_map = cdr(ansatz, observable, device, **options)
        mitigated = noise_map(device(ansatz.circuit))
    else:
        noise_map = vncdr(ansatz, observable, device, noise_levels=LEVELS, **options)
        mitigated = noise_map([device(ansatz.circuit, level) for level in LEVELS])
    return dict(
        error=abs(float(mitigated) - exact),
        distinct=distinct(noise_map.circuits),
        draws=len(noise_map.circuits),
        failed=0,
        training_exact=np.asarray(noise_map.exact),
        hsmpo_s=hsmpo_time(noise_map.circuits, observable),
    )


def run_plan(model, observable, cut, p, rm, seed, args):
    """One VnCDRMitigation run on the HQSMPO plan: the row of the raw results."""
    mitigation = VnCDRMitigation(
        n_training_circuits=args.n_training,
        replacement_probability=p,
        replacement_method=rm,
        noise_levels=LEVELS,
    )
    head = model.resynthesize_head(
        cut, observable=observable if model.light_cone else None
    ).circuit
    draws = mitigation.n_training_circuits * mitigation.candidate_factor
    # The plan draws its own candidates; these, drawn the same way with
    # another seed, count how many distinct ones such a draw gives.
    ansatz, rng = CircuitAnsatz(qibo_circuit=head), random.Random(seed)
    n_distinct = distinct(
        ansatz.partitionate_circuit(p, rm, rng=rng)[1] for _ in range(draws)
    )
    row = dict(distinct=n_distinct, draws=draws)
    start = time.perf_counter()
    try:
        plan = mitigation.build_plan(
            model, observable, cut, n_shots=args.plan_shots, seed=seed
        )
    except ValueError as error:
        if "distinct training circuits" not in str(error):
            raise
        return dict(
            row,
            error=np.nan,
            failed=1,
            training_exact=np.array([]),
            hsmpo_s=np.nan,
        )
    row["hsmpo_s"] = time.perf_counter() - start
    result = plan.estimate(
        execute_plan(DensityMatrixNoiseSimulator(noise_model(), seed=seed), plan)
    )
    _, coefficients = plan.target.recombination
    reference = (
        HSMPO(head, progress=False).expectation(coefficients) + plan.target.constant
    )
    return dict(
        row,
        error=abs(result.value - reference),
        failed=0,
        training_exact=np.asarray(plan.training_exact),
    )


def use_numpy():
    """Density matrices of a few qubits: numpy beats the threaded qibojit."""
    set_backend("numpy")


def run_job(p, rm, seed, args):
    """Every model at one configuration and seed: rows of the raw results."""
    rows = []
    for name, ansatz, observable in circuit_models():
        for method in ("cdr", "vncdr"):
            row = run_circuit_method(method, ansatz, observable, p, rm, seed, args)
            rows.append(dict(row, model=name, method=method, p=p, rm=rm))
    plan_name, model, observable, cut = plan_model()
    row = run_plan(model, observable, cut, p, rm, seed, args)
    rows.append(dict(row, model=plan_name, method="plan", p=p, rm=rm))
    return rows


def summarize(raw):
    """One row per (model, method, p, rm), aggregated over seeds."""
    groups = defaultdict(list)
    for row in raw:
        groups[(row["model"], row["method"], row["p"], row["rm"])].append(row)
    rows = []
    for (model, method, p, rm), runs in groups.items():
        errors = np.array([run["error"] for run in runs], dtype=float)
        exact = np.concatenate([run["training_exact"] for run in runs])
        ok = errors[np.isfinite(errors)]
        rows.append(
            dict(
                model=model,
                method=method,
                p=p,
                rm=rm,
                err_mean=float(ok.mean()) if len(ok) else np.nan,
                err_std=float(ok.std()) if len(ok) else np.nan,
                distinct=float(np.mean([run["distinct"] for run in runs])),
                draws=runs[0]["draws"],
                failed=int(sum(run["failed"] for run in runs)),
                exact_min=float(exact.min()) if len(exact) else np.nan,
                exact_max=float(exact.max()) if len(exact) else np.nan,
                zero_frac=(
                    float(np.mean(np.abs(exact) <= ZERO)) if len(exact) else np.nan
                ),
                hsmpo_s=float(np.nanmean([run["hsmpo_s"] for run in runs])),
                seeds=len(runs),
            )
        )
    return rows


COLUMNS = [
    ("model", "{:<13}"),
    ("method", "{:<6}"),
    ("p", "{:>4.1f}"),
    ("rm", "{:<7}"),
    ("err_mean", "{:>9.4f}"),
    ("err_std", "{:>8.4f}"),
    ("distinct", "{:>8.1f}"),
    ("draws", "{:>5d}"),
    ("failed", "{:>6d}"),
    ("exact_min", "{:>9.3f}"),
    ("exact_max", "{:>9.3f}"),
    ("zero_frac", "{:>9.2f}"),
    ("hsmpo_s", "{:>8.3f}"),
]


def print_table(rows):
    header = "  ".join(
        f"{name:<{len(fmt.format(rows[0][name]))}}" for name, fmt in COLUMNS
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(fmt.format(row[name]) for name, fmt in COLUMNS))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--n-seeds", type=int, default=3, help="seeds per config")
    parser.add_argument(
        "--n-training", type=int, default=16, help="training circuits per run"
    )
    parser.add_argument(
        "--shots", type=int, default=4000, help="shots per cdr/vncdr circuit"
    )
    parser.add_argument(
        "--plan-shots", type=int, default=20000, help="target shots of the plan"
    )
    parser.add_argument(
        "--probabilities", type=float, nargs="+", default=list(PROBABILITIES)
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument(
        "--workers", type=int, default=4, help="processes running the configurations"
    )
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=HERE / "benchmark_training_choices.csv",
        help="where to write the summary table",
    )
    args = parser.parse_args()

    jobs = [
        (p, rm, args.seed + k, args)
        for p in args.probabilities
        for rm in args.methods
        for k in range(args.n_seeds)
    ]
    raw = []
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=use_numpy) as pool:
        for (p, rm, seed, _), rows in zip(jobs, pool.map(run_job, *zip(*jobs))):
            raw += rows
            print(
                f"p={p} {rm} seed={seed}: done ({time.perf_counter() - start:.0f} s)",
                flush=True,
            )

    rows = sorted(
        summarize(raw), key=lambda r: (r["model"], r["method"], r["rm"], r["p"])
    )
    print()
    print_table(rows)
    with open(args.csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWritten to {args.csv}")


if __name__ == "__main__":
    main()
