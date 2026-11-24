"""Gradients of mpstab against PauliPropagation.jl with reverse-mode autodiff.

Both differentiate the energy of the transverse-field Ising ring
``H = - sum_i Z_i Z_{i+1} - sum_i X_i`` on the
:class:`~mpstab.models.ansatze.HardwareEfficient` ansatz (per layer, an RY on
every qubit then a ring of CZ), at the same random parameters:

- mpstab: :meth:`~mpstab.HSMPO.value_and_gradient`, the adjoint gradient on
  the hybrid stabilizer-MPS surrogate, at bond dimension ``chi``;
- PauliPropagation.jl: the Hamiltonian propagated backwards through the
  circuit, differentiated with a compiled ReverseDiff tape as in the package's
  own autodiff example. Only truncations that do not depend on the parameters
  survive in a compiled tape, so the knobs are ``max_weight`` and
  ``max_freq``; ``min_abs_coeff`` is set to zero. The tape is recorded once,
  at the evaluation point (recorded at all-zero angles it drops branches),
  and its compilation is reported separately: a training loop pays it once.

The reference is PauliPropagation with no truncation, which is exact on these
shallow circuits, checked against mpstab with no bond cap and no SVD cutoff up
to ``--exact-up-to`` qubits. The gradient error is the relative Euclidean
error. Each method also reports its own truncation estimate, available
without the exact answer: ``1 - F`` for mpstab, the smallest truncation
fidelity of its state and costate; for PauliPropagation, its ``estimatemse``
certificate per Hamiltonian term (Monte Carlo over the angles), combined into
an RMS energy error ``sum_t |c_t| sqrt(mse_t)``. Both estimate the error of the
energy rather than of the gradient. A configuration is dropped at larger
sizes once one of its gradients, or tape compilations, takes longer than
``--budget`` seconds. Untruncated Pauli propagation grows exponentially with
depth: keep ``--nlayers`` small, it ran out of memory at 6 layers on 12 qubits
on a 16 GB machine.

Needs juliacall with PauliPropagation.jl and ReverseDiff.jl, which the
script installs through juliapkg::

    uv run python examples/scripts/gradient/gradient_paulipropagation.py
"""

import argparse
import json
import pathlib
import sys
import time

import juliapkg
import numpy as np

from mpstab import HSMPO
from mpstab.engines import QuimbEngine
from mpstab.evolutors.gradient import analytic_gradient
from mpstab.models.ansatze import HardwareEfficient

#: juliapkg UUIDs of the Julia packages.
JULIA_PACKAGES = {
    "PauliPropagation": "293282d5-3c99-4fb6-92d0-fd3280a19750",
    "ReverseDiff": "37e2e3b7-166d-5795-8a7a-e32c996b4267",
}


def tfim_terms(nqubits: int) -> dict:
    """The Hamiltonian as Pauli strings, for mpstab."""
    terms = {}
    for q in range(nqubits):
        zz = ["I"] * nqubits
        zz[q] = zz[(q + 1) % nqubits] = "Z"
        terms["".join(zz)] = -1.0
        x = ["I"] * nqubits
        x[q] = "X"
        terms["".join(x)] = -1.0
    return terms


class PauliPropagationGradient:
    """The ReverseDiff gradient of the same energy in PauliPropagation.jl."""

    def __init__(self):
        for name, uuid in JULIA_PACKAGES.items():
            juliapkg.add(name, uuid)
        juliapkg.resolve()
        from juliacall import Main as jl  # after juliapkg has set up Julia

        jl.seval("using PauliPropagation, ReverseDiff")
        jl.seval("using ReverseDiff: GradientTape, gradient!, compile")
        self.jl = jl

    def prepare(self, circuit, parameters, max_weight, max_freq) -> float:
        """Define the loss and compile its tape at ``parameters``; returns the time."""
        n = circuit.nqubits
        gates = []
        for gate in circuit.queue:
            if gate.name == "ry":
                gates.append(f"PauliRotation(:Y, {gate.target_qubits[0] + 1})")
            elif gate.name == "cz":
                q0, q1 = gate.qubits
                gates.append(f"CliffordGate(:CZ, ({q0 + 1}, {q1 + 1}))")
            else:
                raise ValueError(f"Unexpected gate {gate.name}")
        jl = self.jl
        jl.seval(f"circ = Gate[{', '.join(gates)}]")
        jl.seval(
            f"""
        function loss(thetas)
            T = eltype(thetas)
            H = PauliSum(T, {n})
            for q in 1:{n}
                add!(H, :X, q, T(-1.0))
                add!(H, [:Z, :Z], [q, mod(q, {n}) + 1], T(-1.0))
            end
            out = propagate!(circ, wrapcoefficients(H, PauliFreqTracker), thetas;
                             max_weight={max_weight}, max_freq={max_freq},
                             min_abs_coeff=0.0)
            return overlapwithzero(out)
        end"""
        )
        self.set_thetas(parameters)
        start = time.perf_counter()
        jl.seval("tape = compile(GradientTape(loss, thetas)); buf = similar(thetas)")
        return time.perf_counter() - start

    def set_thetas(self, parameters):
        values = ", ".join(repr(float(x)) for x in parameters)
        self.jl.seval(f"thetas = Float64[{values}]")

    def gradient(self, parameters) -> np.ndarray:
        self.set_thetas(parameters)
        self.jl.seval("gradient!(buf, tape, thetas)")
        return np.array(self.jl.seval("buf"))

    def error_estimate(self, n, max_weight, max_freq, samples) -> float:
        """
        PauliPropagation's own truncation certificate: ``estimatemse`` per
        Hamiltonian term (Monte Carlo over the angles), combined into an RMS
        energy error ``sum_t |c_t| sqrt(mse_t)``. Uses the circuit of the last
        :meth:`prepare`.
        """
        return float(
            self.jl.seval(
                f"""
            sum(
                sqrt(estimatemse(circ, term, {samples}; max_weight={max_weight},
                                 max_freq={max_freq}))
                for term in vcat(
                    [PauliString({n}, :X, q) for q in 1:{n}],
                    [PauliString({n}, [:Z, :Z], [q, mod(q, {n}) + 1]) for q in 1:{n}],
                )
            )"""
            )
        )


def relative_error(gradient, exact) -> float:
    """``|g - g_exact| / |g_exact|``, Euclidean norms."""
    return float(np.linalg.norm(gradient - exact) / np.linalg.norm(exact))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--nqubits", type=int, nargs="+", default=[8, 12, 16, 20, 32, 40, 60]
    )
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--bonds", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument(
        "--truncations",
        nargs="+",
        default=["6:inf", "5:25"],
        help="PauliPropagation max_weight:max_freq pairs, next to the untruncated one",
    )
    parser.add_argument("--exact-up-to", type=int, default=16)
    parser.add_argument(
        "--reference",
        choices=["pp", "mpstab"],
        default="pp",
        help="pp: untruncated Pauli propagation (exact, shallow circuits only); "
        "mpstab: exact HSMPO up to --exact-up-to qubits, --reference-bond beyond",
    )
    parser.add_argument("--reference-bond", type=int, default=256)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="gradient_paulipropagation.json")
    parser.add_argument("--plot", default="gradient_paulipropagation.png")
    args = parser.parse_args()

    pp = PauliPropagationGradient()
    results, dropped = [], set()
    print(f"TFIM ring, HardwareEfficient with {args.nlayers} layers\n")
    for n in args.nqubits:
        np.random.seed(args.seed)
        circuit = HardwareEfficient(nqubits=n, nlayers=args.nlayers).circuit
        parameters = np.array([p[0] for p in circuit.get_parameters()])
        terms = tfim_terms(n)
        print(f"n = {n}, {len(parameters)} parameters")

        if args.reference == "pp":
            # Untruncated Pauli propagation is exact: it is the reference.
            compile_time = pp.prepare(circuit, parameters, n, "Inf")
            pp.gradient(parameters)  # warm-up
            start = time.perf_counter()
            exact = pp.gradient(parameters)
            elapsed = time.perf_counter() - start
            results.append(
                dict(n=n, method="PP exact", time=elapsed, compile=compile_time)
            )
            check = ""
            if n <= args.exact_up_to:
                reference = HSMPO(circuit, progress=False)
                reference.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
                _, exact_mpstab = reference.value_and_gradient(terms)
                check = f"  (vs exact HSMPO {relative_error(exact, exact_mpstab):.0e})"
            print(
                f"  {'PP no truncation':>26}: {elapsed:8.3f} s "
                f"(tape {compile_time:6.1f} s)  reference{check}"
            )
        else:
            if n <= args.exact_up_to:
                reference = HSMPO(circuit, progress=False)
                reference.set_engines(tn_engine=QuimbEngine(svd_cutoff=0.0))
                label, costate = "exact HSMPO", None
            else:
                reference = HSMPO(
                    circuit, max_bond_dimension=args.reference_bond, progress=False
                )
                label, costate = (
                    f"mpstab chi={args.reference_bond}",
                    args.reference_bond,
                )
            start = time.perf_counter()
            result = analytic_gradient(reference, terms, costate_bond=costate)
            exact = result.values
            print(
                f"  reference {label}: {time.perf_counter() - start:.1f} s, "
                f"1-F {1 - result.fidelities.min():.1e}"
            )

        for chi in args.bonds:
            name = f"mpstab chi={chi}"
            if name in dropped:
                continue
            hsmpo = HSMPO(circuit, max_bond_dimension=chi, progress=False)
            analytic_gradient(hsmpo, terms)  # warm-up
            start = time.perf_counter()
            result = analytic_gradient(hsmpo, terms)
            elapsed = time.perf_counter() - start
            error = relative_error(result.values, exact)
            estimate = float(1 - result.fidelities.min())
            results.append(
                dict(n=n, method=name, time=elapsed, error=error, estimate=estimate)
            )
            print(
                f"  {name:>26}: {elapsed:8.3f} s                "
                f"rel. error {error:.1e}   1-F {estimate:.1e}"
            )
            if elapsed > args.budget:
                dropped.add(name)

        for truncation in args.truncations:
            weight, freq = truncation.split(":")
            max_weight, max_freq = int(weight), ("Inf" if freq == "inf" else freq)
            name = f"PP max_weight={max_weight} max_freq={max_freq}"
            if name in dropped:
                continue
            compile_time = pp.prepare(circuit, parameters, max_weight, max_freq)
            pp.gradient(parameters)  # warm-up
            start = time.perf_counter()
            gradient = pp.gradient(parameters)
            elapsed = time.perf_counter() - start
            error = relative_error(gradient, exact)
            estimate = pp.error_estimate(n, max_weight, max_freq, args.samples)
            results.append(
                dict(
                    n=n,
                    method=name,
                    time=elapsed,
                    compile=compile_time,
                    error=error,
                    estimate=estimate,
                )
            )
            print(
                f"  {name:>26}: {elapsed:8.3f} s (tape {compile_time:6.1f} s)"
                f"  rel. error {error:.1e}   est. RMS energy error {estimate:.1e}"
            )
            if max(elapsed, compile_time) > args.budget:
                dropped.add(name)

    with open(args.output, "w") as file:
        json.dump(results, file, indent=2)
    print(f"\nresults saved to {args.output}")
    plot(results, args)


def plot(results, args):
    """Time, gradient error and each method's own truncation estimate, against n."""
    import matplotlib.pyplot as plt

    # examples/, home of the plot style

    sys.path.append(str(pathlib.Path(__file__).parents[2]))

    from plotting import COLORS, set_style

    set_style()
    chis = sorted(
        {
            int(r["method"].split("=")[1])
            for r in results
            if r["method"].startswith("mpstab")
        }
    )
    palette = iter([COLORS["red"], COLORS["orange"]])
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    for method in dict.fromkeys(r["method"] for r in results):
        rows = [r for r in results if r["method"] == method]
        if method.startswith("mpstab"):
            rank = chis.index(int(method.split("=")[1]))
            style = dict(
                color=COLORS["blue"],
                alpha=0.3 + 0.7 * rank / max(len(chis) - 1, 1),
                marker="o",
            )
        elif method == "PP exact":
            style = dict(color=COLORS["black"], marker="s")
        else:
            style = dict(color=next(palette), marker="s")
        style["label"] = method.replace("PP", "PauliPropagation")
        n = [r["n"] for r in rows]
        axes[0].semilogy(n, [r["time"] for r in rows], **style)
        if "error" in rows[0]:
            axes[1].semilogy(n, [r["error"] for r in rows], **style)
            # A certificate of exactly zero means that no truncated path was
            # sampled; it has no place on a log scale.
            shown = [r for r in rows if r["estimate"] > 0]
            axes[2].semilogy(
                [r["n"] for r in shown], [r["estimate"] for r in shown], **style
            )
    axes[0].set_ylabel("Time per gradient [s]")
    axes[1].set_ylabel(r"Gradient error $\|\Delta g\| / \|g\|$")
    axes[2].set_ylabel("Own truncation estimate")
    axes[2].text(
        0.01,
        0.02,
        "PauliPropagation: Monte Carlo certificate, resolution "
        f"~$1/\\sqrt{{{args.samples}}}$; zero estimates omitted",
        transform=axes[2].transAxes,
        fontsize=10,
        color=COLORS["gray"],
    )
    axes[2].set_xlabel("Qubits $n$")
    axes[0].set_title(
        f"Ising ring, HardwareEfficient with {args.nlayers} layers\n"
        "estimate: $1 - F$ (mpstab), estimated RMS energy error (PauliPropagation)"
    )
    axes[0].legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.savefig(args.plot)
    print(f"plot saved to {args.plot}")
    plot_tradeoff(results, args)


def plot_tradeoff(results, args, sizes=(16, 20, 32, 40, 60)):
    """
    Gradient error against time per gradient, one panel per width: every
    configuration is a point, so methods compare at equal accuracy.
    """
    import matplotlib.pyplot as plt

    # examples/, home of the plot style

    sys.path.append(str(pathlib.Path(__file__).parents[2]))

    from plotting import COLORS, set_style

    set_style()
    sizes = [n for n in sizes if any(r["n"] == n for r in results)][-3:]
    figure, axes = plt.subplots(
        1, len(sizes), figsize=(5.5 * len(sizes), 5.5), squeeze=False
    )
    for axis, n in zip(axes[0], sizes):
        rows = [r for r in results if r["n"] == n and "error" in r]
        mpstab = sorted(
            (r for r in rows if r["method"].startswith("mpstab")),
            key=lambda r: int(r["method"].split("=")[1]),
        )
        axis.loglog(
            [r["time"] for r in mpstab],
            [r["error"] for r in mpstab],
            "o-",
            color=COLORS["blue"],
            label="mpstab",
        )
        for r in mpstab:
            axis.annotate(
                r"$\chi=" + r["method"].split("=")[1] + "$",
                (r["time"], r["error"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=10,
                color=COLORS["blue"],
            )
        palette = iter([COLORS["red"], COLORS["orange"]])
        for r in rows:
            if r["method"].startswith("PP"):
                axis.loglog(
                    r["time"],
                    r["error"],
                    "s",
                    color=next(palette),
                    label=r["method"].replace("PP ", "PauliPropagation\n"),
                )
        exact = [r for r in results if r["n"] == n and r["method"] == "PP exact"]
        if exact:
            axis.axvline(exact[0]["time"], color=COLORS["black"], ls="--", lw=1.5)
            axis.text(
                exact[0]["time"],
                axis.get_ylim()[0],
                "  PauliPropagation\n  exact (time)",
                fontsize=10,
                va="bottom",
            )
        axis.set_title(f"$n = {n}$")
        axis.set_xlabel("Time per gradient [s]")
    axes[0][0].set_ylabel(r"Gradient error $\|\Delta g\| / \|g\|$")
    axes[0][-1].legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.suptitle(f"Accuracy against cost, {args.nlayers} layers")
    path = args.plot.replace(".png", "_tradeoff.png")
    figure.savefig(path)
    print(f"plot saved to {path}")


if __name__ == "__main__":
    main()
