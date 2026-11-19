"""Analytic HSMPO gradient against the parameter-shift rule.

For a hardware-efficient ansatz (RY layers and a CZ ring) and the nearest-neighbour
``sum_i Z_i Z_{i+1}``, the script times the full gradient three ways:

- adjoint: :meth:`~mpstab.HSMPO.gradient`, one forward pass with checkpoints
  and one backward pass of the costate ``O |psi>``;
- flips: ``method="flips"``, one base chain with checkpoints and one shifted
  tail per distinct sign pattern;
- parameter shift: two light-cone-restricted HSMPO evaluations per parameter,
  each rebuilding its chain from the initial state.

It reports the rotations each one applies to the MPS, the wall time, the
speedup over the shift rule and the largest difference from it. The shift rule can be timed on a
subset of the parameters with ``--psr-parameters``, in which case its totals are
scaled up to all of them, and the difference is taken over that subset.

Run with ``uv run python examples/scripts/gradient/gradient_vs_psr.py``.
"""

import argparse
import time

import numpy as np
from qibo import set_backend

from mpstab import HSMPO
from mpstab.evolutors.gradient import analytic_gradient
from mpstab.models.ansatze import HardwareEfficient
from mpstab.pauli import light_cone


def ring_zz(nqubits: int) -> dict:
    """``sum_i Z_i Z_{i+1}`` with periodic boundaries."""
    terms = {}
    for q in range(nqubits):
        labels = ["I"] * nqubits
        labels[q] = labels[(q + 1) % nqubits] = "Z"
        terms["".join(labels)] = 1.0
    return terms


def shift_rule(hsmpo: HSMPO, observable: dict, parameters):
    """
    The parameter-shift gradient, with the rotations the shifted circuits apply.
    """
    circuit = hsmpo.ansatz.circuit
    values = np.array([p[0] for p in circuit.get_parameters()])
    gradient, n_rotations = [], 0
    for i in parameters:
        evaluations = []
        for sign in (1, -1):
            shifted = values.copy()
            shifted[i] += sign * np.pi / 2
            circuit.set_parameters(shifted)
            expval, partitions = hsmpo.expectation_from_partition(
                observable, replacement_probability=0.0, return_partitions=True
            )
            evaluations.append(expval)
            n_rotations += _light_cone_size(hsmpo, observable, partitions)
        gradient.append((evaluations[0] - evaluations[1]) / 2)
    circuit.set_parameters(values)
    return np.array(gradient), n_rotations


def _light_cone_size(hsmpo, observable, partitions) -> int:
    """Rotations ``expectation_from_partition`` applied, recomputed after the fact."""
    clifford = partitions["only_cliffords"]
    rotations = hsmpo._dress(partitions["magic_gates"], clifford)
    dressed = hsmpo.stab_engine.backpropagate_many(observable, clifford)
    return len(
        light_cone(
            [generator for generator, _ in rotations],
            [pauli for pauli, _ in dressed],
            hsmpo.nqubits,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nqubits", type=int, nargs="+", default=[20, 40, 60])
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument(
        "--psr-parameters",
        type=int,
        default=None,
        help="time the shift rule on this many evenly spaced parameters only",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_backend("numpy")
    print(
        f"HardwareEfficient, {args.nlayers} layers, sum Z_i Z_(i+1), "
        f"max bond {args.max_bond}\n"
    )
    header = (
        f"{'n':>3} | {'params':>6} | {'method':>7} | {'rotations':>9} | "
        f"{'time':>8} | {'speedup':>7} | {'max |diff|':>10} | {'min F':>7}"
    )
    print(header)
    print("-" * len(header))
    for nqubits in args.nqubits:
        np.random.seed(args.seed)
        ansatz = HardwareEfficient(nqubits=nqubits, nlayers=args.nlayers)
        observable = ring_zz(nqubits)
        hsmpo = HSMPO(ansatz, max_bond_dimension=args.max_bond, progress=False)
        nparams = hsmpo.nparams

        results = {}
        for method in ("adjoint", "flips"):
            start = time.perf_counter()
            result = analytic_gradient(hsmpo, observable, method=method)
            results[method] = (result, time.perf_counter() - start)

        subset = np.arange(nparams)
        if args.psr_parameters is not None and args.psr_parameters < nparams:
            subset = np.linspace(0, nparams - 1, args.psr_parameters).astype(int)
        start = time.perf_counter()
        psr, psr_rotations = shift_rule(hsmpo, observable, subset)
        scale = nparams / len(subset)
        psr_time = (time.perf_counter() - start) * scale

        def row(method, rotations, seconds, difference, fidelity):
            print(
                f"{nqubits:>3} | {nparams:>6} | {method:>7} | {rotations:>9} | "
                f"{seconds:>7.2f}s | {psr_time / seconds:>6.1f}x | "
                f"{difference:>10} | {fidelity:>7}"
            )

        for method, (result, seconds) in results.items():
            difference = np.abs(result.values[subset] - psr).max()
            row(
                method,
                result.n_rotations,
                seconds,
                f"{difference:.2e}",
                f"{result.fidelities.min():.4f}",
            )
        row("PSR", round(psr_rotations * scale), psr_time, "-", "-")


if __name__ == "__main__":
    main()
