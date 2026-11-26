"""VQE on the HSMPO surrogate, trained with its analytic gradient.

The :class:`~mpstab.models.ansatze.HardwareEfficient` ansatz (per layer, an RY
on every qubit then a ring of CZ) is trained with Adam to the ground state of
the transverse-field Ising ring

    H = - sum_i Z_i Z_{i+1} - h sum_i X_i.

The same training runs twice, from the same initial parameters and with the
same optimizer, with two gradients:

- analytic: :meth:`~mpstab.HSMPO.gradient`, the adjoint gradient, about twice
  the cost of an expectation value whatever the number of parameters;
- parameter shift: two expectation values per parameter, each on a shifted
  circuit restricted to the observable's light cone.

Each step also evaluates the energy, with :meth:`~mpstab.HSMPO.set_parameters`
and :meth:`~mpstab.HSMPO.expectation`, and that time is counted in both. The
model maps to free fermions, so the exact ground energy is known at any size.
The plot shows the energy error against the step and against the wall time.

Run with ``uv run python examples/scripts/gradient/vqe_training.py``, or e.g.
``--nqubits 40 --nlayers 4 --max-bond 32 --psr-steps 10`` for a size beyond
exact methods, where the shift rule is only run for its first steps.
"""

import argparse
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from qibo import set_backend

from mpstab import HSMPO
from mpstab.models.ansatze import HardwareEfficient

# examples/, home of the plot style
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from plotting import COLORS, set_style


def tfim_ring(nqubits: int, field: float) -> dict:
    """``- sum_i Z_i Z_{i+1} - field sum_i X_i`` with periodic boundaries."""
    terms = {}
    for q in range(nqubits):
        labels = ["I"] * nqubits
        labels[q] = labels[(q + 1) % nqubits] = "Z"
        terms["".join(labels)] = -1.0
        labels = ["I"] * nqubits
        labels[q] = "X"
        terms["".join(labels)] = -field
    return terms


def ground_energy(nqubits: int, field: float) -> float:
    """
    Exact ground energy of :func:`tfim_ring`, which maps to free fermions:
    ``-sum_m sqrt(1 + h^2 - 2 h cos k_m)`` with ``k_m = (2m + 1) pi / n``.
    """
    k = np.pi * (2 * np.arange(nqubits) + 1) / nqubits
    return float(-np.sum(np.sqrt(1 + field**2 - 2 * field * np.cos(k))))


def shift_rule(surrogate: HSMPO, hamiltonian: dict, parameters: np.ndarray):
    """The parameter-shift gradient, for comparison."""
    gradient = np.zeros(len(parameters))
    for i in range(len(parameters)):
        values = []
        for sign in (1, -1):
            shifted = parameters.copy()
            shifted[i] += sign * np.pi / 2
            surrogate.ansatz.circuit.set_parameters(shifted)
            values.append(
                surrogate.expectation_from_partition(
                    hamiltonian, replacement_probability=0.0
                )[0]
            )
        gradient[i] = (values[0] - values[1]) / 2
    surrogate.ansatz.circuit.set_parameters(parameters)
    return gradient


class Adam:
    """The Adam update rule, on a flat parameter vector."""

    def __init__(self, learning_rate: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate, self.beta1, self.beta2, self.eps = (
            learning_rate,
            beta1,
            beta2,
            eps,
        )
        self.m = self.v = None
        self.t = 0

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m, self.v = np.zeros_like(gradient), np.zeros_like(gradient)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m = self.m / (1 - self.beta1**self.t)
        v = self.v / (1 - self.beta2**self.t)
        return parameters - self.learning_rate * m / (np.sqrt(v) + self.eps)


def train(surrogate, hamiltonian, parameters, gradient, steps, learning_rate):
    """
    Adam from ``parameters`` with ``gradient(parameters)``; returns the energy
    and the elapsed wall time after each step, the initial point included.
    """
    optimizer = Adam(learning_rate)
    energies, times = [], []
    start = time.perf_counter()
    for step in range(steps + 1):
        surrogate.set_parameters(parameters)
        energies.append(surrogate.expectation(hamiltonian))
        times.append(time.perf_counter() - start)
        if step < steps:
            parameters = optimizer.step(parameters, gradient(parameters))
    return np.array(energies), np.array(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nqubits", type=int, default=10)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--field", type=float, default=1.0)
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--psr-steps",
        type=int,
        default=None,
        help="steps of the parameter-shift training, --steps by default",
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="vqe_training.png")
    args = parser.parse_args()

    set_backend("numpy")
    np.random.seed(args.seed)
    ansatz = HardwareEfficient(nqubits=args.nqubits, nlayers=args.nlayers)
    hamiltonian = tfim_ring(args.nqubits, args.field)
    surrogate = HSMPO(ansatz, max_bond_dimension=args.max_bond, progress=False)
    parameters = np.array([p[0] for p in surrogate.get_parameters()])

    exact = ground_energy(args.nqubits, args.field)
    print(
        f"TFIM ring, {args.nqubits} qubits, h = {args.field}; HardwareEfficient, "
        f"{args.nlayers} layers, {len(parameters)} parameters, "
        f"max bond {args.max_bond}; exact ground energy {exact:.6f}"
    )

    start = time.perf_counter()
    analytic = surrogate.gradient(hamiltonian)
    analytic_time = time.perf_counter() - start
    start = time.perf_counter()
    reference = shift_rule(surrogate, hamiltonian, parameters)
    psr_time = time.perf_counter() - start
    print(
        f"gradient at the initial point: analytic {analytic_time:.3f}s, "
        f"parameter shift {psr_time:.3f}s ({psr_time / analytic_time:.0f}x), "
        f"max |difference| {np.abs(analytic - reference).max():.1e}\n"
    )

    gradients = {
        "analytic": lambda _: surrogate.gradient(hamiltonian),
        "parameter shift": lambda p: shift_rule(surrogate, hamiltonian, p),
    }
    psr_steps = args.steps if args.psr_steps is None else args.psr_steps
    runs = {}
    for (name, gradient), steps in zip(gradients.items(), (args.steps, psr_steps)):
        energies, times = train(
            surrogate, hamiltonian, parameters, gradient, steps, args.learning_rate
        )
        runs[name] = energies, times
        print(
            f"{name:>15}: {steps} steps in {times[-1]:.1f}s "
            f"({times[-1] / max(steps, 1) * 1e3:.0f} ms per step), "
            f"final energy {energies[-1]:+.6f}, error {energies[-1] - exact:.2e}"
        )

    set_style()
    styles = {
        "analytic": dict(color=COLORS["blue"]),
        "parameter shift": dict(color=COLORS["red"], ls="--"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (energies, times) in runs.items():
        axes[0].semilogy(energies - exact, label=name, **styles[name])
        axes[1].semilogy(times, energies - exact, label=name, **styles[name])
    for axis, xlabel in zip(axes, ("Step", "Wall time [s]")):
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Energy error $E - E_0$")
        axis.legend()
    figure.suptitle(
        f"VQE on the Ising ring, {args.nqubits} qubits, HardwareEfficient with "
        f"{args.nlayers} layers"
    )
    figure.savefig(args.plot)
    print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
