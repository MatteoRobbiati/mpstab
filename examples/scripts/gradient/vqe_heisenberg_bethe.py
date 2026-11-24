"""A VQE on the Heisenberg ring, checked against the Bethe ansatz.

The periodic XXX chain ``H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})``
is integrable: its ground energy follows from the Bethe equations for ``n/2``
magnons, which are solved here for any even ``n`` (and checked against exact
diagonalization on small rings).

The ansatz is the Hamiltonian variational one: a product of singlets on the
pairs ``(0, 1), (2, 3), ...``, prepared by Clifford gates that mpstab absorbs
into its tableau, then layers of RZZ, RXX and RYY rotations on the odd bonds
and on the even bonds, each with its own parameter. It is trained with
:meth:`~mpstab.HSMPO.optimize_parameters` (L-BFGS on the analytic gradient).

At the end the trained parameters are evaluated again at a larger bond
dimension, to check that the energy is the circuit's and not a truncation
artefact, and the script estimates what the same training would have cost
with the parameter-shift rule (two energy evaluations per parameter and step).

Run with ``uv run python examples/scripts/gradient/vqe_heisenberg_bethe.py``,
e.g. ``--nqubits 40 --nlayers 6 --max-bond 64``.
"""

import argparse
import pathlib
import sys
import time
from functools import reduce

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg
from qibo import Circuit, gates, set_backend
from qibo.config import log
from scipy.optimize import fsolve

from mpstab import HSMPO

# examples/, home of the plot style
sys.path.append(str(pathlib.Path(__file__).parents[2]))
from plotting import COLORS, set_style


def heisenberg(nqubits: int) -> dict:
    """``sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})`` on a ring."""
    terms = {}
    for q in range(nqubits):
        for label in "XYZ":
            labels = ["I"] * nqubits
            labels[q] = labels[(q + 1) % nqubits] = label
            terms["".join(labels)] = 1.0
    return terms


def bethe_energy(nqubits: int) -> float:
    """
    Ground energy of :func:`heisenberg` from the Bethe ansatz.

    With ``m = n/2`` rapidities ``lambda_j``, the ground state solves
    ``n arctan(2 lambda_j) = pi I_j + sum_k arctan(lambda_j - lambda_k)`` with
    ``I_j = -(m-1)/2, ..., (m-1)/2``, and ``E = 4 (n/4 - sum_j 1/(2 lambda_j^2 + 1/2))``
    in units of Pauli matrices.
    """
    m = nqubits // 2
    quantum = np.arange(m) - (m - 1) / 2

    def equations(rapidities):
        return (
            nqubits * np.arctan(2 * rapidities)
            - np.pi * quantum
            - np.arctan(rapidities[:, None] - rapidities[None, :]).sum(axis=1)
        )

    rapidities = fsolve(equations, 0.5 * np.tan(np.pi * quantum / nqubits))
    if np.abs(equations(rapidities)).max() > 1e-9:
        raise RuntimeError("The Bethe equations did not converge.")
    return float(4 * (nqubits / 4 - np.sum(0.5 / (rapidities**2 + 0.25))))


def exact_energy(nqubits: int) -> float:
    """Ground energy of :func:`heisenberg` by sparse diagonalization."""
    paulis = {
        "I": sparse.identity(2, format="csr"),
        "X": sparse.csr_matrix([[0, 1], [1, 0]]),
        "Y": sparse.csr_matrix([[0, -1j], [1j, 0]]),
        "Z": sparse.csr_matrix([[1, 0], [0, -1]]),
    }
    operator = sum(
        c * reduce(sparse.kron, [paulis[label] for label in pauli])
        for pauli, c in heisenberg(nqubits).items()
    )
    return float(sparse_linalg.eigsh(operator.tocsr(), k=1, which="SA")[0][0].real)


def hamiltonian_variational(nqubits: int, nlayers: int, seed: int = 0) -> Circuit:
    """Singlets on ``(0, 1), (2, 3), ...``, then layers of RZZ, RXX, RYY."""
    rng = np.random.default_rng(seed)
    circuit = Circuit(nqubits)
    for i in range(0, nqubits, 2):
        circuit.add(gates.H(i))
        circuit.add(gates.CNOT(i, i + 1))
        circuit.add(gates.X(i + 1))
        circuit.add(gates.Z(i))
    for _ in range(nlayers):
        for start in (1, 0):  # odd bonds, then even bonds
            for i in range(start, nqubits, 2):
                pair = (i, (i + 1) % nqubits)
                for gate in (gates.RZZ, gates.RXX, gates.RYY):
                    # Small random angles: away from the Clifford points.
                    circuit.add(gate(*pair, theta=rng.normal(scale=0.05)))
    return circuit


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nqubits", type=int, default=40)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--max-bond", type=int, default=64)
    parser.add_argument("--check-bond", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="vqe_heisenberg_bethe.png")
    args = parser.parse_args()
    set_backend("numpy")
    log.setLevel("WARNING")

    for n in (8, 12):
        assert abs(bethe_energy(n) - exact_energy(n)) < 1e-9
    n = args.nqubits
    reference = bethe_energy(n)
    hamiltonian = heisenberg(n)
    hsmpo = HSMPO(
        hamiltonian_variational(n, args.nlayers, args.seed),
        max_bond_dimension=args.max_bond,
        progress=False,
    )
    print(
        f"Heisenberg ring, {n} qubits; Bethe ansatz ground energy {reference:.8f} "
        f"({reference / n:.6f} per site)\nHamiltonian variational ansatz, "
        f"{args.nlayers} layers, {hsmpo.nparams} parameters, chi = {args.max_bond}\n"
    )

    times = []
    start = time.perf_counter()

    def report(step, energy, parameters):
        times.append(time.perf_counter() - start)
        if step % 25 == 0:
            print(
                f"eval {step:4d}  E = {energy:.6f}  relative error "
                f"{(energy - reference) / abs(reference):.2e}  ({times[-1]:.0f} s)",
                flush=True,
            )

    result = hsmpo.optimize_parameters(
        hamiltonian, method="lbfgs", max_steps=args.max_steps, callback=report
    )
    energies = np.array(result["energy_history"])
    print(
        f"\ntrained: E = {result['energy']:.8f}, relative error "
        f"{(result['energy'] - reference) / abs(reference):.2e}, "
        f"{result['num_steps']} evaluations in {times[-1]:.0f} s "
        f"({times[-1] / result['num_steps']:.2f} s each)"
    )

    check = HSMPO(hsmpo.ansatz, max_bond_dimension=args.check_bond, progress=False)
    checked, fidelity = check.expectation(hamiltonian, return_fidelity=True)
    print(
        f"same parameters at chi = {args.check_bond}: E = {checked:.8f} "
        f"(truncation fidelity {float(np.real(fidelity)):.6f}), relative error "
        f"{(checked - reference) / abs(reference):.2e}"
    )

    start = time.perf_counter()
    hsmpo.expectation_from_partition(hamiltonian, 0.0)
    one = time.perf_counter() - start
    estimate = 2 * hsmpo.nparams * one * result["num_steps"]
    print(
        f"parameter-shift rule, same evaluations: ~{estimate / 3600:.1f} h "
        f"({2 * hsmpo.nparams} energies of {one:.2f} s per gradient)"
    )

    set_style()
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    left.plot(energies, color=COLORS["blue"], label="VQE, analytic gradient")
    left.axhline(reference, color=COLORS["black"], ls="--", label="Bethe ansatz")
    left.set_xlabel("Evaluation")
    left.set_ylabel("Energy")
    left.legend()
    right.semilogy(times, (energies - reference) / abs(reference), color=COLORS["blue"])
    right.axhline(
        (checked - reference) / abs(reference),
        color=COLORS["red"],
        ls=":",
        label=f"final parameters at $\\chi = {args.check_bond}$",
    )
    right.set_xlabel("Wall time [s]")
    right.set_ylabel(r"Relative error $(E - E_{\rm Bethe})/|E_{\rm Bethe}|$")
    right.legend()
    figure.suptitle(
        f"Heisenberg ring, {n} qubits: HVA with {args.nlayers} layers, "
        f"$\\chi = {args.max_bond}$"
    )
    figure.savefig(args.plot)
    print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
