"""Gradient cost of mpstab against PennyLane and qiboml.

Every method differentiates the same thing: the energy of the transverse-field
Ising ring ``H = - sum_i Z_i Z_{i+1} - sum_i X_i`` on the
:class:`~mpstab.models.ansatze.HardwareEfficient` ansatz (per layer, an RY on
every qubit then a ring of CZ), at the same random parameters. The methods are

======================  =============  ===================================
method                  simulation     gradient
======================  =============  ===================================
mpstab adjoint          HSMPO, chi     :meth:`~mpstab.HSMPO.gradient`
mpstab adjoint chi      HSMPO, chi     the same, costate capped at chi
mpstab PSR              HSMPO, chi     parameter shift, light cone
PennyLane TN PSR        MPS, chi       ``default.tensor``, parameter shift
qiboml TN PSR           MPS, chi       qibotn quimb backend, ``PSR``
PennyLane SV adjoint    statevector    ``lightning.qubit``, adjoint
qiboml SV adjoint       statevector    ``Adjoint``
qiboml SV PSR           statevector    ``PSR``
======================  =============  ===================================

The MPS methods share the bond-dimension cap ``chi``. The statevector ones are
exact and give the reference the others are compared with; they stop at
``--max-statevector`` qubits (24 by default: a state on 30 qubits alone fills
16 GB).

A parameter-shift gradient is two energy evaluations per parameter, so the
PSR methods are timed on ``--psr-parameters`` evenly spaced parameters through
each framework's own energy evaluation, and the time is scaled to all of them;
their error is taken on those parameters. Any method is skipped at larger
sizes once its measured time exceeds ``--budget`` seconds.

qiboml's own tensor-network autodiff (``QuimbJax``) is not included: with
qiboml 0.1.1 and qibotn 0.0.7 (and both ``main`` branches as of November 2026)
qibotn cannot import qiboml's jax backend, so it does not run.

The script needs PennyLane, qiboml and qibotn next to mpstab, e.g. in a
Python 3.12 environment::

    uv pip install -e . pennylane jax \\
        "qiboml @ git+https://github.com/qiboteam/qiboml" \\
        "qibotn @ git+https://github.com/qiboteam/qibotn"
    python examples/scripts/gradient/gradient_frameworks.py
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
from qibo import Circuit, gates
from qibo.backends import construct_backend
from qibo.config import log
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Z

from mpstab import HSMPO
from mpstab.models.ansatze import HardwareEfficient


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


def tfim_symbolic(nqubits: int, backend):
    """The Hamiltonian as a qibo SymbolicHamiltonian, for qiboml."""
    form = sum(-Z(q) * Z((q + 1) % nqubits) - X(q) for q in range(nqubits))
    return SymbolicHamiltonian(form, nqubits=nqubits, backend=backend)


def copy_circuit(circuit: Circuit) -> Circuit:
    """A fresh qibo circuit with the same gates, so no method shares state."""
    copy = Circuit(circuit.nqubits)
    for gate in circuit.queue:
        copy.add(gate.__class__(*gate.init_args, **gate.init_kwargs))
    return copy


# Each builder takes (circuit, parameters, chi) and returns gradient(parameters).


def mpstab_adjoint(circuit, parameters, chi):
    hsmpo = HSMPO(copy_circuit(circuit), max_bond_dimension=chi, progress=False)
    terms = tfim_terms(circuit.nqubits)

    def gradient(parameters):
        hsmpo.set_parameters(parameters)
        return hsmpo.gradient(terms)

    return gradient


def mpstab_adjoint_chi(circuit, parameters, chi):
    hsmpo = HSMPO(copy_circuit(circuit), max_bond_dimension=chi, progress=False)
    terms = tfim_terms(circuit.nqubits)

    def gradient(parameters):
        hsmpo.set_parameters(parameters)
        return hsmpo.gradient(terms, costate_bond=chi)

    return gradient


def mpstab_psr(circuit, parameters, chi):
    hsmpo = HSMPO(copy_circuit(circuit), max_bond_dimension=chi, progress=False)
    terms = tfim_terms(circuit.nqubits)

    def energy(parameters):
        hsmpo.ansatz.circuit.set_parameters(parameters)
        return hsmpo.expectation_from_partition(terms, 0.0)[0]

    return energy


def _pennylane(circuit, device, diff_method):
    """The gradient with ``diff_method``, or the energy itself with ``None``."""
    import pennylane as qml

    n = circuit.nqubits
    hamiltonian = qml.Hamiltonian(
        [-1.0] * (2 * n),
        [qml.PauliZ(q) @ qml.PauliZ((q + 1) % n) for q in range(n)]
        + [qml.PauliX(q) for q in range(n)],
    )

    def energy(params):
        k = 0
        for gate in circuit.queue:
            if gate.name == "ry":
                qml.RY(params[k], wires=gate.target_qubits[0])
                k += 1
            elif gate.name == "cz":
                qml.CZ(wires=list(gate.qubits))
            else:
                raise ValueError(f"Unexpected gate {gate.name}")
        return qml.expval(hamiltonian)

    node = qml.QNode(energy, device, diff_method=diff_method)
    if diff_method is None:
        return lambda parameters: float(node(parameters))
    grad = qml.grad(node)
    return lambda parameters: np.array(
        grad(qml.numpy.array(parameters, requires_grad=True))
    )


def pennylane_tn_psr(circuit, parameters, chi):
    import pennylane as qml

    device = qml.device(
        "default.tensor", wires=circuit.nqubits, method="mps", max_bond_dim=chi
    )
    return _pennylane(circuit, device, None)


def pennylane_sv_adjoint(circuit, parameters, chi):
    import pennylane as qml

    return _pennylane(
        circuit, qml.device("lightning.qubit", wires=circuit.nqubits), "adjoint"
    )


def _qiboml(circuit, backend, differentiation=None):
    """The gradient with ``differentiation``, or the energy itself with ``None``."""
    from qiboml.models.decoding import Expectation

    copy = copy_circuit(circuit)
    decoding = Expectation(
        nqubits=circuit.nqubits,
        observable=tfim_symbolic(circuit.nqubits, backend),
        backend=backend,
    )
    if differentiation is None:

        def energy(parameters):
            copy.set_parameters(parameters)
            return float(np.real(backend.to_numpy(decoding(copy))).ravel()[0])

        return energy
    differentiation = differentiation(copy, decoding)
    return lambda parameters: np.array(
        differentiation.evaluate(backend.cast(parameters))
    ).ravel()


def qiboml_tn_psr(circuit, parameters, chi):
    backend = construct_backend("qibotn", platform="quimb", quimb_backend="numpy")
    backend.configure_tn_simulation(ansatz="mps", max_bond_dimension=chi)
    # qibotn 0.0.7 stores the optimizer as ``contractions_optimizer`` but
    # reads ``contraction_optimizer`` when evaluating an expectation value.
    backend.contraction_optimizer = backend.contractions_optimizer
    return _qiboml(circuit, backend)


def qiboml_sv_adjoint(circuit, parameters, chi):
    from qiboml.differentiations import Adjoint

    return _qiboml(circuit, construct_backend("numpy"), Adjoint)


def qiboml_sv_psr(circuit, parameters, chi):
    return _qiboml(circuit, construct_backend("numpy"))


METHODS = {
    "mpstab adjoint": mpstab_adjoint,
    "mpstab adjoint chi": mpstab_adjoint_chi,
    "mpstab PSR": mpstab_psr,
    "PennyLane TN PSR": pennylane_tn_psr,
    "qiboml TN PSR": qiboml_tn_psr,
    "PennyLane SV adjoint": pennylane_sv_adjoint,
    "qiboml SV adjoint": qiboml_sv_adjoint,
    "qiboml SV PSR": qiboml_sv_psr,
}

#: Methods whose builder returns the energy: their gradient is the parameter
#: shift rule, timed on a subset of the parameters.
SHIFTED = ("mpstab PSR", "PennyLane TN PSR", "qiboml TN PSR", "qiboml SV PSR")

#: Statevector methods: exact, and the reference of the others.
STATEVECTOR = ("PennyLane SV adjoint", "qiboml SV adjoint", "qiboml SV PSR")


def shift_rule(energy, parameters, indices):
    """Parameter-shift derivatives at ``indices``, from ``energy``."""
    derivatives = []
    for i in indices:
        values = []
        for sign in (1, -1):
            shifted = parameters.copy()
            shifted[i] += sign * np.pi / 2
            values.append(energy(shifted))
        derivatives.append((values[0] - values[1]) / 2)
    return np.array(derivatives)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--nqubits", type=int, nargs="+", default=[8, 12, 16, 20, 24, 32, 40, 60]
    )
    parser.add_argument("--max-statevector", type=int, default=24)
    parser.add_argument("--psr-parameters", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="gradient_frameworks.json")
    parser.add_argument("--plot", default="gradient_frameworks.png")
    args = parser.parse_args()
    log.setLevel("WARNING")

    # One tiny gradient per method first, so imports and first-call setup
    # stay out of the timings.
    np.random.seed(args.seed)
    tiny = HardwareEfficient(nqubits=4, nlayers=1).circuit
    tiny_parameters = np.array([p[0] for p in tiny.get_parameters()])
    for name in args.methods:
        METHODS[name](tiny, tiny_parameters, args.max_bond)(tiny_parameters)
    print(
        f"TFIM ring, HardwareEfficient with {args.nlayers} layers, MPS bond cap "
        f"{args.max_bond}; PSR timed on {args.psr_parameters} parameters and "
        "scaled (*)\n"
    )

    results = {name: {} for name in args.methods}
    skipped = set()
    for n in args.nqubits:
        np.random.seed(args.seed)
        circuit = HardwareEfficient(nqubits=n, nlayers=args.nlayers).circuit
        parameters = np.array([p[0] for p in circuit.get_parameters()])
        subset = np.linspace(0, len(parameters) - 1, args.psr_parameters)
        subset = np.unique(subset.astype(int))
        gradients = {}
        for name in args.methods:
            if name in skipped or (name in STATEVECTOR and n > args.max_statevector):
                continue
            method = METHODS[name](circuit, parameters, args.max_bond)
            start = time.perf_counter()
            if name in SHIFTED:
                gradients[name] = shift_rule(method, parameters.copy(), subset)
                measured = time.perf_counter() - start
                elapsed = measured * len(parameters) / len(subset)
            else:
                gradients[name] = method(parameters.copy())
                measured = elapsed = time.perf_counter() - start
            results[name][n] = {"time": elapsed, "estimated": name in SHIFTED}
            if measured > args.budget:
                skipped.add(name)
        reference = next((gradients[m] for m in STATEVECTOR if m in gradients), None)
        print(f"n = {n}, {len(parameters)} parameters")
        for name, gradient in gradients.items():
            entry = results[name][n]
            if reference is not None:
                expected = reference
                if name in SHIFTED and len(reference) == len(parameters):
                    expected = reference[subset]
                elif name not in SHIFTED and len(reference) != len(parameters):
                    gradient = gradient[subset]
                entry["error"] = float(np.abs(gradient - expected).max())
            error = f"   max |error| {entry['error']:.1e}" if "error" in entry else ""
            star = "*" if entry["estimated"] else " "
            print(f"  {name:>21}: {entry['time']:9.3f} s{star}{error}")

    with open(args.output, "w") as file:
        json.dump(
            {
                "nlayers": args.nlayers,
                "max_bond": args.max_bond,
                "results": {
                    name: {str(n): entry for n, entry in sizes.items()}
                    for name, sizes in results.items()
                },
            },
            file,
            indent=2,
        )
    print(f"\nresults saved to {args.output}")
    plot(results, args)


#: Line style of each method: tensor networks solid, statevectors dashed.
STYLES = {
    "mpstab adjoint": dict(color="blue", marker="o"),
    "mpstab adjoint chi": dict(color="blue", marker="o", ls=":"),
    "mpstab PSR": dict(color="blue", marker="o", mfc="white"),
    "PennyLane TN PSR": dict(color="red", marker="s"),
    "qiboml TN PSR": dict(color="orange", marker="D"),
    "PennyLane SV adjoint": dict(color="red", marker="s", ls="--"),
    "qiboml SV adjoint": dict(color="orange", marker="D", ls="--"),
    "qiboml SV PSR": dict(color="black", marker="^", ls="--"),
}


def plot(results, args):
    """Time per gradient against the number of qubits, one line per method."""
    import matplotlib.pyplot as plt

    # examples/, home of the plot style

    sys.path.append(str(pathlib.Path(__file__).parents[2]))

    from plotting import COLORS, set_style

    set_style()
    figure, axis = plt.subplots(figsize=(12, 6))
    for name, sizes in results.items():
        if not sizes:
            continue
        style = dict(STYLES[name])
        style["color"] = COLORS[style["color"]]
        axis.semilogy(
            [int(n) for n in sizes],
            [entry["time"] for entry in sizes.values()],
            label=name,
            **style,
        )
    axis.set_xlabel("Qubits $n$")
    axis.set_ylabel("Time per gradient [s] (PSR scaled from a subset)")
    axis.set_title(
        f"Ising ring, HardwareEfficient with {args.nlayers} layers, "
        f"$\\chi = {args.max_bond}$"
    )
    axis.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.savefig(args.plot)
    print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
