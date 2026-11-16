"""Helpers for ``frameworks.ipynb``: one circuit, three simulators.

Every runner takes the same qibo circuit, the same list of
:mod:`mpstab.noise` events and the same observable, and returns a dict with at
least ``value`` and ``time`` (seconds, construction included):

- :func:`run_mpstab` / :func:`run_mpstab_noisy` -- the HSMPO surrogate;
- :func:`run_qibojit` -- qibo's numba backend, statevector without noise and
  density matrix with it: the exact reference;
- :class:`PauliPropagation` -- PauliPropagation.jl through juliacall, a
  truncated Heisenberg-picture Pauli-sum simulator.

Observables are a Pauli string (qubit 0 leftmost) or a ``{pauli: coefficient}``
dict. Noise is limited to what all three can express: a uniform depolarizing
channel per qubit and Pauli rotations (e.g. ZZ crosstalk).
"""

from __future__ import annotations

import time

import numpy as np
from qibo import Circuit, gates, symbols
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab import HSMPO
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.noise import (
    CoherentRotation,
    PauliChannel,
    compile_noise,
    noisy_expectation,
)

# -- Circuits and noise ---------------------------------------------------------


def kicked_ising(n: int, steps: int, theta_h=0.7, theta_z=0.3, theta_j=-np.pi / 2):
    """Floquet kicked Ising: RX/RZ magic, CNOT-RZ-CNOT couplings (Clifford at -pi/2)."""
    return KickedIsing(
        nqubits=n, n_steps=steps, theta_h=theta_h, theta_z=theta_z, theta_j=theta_j
    ).circuit


def hardware_efficient(n: int, layers: int, seed: int = 0):
    """RY layers with random angles (drawn with ``seed``) and CZ ladders."""
    # HardwareEfficient draws its angles from NumPy's global generator.
    np.random.seed(seed)
    return HardwareEfficient(nqubits=n, nlayers=layers).circuit


def noise_events(circuit: Circuit, p: float = 0.0, crosstalk: float = 0.0):
    """
    Depolarizing noise of total probability ``p`` on every qubit a gate touches,
    after the gate, plus a ``ZZ`` rotation of angle ``crosstalk`` after every
    two-qubit gate.
    """
    n = circuit.nqubits
    events = []
    for i, gate in enumerate(circuit.queue):
        if p > 0:
            for q in gate.qubits:
                events.append(PauliChannel(i, (q,), {l: p / 3 for l in "XYZ"}))
        if crosstalk and len(gate.qubits) == 2:
            generator = "".join("Z" if q in gate.qubits else "I" for q in range(n))
            events.append(CoherentRotation(i, generator, crosstalk))
    return events


def local_z(n: int, sites=None) -> str:
    """``Z`` on ``sites`` (default: the two end qubits), identity elsewhere."""
    sites = (0, n - 1) if sites is None else sites
    return "".join("Z" if q in sites else "I" for q in range(n))


def zz_chain(n: int) -> dict:
    """``sum_i Z_i Z_{i+1}`` with unit coefficients."""
    return {local_z(n, (i, i + 1)): 1.0 for i in range(n - 1)}


def _terms(observable) -> dict:
    return {observable: 1.0} if isinstance(observable, str) else dict(observable)


def _by_gate(events) -> dict:
    grouped = {}
    for event in events:
        grouped.setdefault(event.after_gate, []).append(event)
    return grouped


# -- mpstab ---------------------------------------------------------------------


def run_mpstab(circuit, observable, max_bond=None, light_cone=True):
    """
    Noiseless HSMPO expectation value.

    With ``light_cone`` only the dressed rotations inside the observable's light
    cone are applied (``expectation_from_partition`` with nothing replaced);
    without it the whole circuit is evolved, as ``HSMPO.expectation`` does.
    """
    t = time.perf_counter()
    if light_cone:
        # rotations_only skips the full evolution HSMPO() runs at construction.
        hsmpo = HSMPO.rotations_only(circuit)
        hsmpo.max_bond_dimension = max_bond
        value, _ = hsmpo.expectation_from_partition(
            observable, replacement_probability=0.0
        )
        state = hsmpo.mps
    else:
        hsmpo = HSMPO(circuit, max_bond_dimension=max_bond)
        value = hsmpo.expectation(observable)
        state = hsmpo.original_circuit_mps
    elapsed = time.perf_counter() - t
    return dict(
        value=float(np.real(value)),
        time=elapsed,
        bond=int(state.max_bond()),
        fidelity=float(np.real(state.norm(squared=True))),
        magic=len(hsmpo.magic_gates),
    )


def run_mpstab_noisy(
    circuit, events, observable, n_samples=1000, max_bond=None, seed=0, light_cone=True
):
    """Noisy HSMPO expectation value: exact for 0 and 1 faults, sampled beyond."""
    t = time.perf_counter()
    hsmpo = HSMPO.rotations_only(circuit)
    hsmpo.max_bond_dimension = max_bond
    result = noisy_expectation(
        compile_noise(hsmpo, events),
        observable,
        n_samples=n_samples,
        seed=seed,
        light_cone=light_cone,
    )
    elapsed = time.perf_counter() - t
    return dict(
        value=result.mean,
        time=elapsed,
        stderr=result.stderr,
        patterns=result.n_patterns,
        p_two_or_more=result.stratum_weights[2],
        min_fidelity=float(np.nanmin(result.fidelities)),
    )


# -- qibo (qibojit) -------------------------------------------------------------


def qibojit_backend(threads: int | None = None):
    """qibo's numba backend, as an object: the global qibo backend is untouched."""
    from qibo.backends import construct_backend

    backend = construct_backend("qibojit", platform="numba")
    if threads is not None:
        backend.set_threads(threads)
    # numba compiles its kernels on first use: pay for that here, for both the
    # statevector and the density-matrix paths, rather than in the first timing.
    warmup = kicked_ising(4, 1)
    for events in ([], noise_events(warmup, 0.01, 0.01)):
        run_qibojit(backend, warmup, events, local_z(4))
    return backend


def run_qibojit(backend, circuit, events, observable):
    """
    Exact expectation value: statevector without events, density matrix with
    them, both executed by ``backend``.

    The observable is a qibo ``SymbolicHamiltonian`` on the same backend, whose
    ``expectation(circuit)`` runs the circuit and contracts each Pauli term
    with the qubits it acts on, never building a dense ``2^n x 2^n`` matrix.
    Building the Hamiltonian is setup and is not timed.
    """
    hamiltonian = qibo_hamiltonian(observable, circuit.nqubits, backend)
    target = Circuit(circuit.nqubits, density_matrix=bool(events))
    grouped = _by_gate(events)
    for i, gate in enumerate(circuit.queue):
        target.add(gate)
        for event in grouped.get(i, ()):
            target.add(_qibo_gate(event))
    t = time.perf_counter()
    value = hamiltonian.expectation(target)
    return dict(value=float(np.real(value)), time=time.perf_counter() - t)


def qibo_hamiltonian(observable, nqubits: int, backend) -> SymbolicHamiltonian:
    """The observable as a qibo ``SymbolicHamiltonian`` on ``backend``."""
    form = 0
    for pauli, coefficient in _terms(observable).items():
        term = coefficient
        for q, label in enumerate(pauli):
            if label != "I":
                term *= getattr(symbols, label)(q)
        form += term
    return SymbolicHamiltonian(form=form, nqubits=nqubits, backend=backend)


def _qibo_gate(event):
    if isinstance(event, PauliChannel):
        return gates.PauliNoiseChannel(event.qubits, list(event.probs.items()))
    qubits = [q for q, l in enumerate(event.generator) if l != "I"]
    labels = "".join(event.generator[q] for q in qubits)
    rotations = {"X": gates.RX, "Y": gates.RY, "Z": gates.RZ}
    pairs = {"XX": gates.RXX, "YY": gates.RYY, "ZZ": gates.RZZ}
    if len(qubits) == 1:
        return rotations[labels](qubits[0], theta=event.angle)
    if labels in pairs:
        return pairs[labels](*qubits, theta=event.angle)
    raise NotImplementedError(f"No qibo gate for the rotation {event.generator}.")


# -- PauliPropagation.jl --------------------------------------------------------

#: PauliPropagation.jl's package UUID, for juliapkg.
PAULI_PROPAGATION_UUID = "293282d5-3c99-4fb6-92d0-fd3280a19750"

_CLIFFORDS = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "s": "S",
    "cx": "CNOT",
    "cnot": "CNOT",
    "cz": "CZ",
    "swap": "SWAP",
}


class PauliPropagation:
    """
    PauliPropagation.jl through juliacall.

    The first instance installs the Julia package if needed (a few minutes the
    very first time) and compiles it on a tiny circuit, so later timings do
    not include Julia's JIT.
    """

    def __init__(self):
        import juliapkg

        juliapkg.add("PauliPropagation", PAULI_PROPAGATION_UUID)
        juliapkg.resolve()
        from juliacall import Main as jl

        self.jl = jl
        jl.seval("using PauliPropagation")
        warmup = kicked_ising(4, 1)
        self.run(warmup, noise_events(warmup, 0.01, 0.01), local_z(4))

    def run(
        self,
        circuit,
        events,
        observable,
        min_abs_coeff=1e-8,
        max_weight=None,
        repeats=2,
    ):
        """
        Propagate the observable back through the circuit, dropping Pauli
        terms with ``|coefficient| < min_abs_coeff`` (and, optionally, weight
        above ``max_weight``), then overlap with ``|0...0>``. The reported
        time is the best of ``repeats`` identical runs, so it excludes Julia's
        compilation.
        """
        n = circuit.nqubits
        self.jl.seval(f"circ = [{', '.join(self._circuit(circuit, events))}]")
        self.jl.seval(f"obs = PauliSum({n})")
        for pauli, c in _terms(observable).items():
            sites = [q + 1 for q, l in enumerate(pauli) if l != "I"]
            symbols = ", ".join(f":{pauli[s - 1]}" for s in sites)
            self.jl.seval(f"add!(obs, [{symbols}], {sites}, {float(c)!r})")
        options = f"min_abs_coeff={min_abs_coeff!r}"
        if max_weight is not None:
            options += f", max_weight={int(max_weight)}"
        # Julia compiles propagate anew for every combination of Pauli integer
        # width (set by n) and gate types, and the first call pays for it.
        # Propagation is deterministic, so time the best of `repeats` runs.
        # Building the circuit above is translation, as for qibo: not timed.
        elapsed = np.inf
        for _ in range(repeats):
            t = time.perf_counter()
            self.jl.seval(f"psum = propagate(circ, obs; {options})")
            elapsed = min(elapsed, time.perf_counter() - t)
        value = float(self.jl.seval("overlapwithzero(psum)"))
        return dict(value=value, time=elapsed, terms=int(self.jl.seval("length(psum)")))

    @staticmethod
    def _circuit(circuit, events) -> list:
        grouped = _by_gate(events)
        lines = []
        for i, gate in enumerate(circuit.queue):
            q = [x + 1 for x in gate.qubits]
            if gate.name in ("rx", "ry", "rz"):
                lines.append(
                    f"PauliRotation(:{gate.name[1].upper()}, {q[0]}, {gate.parameters[0]!r})"
                )
            elif gate.name in _CLIFFORDS:
                lines.append(f"CliffordGate(:{_CLIFFORDS[gate.name]}, {q})")
            else:
                raise NotImplementedError(f"No PauliPropagation gate for {gate.name}.")
            for event in grouped.get(i, ()):
                lines.append(PauliPropagation._event(event))
        return lines

    @staticmethod
    def _event(event) -> str:
        if isinstance(event, PauliChannel):
            probs = [event.probs.get(l, 0.0) for l in "XYZ"]
            if len(event.qubits) != 1 or not np.allclose(probs, probs[0]):
                raise NotImplementedError(
                    "Only single-qubit depolarizing channels translate to "
                    "PauliPropagation.jl here."
                )
            # p/3 per Pauli damps every non-identity Pauli by 1 - 4p/3.
            return f"DepolarizingNoise({event.qubits[0] + 1}, {4 * probs[0]!r})"
        sites = [q + 1 for q, l in enumerate(event.generator) if l != "I"]
        symbols = ", ".join(f":{event.generator[s - 1]}" for s in sites)
        return f"PauliRotation([{symbols}], {sites}, {event.angle!r})"


# -- Sweeps and plots -----------------------------------------------------------

#: Fixed colour per framework, so every plot reads the same way.
COLORS = {"mpstab": "#2a78d6", "qibojit": "#eb6834", "PauliPropagation": "#1baf7a"}

#: Widest exact qibojit simulation: a statevector of 2^26 complex128 is 1 GiB,
#: and a density matrix of 13 qubits is the same size.
QIBOJIT_MAX_QUBITS = {"statevector": 26, "density_matrix": 13}


def compare(
    circuit,
    events,
    observable,
    *,
    qibojit=None,
    pauli_propagation=None,
    mpstab_options=None,
    pp_options=None,
    qibojit_max_qubits=None,
):
    """
    Run every available simulator on one case.

    ``qibojit`` is the backend from :func:`qibojit_backend` and
    ``pauli_propagation`` a :class:`PauliPropagation`; pass ``None`` to skip
    either. qibojit is also skipped above ``qibojit_max_qubits`` (default:
    :data:`QIBOJIT_MAX_QUBITS`). Returns one row per simulator, with
    ``error = |value - qibojit|`` when qibojit ran.
    """
    n = circuit.nqubits
    rows = []
    mpstab_options = dict(mpstab_options or {})
    if events:
        rows.append(
            ("mpstab", run_mpstab_noisy(circuit, events, observable, **mpstab_options))
        )
    else:
        mpstab_options.pop("n_samples", None)
        mpstab_options.pop("seed", None)
        rows.append(("mpstab", run_mpstab(circuit, observable, **mpstab_options)))
    limit = (
        qibojit_max_qubits
        or QIBOJIT_MAX_QUBITS["density_matrix" if events else "statevector"]
    )
    if qibojit is not None and n <= limit:
        rows.append(("qibojit", run_qibojit(qibojit, circuit, events, observable)))
    if pauli_propagation is not None:
        rows.append(
            (
                "PauliPropagation",
                pauli_propagation.run(
                    circuit, events, observable, **(pp_options or {})
                ),
            )
        )
    reference = dict(rows).get("qibojit", {}).get("value")
    return [
        dict(
            framework=name,
            n=n,
            gates=len(circuit.queue),
            error=np.nan if reference is None else abs(result["value"] - reference),
            **result,
        )
        for name, result in rows
    ]


def sweep(cases, label="x", **options):
    """
    :func:`compare` over ``cases``, an iterable of ``(x, circuit, events,
    observable)``, as a pandas DataFrame with an ``x`` column named ``label``.
    Prints one line per case as it goes.
    """
    import pandas as pd

    rows = []
    for x, circuit, events, observable in cases:
        case = compare(circuit, events, observable, **options)
        for row in case:
            row[label] = x
        rows += case
        summary = "  ".join(f"{r['framework']} {r['time']:.3g}s" for r in case)
        print(f"{label}={x}: {summary}", flush=True)
    return pd.DataFrame(rows)


def plot(df, x="n", title=None):
    """Wall time and error against qibojit, one line per framework."""
    import matplotlib.pyplot as plt

    fig, (ax_time, ax_error) = plt.subplots(
        1, 2, figsize=(11, 4), constrained_layout=True
    )
    for name, group in df.groupby("framework", sort=False):
        group = group.sort_values(x)
        style = dict(
            color=COLORS.get(name), marker="o", markersize=6, linewidth=2, label=name
        )
        ax_time.plot(group[x], group["time"], **style)
        errors = group.dropna(subset=["error"])
        if len(errors) and name != "qibojit":  # the reference: zero by definition
            # exact zeros would vanish on a log axis
            ax_error.plot(errors[x], errors["error"].clip(lower=1e-17), **style)
        if (
            "stderr" in group
            and group["stderr"].notna().any()
            and group["stderr"].max() > 0
        ):
            ax_error.plot(
                group[x],
                group["stderr"],
                color=COLORS.get(name),
                linestyle="--",
                linewidth=1.5,
                label=f"{name} stderr",
            )
    ax_time.set(yscale="log", xlabel=x, ylabel="wall time [s]", title="time")
    ax_error.set(yscale="log", xlabel=x, ylabel="|value - qibojit|", title="error")
    for ax in (ax_time, ax_error):
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    if title:
        fig.suptitle(title)
    return fig
