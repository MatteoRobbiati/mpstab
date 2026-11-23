"""Helpers for the benchmark notebooks: one circuit, three simulators.

Every runner takes the same qibo circuit, the same list of
:mod:`mpstab.models.noise` events and the same observable, and returns a dict
with at least ``value`` and ``time`` (seconds, construction included):

- :func:`run_mpstab` / :func:`run_mpstab_noisy` -- the HSMPO surrogate;
- :func:`run_qibojit` -- qibo's numba backend, statevector without noise and
  density matrix with it: the exact reference;
- :class:`PauliPropagation` -- PauliPropagation.jl through juliacall, a
  truncated Heisenberg-picture Pauli-sum simulator, and :class:`IsolatedPP`,
  the same in a child process with a timeout.

Observables are a Pauli string (qubit 0 leftmost) or a ``{pauli: coefficient}``
dict. Noise is limited to what all three can express: a uniform depolarizing
channel per qubit and Pauli rotations (e.g. ZZ crosstalk).

The module also holds the exact free-fermion reference used by
``free_fermions.ipynb`` and the plotting helpers shared by the notebooks.
Importing it silences qibo's INFO log, juliacall's IPython banner and the
spurious floating-point warnings NumPy's Accelerate matmul emits on macOS.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

# juliacall prints a banner when it detects IPython; the benchmarks do not need
# its IPython extension.
os.environ.setdefault("PYTHON_JULIACALL_AUTOLOAD_IPYTHON_EXTENSION", "no")

import numpy as np
from qibo import Circuit, gates, symbols
from qibo.config import log
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab import HSMPO
from mpstab.engines import QuimbEngine
from mpstab.evolutors.utils import gate_angle, gate_generator
from mpstab.models.ansatze import HardwareEfficient, KickedIsing
from mpstab.models.noise.compile import compile_noise
from mpstab.models.noise.events import CoherentRotation, PauliChannel
from mpstab.models.noise.mc_noise_estimator import noisy_expectation

sys.path.append(str(Path(__file__).resolve().parents[2]))  # examples/
from plotting import COLORS  # noqa: E402

log.setLevel("WARNING")
# NumPy >= 2 on Apple's Accelerate raises divide/overflow/invalid warnings in
# matmul on perfectly finite inputs; the results are correct.
warnings.filterwarnings("ignore", r".* encountered in matmul", RuntimeWarning)

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


def echo_circuit(n: int, steps: int, theta_h=0.7, theta_z=0.3):
    """``steps`` kicked Ising steps ``U`` followed by ``U^dagger``: the identity."""
    forward = kicked_ising(n, steps, theta_h=theta_h, theta_z=theta_z)
    return forward + forward.invert()


def free_fermion_circuit(n: int, steps: int, theta_h, theta_j=np.pi / 2):
    """``RZ(theta_h)`` on every qubit, then ``RXX(theta_j)`` on every bond, ``steps`` times."""
    circuit = Circuit(n)
    for _ in range(steps):
        for q in range(n):
            circuit.add(gates.RZ(q, theta=theta_h))
        for start in (0, 1):  # even bonds, then odd bonds (they commute)
            for q in range(start, n - 1, 2):
                circuit.add(gates.RXX(q, q + 1, theta=theta_j))
    return circuit


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


def run_mpstab(circuit, observable, max_bond=None, light_cone=True, svd_cutoff=None):
    """
    Noiseless HSMPO expectation value.

    With ``light_cone`` only the dressed rotations inside the observable's light
    cone are applied (``expectation_from_partition`` with nothing replaced);
    without it the whole circuit is evolved, as ``HSMPO.expectation`` does.
    ``svd_cutoff`` is quimb's relative singular-value cutoff (``None``: quimb's
    default, 1e-10, which leaves errors around 1e-7; ``0``: keep everything).
    """
    engine = QuimbEngine(svd_cutoff=svd_cutoff)
    t = time.perf_counter()
    if light_cone:
        # rotations_only skips the full evolution HSMPO() runs at construction.
        hsmpo = HSMPO.rotations_only(circuit, tn_engine=engine)
        hsmpo.max_bond_dimension = max_bond
        hsmpo.progress = False  # a benchmark should not time (or print) the bar
        value, partitions = hsmpo.expectation_from_partition(
            observable, replacement_probability=0.0, return_partitions=True
        )
        state = partitions["mps"]
    else:
        hsmpo = HSMPO(
            circuit, max_bond_dimension=max_bond, progress=False, tn_engine=engine
        )
        value = hsmpo.expectation(observable)
        state = hsmpo.original_circuit_mps
    elapsed = time.perf_counter() - t
    return dict(
        value=float(np.real(value)),
        time=elapsed,
        bond=int(hsmpo.tn_engine.bond_dimension(state)),
        fidelity=hsmpo.tn_engine.squared_norm(state),
        magic=len(hsmpo.magic_gates),
    )


def run_mpstab_exact(circuit, observable):
    """:func:`run_mpstab` with no bond cap and no SVD cutoff: no approximation at all."""
    return run_mpstab(circuit, observable, max_bond=None, svd_cutoff=0.0)


def run_mpstab_noisy(
    circuit, events, observable, n_samples=1000, max_bond=None, seed=0, light_cone=True
):
    """Noisy HSMPO expectation value: exact for 0 and 1 faults, sampled beyond."""
    t = time.perf_counter()
    hsmpo = HSMPO.rotations_only(circuit)
    hsmpo.max_bond_dimension = max_bond
    hsmpo.progress = False
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

    def __init__(self, warmup: bool = True):
        import juliapkg

        juliapkg.add("PauliPropagation", PAULI_PROPAGATION_UUID)
        juliapkg.resolve()
        from juliacall import Main as jl

        self.jl = jl
        jl.seval("using PauliPropagation")
        if warmup:
            circuit = kicked_ising(4, 1)
            self.run(circuit, noise_events(circuit, 0.01, 0.01), local_z(4))

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
        return self.run_lines(
            self._circuit(circuit, events),
            circuit.nqubits,
            observable,
            min_abs_coeff=min_abs_coeff,
            max_weight=max_weight,
            repeats=repeats,
        )

    def run_lines(
        self, lines, nqubits, observable, min_abs_coeff=1e-8, max_weight=None, repeats=2
    ):
        """:meth:`run` on a circuit already translated to Julia gate expressions."""
        self.jl.seval(f"circ = [{', '.join(lines)}]")
        self.jl.seval(f"obs = PauliSum({nqubits})")
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
            elif gate.name in ("rxx", "ryy", "rzz"):
                axis = gate.name[1].upper()
                lines.append(
                    f"PauliRotation([:{axis}, :{axis}], {q}, {gate.parameters[0]!r})"
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


def _isolated_worker(lines, warmup_lines, nqubits, observable, options, queue):
    pp = PauliPropagation(warmup=False)
    # Compile for this width and gate set on a short prefix, untimed.
    pp.run_lines(warmup_lines, nqubits, observable, repeats=1)
    queue.put("ready")
    queue.put(pp.run_lines(lines, nqubits, observable, repeats=1, **options))


#: Longest wait for a child Julia process to start and compile, in seconds.
PP_STARTUP_LIMIT = 900


def run_pp_isolated(circuit, events, observable, timeout, warmup_gates=None, **options):
    """
    :meth:`PauliPropagation.run` in a child process, abandoned if the
    propagation takes more than ``timeout`` seconds: Julia cannot be
    interrupted from Python, and a propagation that blows up can run for
    hours. Returns ``None`` on timeout.

    The child starts Julia and compiles ``propagate`` on the first
    ``warmup_gates`` gates (one Floquet step is enough: same width, same gate
    types) before the clock starts; then it propagates the full circuit once.
    Call it under ``if __name__ == "__main__":`` outside notebooks.
    """
    import multiprocessing as mp
    import queue as queues

    lines = PauliPropagation._circuit(circuit, events)
    warmup_lines = lines[: warmup_gates or 4 * circuit.nqubits]
    context = mp.get_context("spawn")
    channel = context.Queue()
    process = context.Process(
        target=_isolated_worker,
        args=(lines, warmup_lines, circuit.nqubits, observable, options, channel),
    )
    process.start()
    try:
        channel.get(timeout=PP_STARTUP_LIMIT)  # "ready": Julia started and compiled
        return channel.get(timeout=timeout)
    except queues.Empty:
        return None
    finally:
        if process.is_alive():
            process.kill()
        process.join()


class IsolatedPP:
    """
    :func:`run_pp_isolated` for sweeps of growing cost.

    Each call runs one truncation threshold on one circuit and returns a row
    (``method``, ``value``, ``error``, ``time``, ``timed_out``). Once a
    threshold times out, later calls with that threshold are skipped (NaN
    time, ``timed_out=False``): the sweeps only make circuits harder.
    """

    def __init__(self, timeout: float):
        self.timeout = timeout
        self.gave_up = set()

    def __call__(
        self, circuit, observable, threshold, reference, warmup_gates=None, events=()
    ):
        method = f"PauliPropagation {threshold:.0e}"
        row = dict(
            method=method, value=np.nan, error=np.nan, time=np.nan, timed_out=False
        )
        if threshold in self.gave_up:
            return row
        result = run_pp_isolated(
            circuit,
            list(events),
            observable,
            timeout=self.timeout,
            warmup_gates=warmup_gates,
            min_abs_coeff=threshold,
        )
        if result is None:
            self.gave_up.add(threshold)
            row["timed_out"] = True
            return row
        row.update(
            value=result["value"],
            error=abs(result["value"] - reference),
            time=result["time"],
        )
        return row


# -- Exact free-fermion reference -------------------------------------------------


def majorana_form(pauli: str):
    """``pauli = phase * c_{a_1} ... c_{a_m}``, with ``a_1 < ... < a_m``."""
    phase, word = 1 + 0j, []
    for i, label in enumerate(pauli):
        if label == "Z":
            phase *= -1j
            word += [2 * i, 2 * i + 1]
        elif label in "XY":
            phase *= (-1j) ** i  # the Jordan-Wigner string Z_0 ... Z_{i-1}
            word += list(range(2 * i)) + [2 * i if label == "X" else 2 * i + 1]
    for end in range(len(word) - 1, 0, -1):  # sort: -1 per swap of distinct modes
        for k in range(end):
            if word[k] > word[k + 1]:
                word[k], word[k + 1] = word[k + 1], word[k]
                phase = -phase
    reduced = []
    for a in word:  # c^2 = 1
        if reduced and reduced[-1] == a:
            reduced.pop()
        else:
            reduced.append(a)
    return phase, reduced


def pfaffian(matrix) -> complex:
    """Pfaffian of an antisymmetric matrix (pivoted Parlett-Reid)."""
    a = np.array(matrix, dtype=complex)
    n = a.shape[0]
    if n % 2:
        return 0.0
    result = 1.0 + 0j
    for k in range(0, n - 1, 2):
        pivot = k + 1 + int(np.argmax(np.abs(a[k, k + 1 :])))
        if pivot != k + 1:
            a[[k + 1, pivot]] = a[[pivot, k + 1]]
            a[:, [k + 1, pivot]] = a[:, [pivot, k + 1]]
            result = -result
        if a[k, k + 1] == 0:
            return 0.0
        result *= a[k, k + 1]
        if k + 2 < n:
            tau = a[k, k + 2 :] / a[k, k + 1]
            column = a[k + 2 :, k + 1].copy()
            a[k + 2 :, k + 2 :] += np.outer(tau, column) - np.outer(column, tau)
    return result


def majorana_covariance(circuit) -> np.ndarray:
    """Majorana covariance ``Gamma_ab = i/2 <[c_a, c_b]>`` of ``circuit|0...0>``."""
    n = circuit.nqubits
    gamma = np.zeros((2 * n, 2 * n))
    for i in range(n):  # Z_i = +1  <=>  <c_2i c_2i+1> = i
        gamma[2 * i, 2 * i + 1], gamma[2 * i + 1, 2 * i] = -1.0, 1.0
    for gate in circuit.queue:
        phase, modes = majorana_form(gate_generator(gate, n))
        if len(modes) != 2:
            raise ValueError(f"{gate.name} is not a free-fermion gate")
        angle = np.sign(phase.imag) * gate_angle(gate)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, s], [-s, c]])
        gamma[modes, :] = rotation @ gamma[modes, :]
        gamma[:, modes] = gamma[:, modes] @ rotation.T
    return gamma


def free_fermion_expectation(gamma, pauli: str) -> float:
    """``<pauli>`` from the Majorana covariance ``gamma``, by Wick's theorem."""
    phase, modes = majorana_form(pauli)
    if len(modes) % 2:
        return 0.0  # odd parity vanishes
    value = pfaffian(-1j * gamma[np.ix_(modes, modes)]) if modes else 1.0
    return float((phase * value).real)


# -- Sweeps -------------------------------------------------------------------------

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
    ``error = |value - qibojit|`` when qibojit ran (NaN for qibojit itself).
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
            method=name,
            n=n,
            gates=len(circuit.queue),
            error=(
                np.nan
                if reference is None or name == "qibojit"
                else abs(result["value"] - reference)
            ),
            **result,
        )
        for name, result in rows
    ]


def sweep(cases, label="x", **options):
    """
    :func:`compare` over ``cases``, an iterable of ``(x, circuit, events,
    observable)``, as a pandas DataFrame with an ``x`` column named ``label``.
    """
    import pandas as pd

    rows = []
    for x, circuit, events, observable in cases:
        for row in compare(circuit, events, observable, **options):
            row[label] = x
            rows.append(row)
    return pd.DataFrame(rows)


def tradeoff(
    circuit, events, observable, reference, mpstab_settings, pp_thresholds, pp
):
    """
    Error and time of mpstab and Pauli propagation over their accuracy knobs:
    ``mpstab_settings`` are bond dimensions (noiseless) or sample counts
    (noisy), ``pp_thresholds`` values of ``min_abs_coeff``.
    """
    import pandas as pd

    rows = []
    for setting in mpstab_settings:
        if events:
            r = run_mpstab_noisy(circuit, events, observable, n_samples=setting, seed=0)
        else:
            r = run_mpstab(circuit, observable, max_bond=setting)
        rows.append(
            dict(
                method="mpstab",
                setting=setting,
                time=r["time"],
                error=abs(r["value"] - reference),
            )
        )
    for threshold in pp_thresholds:
        r = pp.run(circuit, events, observable, min_abs_coeff=threshold)
        rows.append(
            dict(
                method="PauliPropagation",
                setting=threshold,
                time=r["time"],
                error=abs(r["value"] - reference),
            )
        )
    return pd.DataFrame(rows)


# -- Plots ----------------------------------------------------------------------------

#: One color per simulator, from the shared mpstab palette.
FAMILY_COLORS = {
    "mpstab": COLORS["blue"],
    "PauliPropagation": COLORS["red"],
    "qibojit": COLORS["black"],
}
#: Variants of one simulator (bond dimensions, thresholds) share its color and
#: differ by marker and line style, in this order.
VARIANT_MARKERS = ["o", "s", "^", "D"]
VARIANT_LINES = ["-", "--", ":", "-."]
#: Errors below this are drawn at it, so exact zeros stay on a log axis.
ERROR_FLOOR = 1e-16


def method_styles(methods) -> dict:
    """
    A matplotlib style per method name. The family is the first word of the
    name (``"mpstab $\\chi$=64"`` belongs to ``mpstab``) and sets the color;
    the n-th variant of a family gets the n-th marker and line style.
    """
    seen, styles = {}, {}
    for method in methods:
        family = method.split()[0]
        i = seen.get(family, 0)
        seen[family] = i + 1
        styles[method] = dict(
            color=FAMILY_COLORS[family],
            marker=VARIANT_MARKERS[i % len(VARIANT_MARKERS)],
            linestyle=VARIANT_LINES[i % len(VARIANT_LINES)],
            label=method,
        )
    return styles


def plot_methods(df, x, xlabel, title=None, timeout=None, stderr=False):
    """
    Error (left) and wall time (right) against ``x``, one line per method.

    ``df`` has columns ``method``, ``x``, ``error`` and ``time``; rows with a
    NaN error (exact references) are left out of the error panel. Rows with
    ``timed_out`` set are drawn as "x" markers at ``timeout`` on the time
    panel. With ``stderr``, mpstab's statistical error is added as a dotted
    line on the error panel.
    """
    import matplotlib.pyplot as plt

    fig, (ax_error, ax_time) = plt.subplots(1, 2, figsize=(13, 5.4))
    styles = method_styles(df["method"].unique())
    for method, d in df.groupby("method", sort=False):
        d = d.sort_values(x)
        style = styles[method]
        done = d.dropna(subset=["time"])
        ax_time.plot(done[x], done["time"], **style)
        errors = done.dropna(subset=["error"])
        if len(errors):
            ax_error.plot(errors[x], errors["error"].clip(lower=ERROR_FLOOR), **style)
        if stderr and "stderr" in d and d["stderr"].notna().any():
            ax_error.plot(
                d[x],
                d["stderr"],
                color=style["color"],
                linestyle=":",
                label=f"{method} stderr",
            )
        if "timed_out" in d and d["timed_out"].any():
            hit = d[d["timed_out"].astype(bool)]
            ax_time.plot(
                hit[x], [timeout] * len(hit), "X", color=style["color"], markersize=13
            )
    if "timed_out" in df and df["timed_out"].any():
        ax_time.axhline(
            timeout,
            color=COLORS["gray"],
            linestyle=":",
            label=f"timeout ({timeout:g} s)",
        )
    for ax in (ax_error, ax_time):
        ax.set_xticks(sorted(df[x].unique()))
    ax_error.set(yscale="log", xlabel=xlabel, ylabel="|value - exact|", title="Error")
    ax_time.set(yscale="log", xlabel=xlabel, ylabel="wall time (s)", title="Time")
    # One legend for both panels, below them, so it never hides a curve.
    legend = {}
    for ax in (ax_time, ax_error):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, handle)
    fig.legend(
        legend.values(),
        legend.keys(),
        loc="outside lower center",
        ncol=len(legend) if len(legend) <= 4 else -(-len(legend) // 2),
    )
    if title:
        fig.suptitle(title)
    plt.show()


def plot_tradeoff(panels):
    """
    Error against wall time, one panel per ``(title, df)`` from :func:`tradeoff`,
    each point labelled with the value of its accuracy knob.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5))
    for ax, (title, df) in zip(np.atleast_1d(axes), panels):
        styles = method_styles(df["method"].unique())
        # mpstab's labels go left of its points, Pauli propagation's right.
        for side, (method, d) in zip((-1, 1), df.groupby("method", sort=False)):
            ax.plot(d["time"], d["error"].clip(lower=ERROR_FLOOR), **styles[method])
            for _, row in d.iterrows():
                ax.annotate(
                    f"{row['setting']:g}",
                    (row["time"], max(row["error"], ERROR_FLOOR)),
                    textcoords="offset points",
                    xytext=(9 * side, 0),
                    ha="left" if side > 0 else "right",
                    va="center",
                    color=styles[method]["color"],
                )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="wall time (s)",
            ylabel="|value - exact|",
            title=title,
        )
        ax.margins(x=0.15)
        ax.legend()
    plt.show()
