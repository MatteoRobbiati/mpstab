"""Shared pieces of the noise-assessment benchmarks: circuits, device noise, masses.

Everything here uses the public event types of ``mpstab.models.noise.events`` and
``compile_noise``; nothing in mpstab is patched.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
from qibo import Circuit, gates

from mpstab.models.noise.events import CoherentRotation, KrausChannel, PauliChannel
from mpstab.models.noise.model import ThermalRelaxationError

RESULTS = Path(__file__).parent / "results"

# Gate durations in seconds; RZ is a virtual frame change on transmons.
DURATIONS = {"RX": 30e-9, "H": 30e-9, "CZ": 150e-9, "CNOT": 150e-9, "RZ": 0.0}


@dataclass(frozen=True)
class Device:
    """Placeholder device parameters of the task (times in seconds)."""

    t1: float = 100e-6
    t2: float = 80e-6
    p1q: float = 1e-3  # DepolarizingError lam on physical 1q gates
    p2q: float = 5e-3  # DepolarizingError lam on 2q gates
    zz_hz: float = 50e3  # static ZZ shift zeta (placeholder, not given)
    readout_p0: float = 0.01  # placeholder, not given
    readout_p1: float = 0.03  # placeholder, not given
    gate_thermal: bool = True  # T1/T2 during gates, on top of depolarizing


# -- Circuits ------------------------------------------------------------------


def kicked_ising_layered(
    nqubits: int,
    n_steps: int,
    theta_h: float = 0.7,
    theta_z: float = 0.3,
    theta_j: float = -1.3,
    native: str = "cnot",
) -> Circuit:
    """
    The KickedIsing circuit of the error-mitigation tutorial, queued layer by layer.

    Same gates as ``mpstab.models.ansatze.KickedIsing`` (RX, RZ kicks; RZZ as
    CNOT RZ CNOT on even then odd bonds), only reordered so that commuting gates
    of one layer are consecutive in the queue. With ``native="cz"`` each CNOT is
    written as H CZ H on the target.
    """
    c = Circuit(nqubits)
    bonds = [[(q, q + 1) for q in range(s, nqubits - 1, 2)] for s in (0, 1)]

    def cnot_layer(layer):
        if native == "cz":
            c.add([gates.H(b) for _, b in layer])
            c.add([gates.CZ(a, b) for a, b in layer])
            c.add([gates.H(b) for _, b in layer])
        else:
            c.add([gates.CNOT(a, b) for a, b in layer])

    for _ in range(n_steps):
        c.add([gates.RX(q, theta=theta_h) for q in range(nqubits)])
        c.add([gates.RZ(q, theta=theta_z) for q in range(nqubits)])
        for layer in bonds:
            cnot_layer(layer)
            c.add([gates.RZ(b, theta=theta_j) for _, b in layer])
            cnot_layer(layer)
    return c


def gate_name(gate) -> str:
    return type(gate).__name__


def moments(circuit: Circuit):
    """
    In-order layering, the rule ``NoiseModel.add_idle`` uses: a gate lands one
    moment after the last gate on its qubits and never before the gate queued
    ahead of it. Returns ``(moment of each queue index, history per qubit)``.
    """
    depth = [0] * circuit.nqubits
    moment, of_gate = 0, []
    history = {q: [] for q in range(circuit.nqubits)}
    for index, gate in enumerate(circuit.queue):
        moment = max([moment] + [depth[q] for q in gate.qubits])
        for q in gate.qubits:
            depth[q] = moment + 1
            history[q].append((moment, index))
        of_gate.append(moment)
    return of_gate, history


def depth_stats(circuit: Circuit) -> dict:
    of_gate, _ = moments(circuit)
    n_moments = max(of_gate) + 1 if of_gate else 0
    by_moment = [[] for _ in range(n_moments)]
    for index, m in enumerate(of_gate):
        by_moment[m].append(gate_name(circuit.queue[index]))
    durations = [max(DURATIONS[g] for g in names) for names in by_moment]
    two_qubit = sum(any(g in ("CZ", "CNOT") for g in names) for names in by_moment)
    return {
        "moments": n_moments,
        "moments_nonzero_duration": int(sum(d > 0 for d in durations)),
        "two_qubit_layers": int(two_qubit),
        "duration_us": float(sum(durations) * 1e6),
    }


# -- Device noise as tagged events ----------------------------------------------


def _depolarizing(index, support, lam):
    k = len(support)
    labels = ("".join(p) for p in product("IXYZ", repeat=k))
    probs = {lab: lam / 4**k for lab in labels if set(lab) != {"I"}}
    return PauliChannel(index, tuple(support), probs)


def _thermal(index, q, device: Device, time: float):
    error = ThermalRelaxationError(t1=device.t1, t2=device.t2, time=time)
    return KrausChannel(index, (q,), error.kraus)


def _last_before(history, q, moment):
    previous = -1
    for m, index in history[q]:
        if m >= moment:
            break
        previous = index
    return previous


def device_events(circuit: Circuit, device: Device) -> dict[str, list]:
    """
    Noise events of ``device`` on ``circuit``, by source.

    - ``dep_1q``: depolarizing after every physical 1q gate (RX, H; RZ is virtual).
    - ``dep_2q``: two-qubit depolarizing after every CZ/CNOT.
    - ``gate_thermal``: thermal relaxation on each gate qubit for the gate time.
    - ``idle_thermal``: thermal relaxation on every qubit for the part of each
      moment it is not driven (moment duration minus its own gate's); zero-length
      moments (virtual RZ only) add nothing.
    - ``idle_zz``: static ZZ, ``H = (2 pi zeta / 4) ZZ``, i.e. the rotation
      ``exp(-i (pi zeta t / 2) ZZ)`` per nearest-neighbour pair per moment
      for the moment duration, except on a pair driven by its own two-qubit
      gate in that moment. Coherent: it joins the
      rotation chain and adds no quasi-probability mass.
    """
    n = circuit.nqubits
    queue = circuit.queue
    of_gate, history = moments(circuit)
    events = {
        k: [] for k in ("dep_1q", "dep_2q", "gate_thermal", "idle_thermal", "idle_zz")
    }
    for index, gate in enumerate(queue):
        name = gate_name(gate)
        if name in ("RX", "H"):
            events["dep_1q"].append(_depolarizing(index, gate.qubits, device.p1q))
        elif name in ("CZ", "CNOT"):
            events["dep_2q"].append(_depolarizing(index, gate.qubits, device.p2q))
        if device.gate_thermal and DURATIONS[name] > 0:
            for q in gate.qubits:
                events["gate_thermal"].append(
                    _thermal(index, q, device, DURATIONS[name])
                )

    n_moments = max(of_gate) + 1 if of_gate else 0
    busy = [dict() for _ in range(n_moments)]  # qubit -> own gate duration
    driven_pairs = [
        set() for _ in range(n_moments)
    ]  # ZZ there is in the gate calibration
    for index, m in enumerate(of_gate):
        for q in queue[index].qubits:
            busy[m][q] = DURATIONS[gate_name(queue[index])]
        if len(queue[index].qubits) == 2:
            driven_pairs[m].add(tuple(sorted(queue[index].qubits)))
    for m in range(n_moments):
        duration = max(busy[m].values(), default=0.0)
        if duration == 0:
            continue
        for q in range(n):
            idle = duration - busy[m].get(q, 0.0)
            if idle > 0:
                anchor = _last_before(history, q, m)
                events["idle_thermal"].append(_thermal(anchor, q, device, idle))
        if device.zz_hz > 0:
            for q in range(n - 1):
                if (q, q + 1) in driven_pairs[m]:
                    continue
                anchor = max(
                    _last_before(history, q, m), _last_before(history, q + 1, m)
                )
                label = "I" * q + "ZZ" + "I" * (n - q - 2)
                angle = 2 * np.pi * device.zz_hz * duration / 2
                events["idle_zz"].append(CoherentRotation(anchor, label, angle))
    return events


# -- Masses and strata ----------------------------------------------------------


def strata(
    location_probs,
    coherence_mass,
    fault_probs,
    fault_location,
    coherence_weights,
    coherence_location,
):
    """
    ``(W0, W1, W>=2)`` with the formulas of ``noisy_expectation`` (|chi| counted
    for coherences), computed independently here and validated against it.
    """
    null = 1.0 - location_probs
    mass = location_probs + coherence_mass
    prefix = np.concatenate([[1.0], np.cumprod(null)[:-1]])
    suffix = np.concatenate([np.cumprod(null[::-1])[::-1][1:], [1.0]])
    loo = prefix * suffix
    w0 = float(np.prod(null))
    w1 = float(
        (fault_probs * loo[fault_location]).sum()
        + (2 * np.abs(coherence_weights) * loo[coherence_location]).sum()
    )
    tails = np.array([1.0, 0.0, 0.0])
    for m, q in zip(mass[::-1], null[::-1]):
        tails = np.array(
            [
                (m + q) * tails[0],
                m * tails[0] + q * tails[1],
                m * tails[1] + q * tails[2],
            ]
        )
    return w0, w1, float(tails[2]), float(tails[0])


def summarize(compiled) -> dict:
    """Masses, strata and pattern counts of a (possibly restricted) CompiledNoise."""
    twirl = float(compiled.location_probs.sum())
    coh = float(compiled.coherence_mass.sum())
    w0, w1, w2, norm = strata(
        compiled.location_probs,
        compiled.coherence_mass,
        compiled.fault_probs,
        compiled.fault_location,
        compiled.coherence_weights,
        compiled.coherence_location,
    )
    flips = compiled.flips
    if flips.shape[1]:
        patterns = np.unique(
            np.vstack([np.zeros((1, flips.shape[1]), bool), flips]), axis=0
        )
        n_patterns = len(patterns)
    else:
        n_patterns = 1
    return {
        "locations": int(len(compiled.location_probs)),
        "rotations": int(len(compiled.generators)),
        "faults": int(len(compiled.fault_probs)),
        "M": twirl + coh,
        "M_twirl": twirl,
        "M_coherence": coh,
        "W0": w0,
        "W1": w1,
        "W2plus": w2,
        "quasi_norm": norm,
        "k1_patterns": int(n_patterns),
        "k1_transitions": int(len(compiled.coherence_weights)),
    }


# -- Bookkeeping ----------------------------------------------------------------


def environment() -> dict:
    import qibo
    import quimb
    import stim

    import mpstab

    try:  # the commit of the mpstab actually imported, not of this script
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(mpstab.__file__).parent,
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    return {
        "machine": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "mpstab_commit": commit,
        "mpstab_path": str(Path(mpstab.__file__).parent),
        "qibo": qibo.__version__,
        "quimb": quimb.__version__,
        "stim": stim.__version__,
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def device_dict(device: Device) -> dict:
    return asdict(device)
