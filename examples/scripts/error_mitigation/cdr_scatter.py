"""CDR and vnCDR on a 10-qubit kicked Ising circuit with superconducting-like noise.

Everything is simulated with mpstab: the noiseless values with HSMPO, the
noisy ones with :meth:`~mpstab.HSMPO.noisy_expectation`. The data go to
``cdr_scatter.npz`` and the figure to ``cdr_scatter.png``, next to this file::

    python examples/scripts/error_mitigation/cdr_scatter.py            # compute + plot
    python examples/scripts/error_mitigation/cdr_scatter.py --plot     # plot only
"""

import pathlib
import sys
import time

import numpy as np
from qibo import gates

from mpstab import (
    HSMPO,
    CoherentError,
    DepolarizingError,
    NoiseModel,
    ReadoutError,
    StatePreparationError,
    ThermalRelaxationError,
    cdr,
    noisy_value,
    vncdr,
)
from mpstab.models.ansatze import KickedIsing

HERE = pathlib.Path(__file__).parent
DATA, FIGURE = HERE / "cdr_scatter.npz", HERE / "cdr_scatter.png"

NQUBITS = 10
OBSERVABLE = "I" * 4 + "ZZ" + "I" * 4
LEVELS = (1, 2, 3)
TRAINING = dict(
    n_training_circuits=24, replacement_probability=0.7, n_samples=1000, seed=0
)
N_WORKERS = 8

T1, T2 = 100e-6, 80e-6  # s
T_1Q, T_2Q = 30e-9, 300e-9  # gate times, s


def superconducting_noise(nqubits):
    """Depolarizing, T1/T2, ZZ coupling and crosstalk, idling, SPAM."""
    chain = [(q, q + 1) for q in range(nqubits - 1)]
    noise = NoiseModel()
    noise.add(DepolarizingError(1e-3), gates.RX)
    noise.add(ThermalRelaxationError(T1, T2, T_1Q), gates.RX)
    noise.add(DepolarizingError(1e-2), gates.CNOT)
    noise.add(ThermalRelaxationError(T1, T2, T_2Q), gates.CNOT)
    noise.add(CoherentError("ZZ", 0.02), gates.CNOT)
    noise.add_crosstalk(CoherentError("ZZ", 0.01), chain, gates.CNOT)
    noise.add_idle(ThermalRelaxationError(T1, T2, T_1Q))
    noise.add(StatePreparationError(5e-3))
    noise.add(ReadoutError(0.01, 0.03))
    return noise


def compute():
    ansatz = KickedIsing(
        nqubits=NQUBITS, n_steps=3, theta_h=0.7, theta_z=0.3, theta_j=-1.3
    )
    noise = superconducting_noise(NQUBITS)
    target = HSMPO(ansatz, progress=False)
    print(
        f"{NQUBITS} qubits, {target.n_gates} gates ({target.n_magic_gates} magic), "
        f"{len(noise.events(ansatz.circuit))} noise events",
        flush=True,
    )

    start = time.perf_counter()
    exact = target.expectation(OBSERVABLE)
    target_noisy = [
        noisy_value(target, OBSERVABLE, noise, level, n_samples=2000, seed=1)
        for level in LEVELS
    ]
    t_target = time.perf_counter() - start
    print(
        f"target: exact {exact:+.4f}, noisy {target_noisy} ({t_target:.0f}s)",
        flush=True,
    )

    start = time.perf_counter()
    m_cdr = cdr(ansatz, OBSERVABLE, noise, n_workers=N_WORKERS, **TRAINING)
    t_cdr = time.perf_counter() - start
    print(f"CDR trained in {t_cdr:.0f}s", flush=True)

    start = time.perf_counter()
    m_vn = vncdr(ansatz, OBSERVABLE, noise, LEVELS, n_workers=N_WORKERS, **TRAINING)
    t_vn = time.perf_counter() - start
    print(f"vnCDR trained in {t_vn:.0f}s", flush=True)

    np.savez(
        DATA,
        exact=exact,
        target_noisy=target_noisy,
        levels=LEVELS,
        cdr_exact=m_cdr.exact,
        cdr_noisy=m_cdr.noisy,
        cdr_stderr=m_cdr.noisy_stderr,
        cdr_params=m_cdr.params,
        cdr_mitigated=m_cdr(target_noisy[0]),
        vn_exact=m_vn.exact,
        vn_noisy=m_vn.noisy,
        vn_stderr=m_vn.noisy_stderr,
        vn_params=m_vn.params,
        vn_mitigated=m_vn(target_noisy),
        times=[t_target, t_cdr, t_vn],
    )


def plot():
    import matplotlib.pyplot as plt

    from mpstab.plotting import COLORS, set_style

    set_style()
    d = np.load(DATA)
    exact, levels = float(d["exact"]), list(d["levels"])
    target_noisy = list(d["target_noisy"])
    noisy = target_noisy[0]
    cdr_value, vn_value = float(d["cdr_mitigated"]), float(d["vn_mitigated"])
    a, b = d["cdr_params"]
    t_target, t_cdr, t_vn = d["times"]

    fig, (ax_map, ax_levels) = plt.subplots(1, 2, figsize=(15, 6))

    # The CDR noise map: training circuits, fit, and the target projected on it.
    values = np.concatenate([d["cdr_noisy"], d["cdr_exact"], [noisy, exact]])
    lims = np.array([values.min() - 0.08, values.max() + 0.08])
    ax_map.plot(lims, lims, ":", color=COLORS["gray"], label="no noise")
    ax_map.errorbar(
        d["cdr_noisy"],
        d["cdr_exact"],
        xerr=d["cdr_stderr"],
        fmt="o",
        color=COLORS["blue"],
        label="training circuits",
    )
    ax_map.plot(lims, a * lims + b, "-", color=COLORS["blue"], label="CDR fit")
    ax_map.axhline(exact, ls="--", lw=1.5, color=COLORS["gray"], label="target, exact")
    ax_map.plot(
        [noisy, noisy], [lims[0], cdr_value], "-", lw=1.5, color=COLORS["orange"]
    )
    ax_map.plot(
        noisy, cdr_value, "D", ms=13, color=COLORS["orange"], label="target, mitigated"
    )
    ax_map.set(
        xlim=lims,
        ylim=lims,
        xlabel="noisy value",
        ylabel="noiseless value",
        title=f"CDR noise map ({len(d['cdr_exact'])} training circuits)",
    )
    ax_map.legend(loc="upper left")

    # The target at each noise level, and where CDR and vnCDR bring it.
    ax_levels.axhline(exact, ls="--", lw=1.5, color=COLORS["gray"], label="exact")
    ax_levels.plot(
        levels, target_noisy, "s-", color=COLORS["red"], label="measured (noisy)"
    )
    ax_levels.plot(
        0,
        cdr_value,
        "o",
        ms=13,
        color=COLORS["blue"],
        label=f"CDR, error {abs(cdr_value - exact):.3f}",
    )
    ax_levels.plot(
        0,
        vn_value,
        "D",
        ms=13,
        color=COLORS["orange"],
        label=f"vnCDR, error {abs(vn_value - exact):.3f}",
    )
    ax_levels.set(
        xticks=[0, *levels],
        xlabel=r"noise level $\lambda$",
        ylabel=f"$\\langle Z_{{{NQUBITS // 2 - 1}}} Z_{{{NQUBITS // 2}}} \\rangle$",
        title="Mitigating the target circuit",
    )
    ax_levels.legend(loc="lower left")

    fig.suptitle(
        f"Kicked Ising, {NQUBITS} qubits, superconducting-like noise.  Time: target "
        f"{t_target:.0f} s, CDR {t_cdr:.0f} s, vnCDR {t_vn:.0f} s ({N_WORKERS} workers)"
    )
    fig.savefig(FIGURE)
    print(f"saved {FIGURE}")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        compute()
    plot()
