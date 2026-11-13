"""
What every measurement method shares: the plan it builds before anything runs,
and the result it estimates afterwards.

A :class:`MeasurementPlan` fixes every circuit and its shot count up front, so
the same plan can be replayed on a simulator or on hardware, and estimation only
ever reads back frequencies.

Basis rotation is left to qibo: ``gates.M(*qubits, basis=[...])`` measures each
qubit in the given single-qubit Pauli basis, so no basis-change circuit is built
here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qibo import Circuit, gates

_BASIS_GATES = {"X": gates.X, "Y": gates.Y, "Z": gates.Z}


@dataclass(frozen=True)
class MeasurementPlan:
    """
    Circuits and their shot allocation, with whatever the estimator needs to
    recombine the results.

    ``recombination`` is route-specific:

    - ``"pauli"``: ``(groups, coefficients)``, one :class:`~mpstab.qpu.measurement.pauli.QWCGroup` per circuit
      in the same order, plus the ``{pauli_string: c_P}`` map the groups came
      from.
    - ``"shadows"``: ``(mpo_terms, bases)``, where ``mpo_terms`` holds
      ``(label, coefficient, sign, mpo)`` per Hamiltonian term and ``bases`` has
      one basis string per circuit.

    Attributes:
        retained_weight: ``"pauli"`` only, ``None`` for ``"shadows"``/``"tnice"``
            (which sample nothing): the |coefficient|-weighted average, over
            Hamiltonian terms, of each term's
            :attr:`~mpstab.qpu.measurement.pauli_expansion.PauliEnsemble.retained_weight`
            -- the fraction of the tail-folded observable's Frobenius weight
            that ``n_string_samples`` actually covered.
        bond_truncation_l2: ``"pauli"`` only, ``None`` for ``"shadows"``/``"tnice"``
            (whose ``truncation_l2`` already is this quantity): the bias
            estimate of folding the tail at a finite bond dimension,
            ``sum_terms |coefficient| * sqrt(1 - ||O'||_F**2 / 2**n)``, from the
            norm each folded MPO lost. ``0.0`` for an untruncated fold. The
            ``"pauli"`` route's ``truncation_l1``/``truncation_l2`` only count
            the Pauli strings left unsampled *of the already truncated MPO*,
            so they miss it.
        antihermitian_l2: ``"pauli"`` only, ``None`` otherwise: the size of the
            anti-Hermitian part the bond truncation left in the sampled
            coefficients and that was projected out,
            ``sqrt(sum_P Im(c_P)**2)``. A diagnostic of how hard the
            truncation hit, not an error of the estimate: the projection never
            moves the operator away from the exact, Hermitian one.
    """

    method: str
    circuits: tuple
    shots: tuple
    recombination: object
    constant: float
    truncation_l1: float | None
    truncation_l2: float
    retained_weight: float | None = None
    bond_truncation_l2: float | None = None
    antihermitian_l2: float | None = None


@dataclass(frozen=True)
class ExpectationResult:
    """
    A measured expectation value, its shot noise and its truncation budget.

    Attributes:
        value: the (real) expectation value.
        stderr: standard error from shot noise alone.
        truncation_l1: heuristic discarded-Pauli-mass estimate (see
            :func:`~mpstab.qpu.measurement.pauli_expansion.truncation_error_estimate`
            -- an extrapolation, not a rigorous bound); ``None`` for the
            ``"shadows"``/``"tnice"`` routes, whose bond truncation has no
            L1/L2 split.
        truncation_l2: typical-case truncation estimate -- Pauli-set truncation
            for ``"pauli"``, MPO bond truncation for ``"shadows"``/``"tnice"``.
        n_settings: number of distinct circuits the shots came from.
        n_shots: total shots used.
        retained_weight: ``"pauli"`` only, ``None`` otherwise -- see
            :attr:`~mpstab.qpu.measurement.abstract.MeasurementPlan.retained_weight`.
        state_fidelity: ``"simulation"`` only, ``None`` otherwise: the squared
            norm of the head MPS, which its truncation pushes below 1. The
            hardware methods run the head on the device, so they have no head
            truncation to report.
        bond_truncation_l2: ``"pauli"`` only, ``None`` otherwise (there
            ``truncation_l2`` already is the bond truncation): the bias
            estimate of folding the tail at a finite bond dimension -- see
            :attr:`~mpstab.qpu.measurement.abstract.MeasurementPlan.bond_truncation_l2`.
            Part of ``total_error``.
        antihermitian_l2: ``"pauli"`` only, ``None`` otherwise: the size of the
            anti-Hermitian part the bond truncation left in the sampled
            coefficients, which was projected out -- see
            :attr:`~mpstab.qpu.measurement.abstract.MeasurementPlan.antihermitian_l2`.
            A diagnostic, not part of ``total_error``.
    """

    value: float
    stderr: float
    truncation_l1: object
    truncation_l2: float
    n_settings: int
    n_shots: int
    retained_weight: float | None = None
    state_fidelity: float | None = None
    bond_truncation_l2: float | None = None
    antihermitian_l2: float | None = None

    @property
    def total_error(self) -> float:
        """
        A practical accuracy *bound*, not a standard error itself:
        ``sqrt(stderr**2 + truncation_l2**2 + bond_truncation_l2**2)``
        (the last term for ``"pauli"`` only, where ``truncation_l2`` is the
        Pauli-set truncation and the bond truncation is reported apart),
        combining contributions of different nature in quadrature. ``stderr`` is a statistical quantity
        (shot noise, shrinks with more shots); ``truncation_l2`` is a
        systematic-bias estimate (bond-dimension or Pauli-coverage
        truncation, independent of shot count -- see ``truncation_l1`` and
        ``retained_weight`` to tell which one is dominating). Combining them
        in quadrature is a convenient, common convention, not a rigorous
        derivation of a single confidence interval -- but it is still the
        number to report in a results table, since ``stderr`` alone omits the
        truncation bias entirely. For ``"simulation"`` it covers the tail
        truncation only; the head truncation shows in ``state_fidelity``.
        """
        bond = self.bond_truncation_l2 or 0.0
        return float(np.sqrt(self.stderr**2 + self.truncation_l2**2 + bond**2))

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self) -> str:
        return (
            f"ExpectationResult({self.value:+.6f}, total_error={self.total_error:.6f}, "
            f"n_shots={self.n_shots}, n_settings={self.n_settings})"
        )


def basis_measurement(basis: str, nqubits: int) -> Circuit:
    """A single ``gates.M`` layer measuring each qubit in its given Pauli basis."""
    circuit = Circuit(nqubits)
    circuit.add(
        gates.M(*range(nqubits), basis=[_BASIS_GATES[label] for label in basis])
    )
    return circuit
