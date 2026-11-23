"""Error mitigation of an HQSMPO measurement plan with variable-noise CDR.

An :class:`~mpstab.evolutors.hqsmpo.HQSMPO` plan runs one head circuit under
many measurement settings, each measuring a group of Pauli strings of the
folded observable ``O = sum_P c_P P``. Its noise splits in two:

- the gates of the head, the same for every circuit of the plan, mitigated
  with variable-noise Clifford data regression (vnCDR);
- the measurement layer, basis rotations and readout, which changes with the
  setting, calibrated once for every Pauli string at a low overhead.

**One map, fitted and applied per setting.** The training circuits are
low-magic variants of the head. HSMPO gives the noiseless value of every
setting of a training circuit, so each training measurement, one circuit in
one setting at every noise level, is a data point of its own: one vnCDR map is
fitted on all of them and applied to every setting of the target. Fitting the
map on whole-observable values instead would fail whenever terms of opposite
sign partly cancel the noise in the sum.

**Choosing the data.** Training is cheap when the data are informative:

- *circuits*: a pool of candidates is sampled and the training set picked to
  spread its noiseless values evenly, since clustered values carry little
  information on the map (Czarnik et al., arXiv:2204.07109);
- *settings*: each training circuit can measure only a few settings, drawn by
  the size of their noiseless contribution to it. A setting worth nothing on a
  training circuit says nothing about the map.

**The measurement layer** is calibrated by preparing both eigenstates of the
X, Y and Z bases on every qubit and measuring them in the same basis, six
circuits in all. A noisy measurement of qubit ``q`` in basis ``B`` reads
``z' = c z + a`` on average, with an attenuation ``c`` and, for an asymmetric
readout, an offset ``a``, both read off the two eigenstates. On estimation each
factor ``z'_q`` of a shot's parity becomes ``(z'_q - a_q) / c_q``, which undoes
the measurement noise exactly when it acts on each qubit independently.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from qibo import Circuit, gates

from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz
from mpstab.models.error_mitigation.folding import fold_circuit
from mpstab.qibo_tools.utilized_backends.statevector_exact_simulator import (
    StatevectorExactSimulator,
)
from mpstab.qpu.execution import execute_plan
from mpstab.qpu.measurement.abstract import (
    basis_measurement,
    weighted_mean_and_variance,
)
from mpstab.qpu.measurement.pauli import (
    PauliMeasurementPlan,
    PauliSampling,
    allocate_shots_by_variance,
)

#: Single-qubit gates preparing the +1 eigenstate of each basis from ``|0>``;
#: an X first prepares the -1 eigenstate.
_EIGENSTATE_PREPARATION = {"X": (gates.H,), "Y": (gates.H, gates.S), "Z": ()}


@dataclass(frozen=True)
class _Measurement:
    """One circuit of a :class:`MitigationPlan` and how its counts are used."""

    kind: str  # "target", "training" or "calibration"
    level: int | None = None
    training_index: int | None = None
    group: int | None = None  # QWC group of the target plan
    basis: str | None = None  # calibration basis
    eigenvalue: int = 1  # calibration eigenstate


@dataclass(frozen=True)
class MitigatedResult:
    """
    The result of :meth:`MitigationPlan.estimate`.

    Attributes:
        value: the vnCDR-mitigated expectation value.
        unmitigated: the plain estimate of the target at noise level 1, as the
            plan alone would give it.
        measurement_mitigated: the same, with the measurement layer calibrated
            out; ``None`` without calibration.
        cdr: the CDR-mitigated value, from level 1 only, for comparison: one
            factor rescaling every setting.
        stderr: standard error of ``value`` from the target's shots alone; the
            uncertainty of the fitted map is not included.
        settings: the mitigated value of each setting of the target, whose sum
            plus the observable's constant is ``value``.
        settings_noisy: each setting's measured value at every noise level,
            measurement calibrated when available, shape
            ``(n_settings, len(noise_levels))``.
        params: the vnCDR coefficients, one per noise level.
        training_exact: noiseless value of each training measurement, one per
            training circuit and measured setting.
        training_noisy: their measured values, shape
            ``(n_points, len(noise_levels))``.
        n_shots: shots spent by the whole plan.
    """

    value: float
    unmitigated: float
    measurement_mitigated: float | None
    cdr: float
    stderr: float
    settings: np.ndarray
    settings_noisy: np.ndarray
    params: np.ndarray
    training_exact: np.ndarray
    training_noisy: np.ndarray
    n_shots: int

    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class MitigationPlan:
    """
    Every circuit to run for a mitigated measurement, with its shots.

    It runs anywhere a :class:`~mpstab.qpu.measurement.abstract.MeasurementPlan`
    does, e.g. :func:`~mpstab.qpu.execution.execute.execute_plan` or
    :func:`~mpstab.qpu.execution.storage.save_circuits`; :meth:`estimate` turns
    the counts into a :class:`MitigatedResult`.

    Attributes:
        target: the plan of the circuit of interest, at noise level 1.
        noise_levels: the folding levels, odd integers starting at 1.
        circuits: every circuit to run.
        shots: their shot counts.
        measurements: what each circuit is for, in the same order.
        training_exact: noiseless value of the whole observable on each
            training circuit.
        training_group_exact: noiseless value of each setting on each training
            circuit, shape ``(n_training, n_settings)``.
        training_circuits: the training circuits, without measurement.
    """

    target: object
    noise_levels: tuple
    circuits: tuple
    shots: tuple
    measurements: tuple
    training_exact: np.ndarray
    training_group_exact: np.ndarray = field(repr=False)
    training_circuits: tuple = field(default=(), repr=False)

    def estimate(self, frequencies) -> MitigatedResult:
        """
        Mitigate from one ``{bitstring: count}`` dict per circuit, in
        :attr:`circuits` order.
        """
        if len(frequencies) != len(self.circuits):
            raise ValueError(
                f"Expected {len(self.circuits)} frequency dicts, got {len(frequencies)}."
            )
        groups, coefficients = self.target.recombination
        nqubits = self.target.circuits[0].nqubits
        calibration = _calibration(self.measurements, frequencies, nqubits)

        levels = {level: k for k, level in enumerate(self.noise_levels)}
        n_settings = len(groups)
        target = np.zeros((n_settings, len(self.noise_levels)))
        target_variance = np.zeros_like(target)
        raw = np.zeros(n_settings)
        training: dict = {}
        for measurement, freq in zip(self.measurements, frequencies):
            kind, level = measurement.kind, measurement.level
            i, g = measurement.training_index, measurement.group
            if kind == "calibration":
                continue
            k = levels[level]
            mean, variance = _group_value(groups[g], coefficients, freq, calibration)
            if kind == "target":
                target[g, k], target_variance[g, k] = mean, variance
                if level == 1:
                    raw[g] = _group_value(groups[g], coefficients, freq, None)[0]
            else:
                training.setdefault((i, g), np.zeros(len(self.noise_levels)))[k] = mean

        points = sorted(training)
        noisy = np.array([training[point] for point in points])
        exact = np.array([self.training_group_exact[i, g] for i, g in points])
        params, *_ = np.linalg.lstsq(noisy, exact, rcond=None)
        slope = float(noisy[:, 0] @ exact / (noisy[:, 0] @ noisy[:, 0]))

        constant = self.target.constant
        level_one = target[:, levels[1]]
        settings = target @ params
        return MitigatedResult(
            value=float(settings.sum() + constant),
            unmitigated=float(raw.sum() + constant),
            measurement_mitigated=(
                float(level_one.sum() + constant) if calibration is not None else None
            ),
            cdr=float(slope * level_one.sum() + constant),
            stderr=float(np.sqrt(np.sum(target_variance @ params**2))),
            settings=settings,
            settings_noisy=target,
            params=params,
            training_exact=exact,
            training_noisy=noisy,
            n_shots=int(sum(self.shots)),
        )


@dataclass(frozen=True)
class VnCDRMitigation:
    """
    Variable-noise CDR on the head of an HQSMPO plan, plus a calibration of
    the measurement layer.

    Args:
        noise_levels: folding levels of the head, odd integers starting at 1:
            level ``2k + 1`` runs ``U (U^dag U)^k``. Only the head is folded,
            not the measurement layer.
        n_training_circuits: training circuits in the regression.
        replacement_probability: chance of replacing each non-Clifford
            rotation of the head by a Clifford one in a training circuit.
        replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
        candidate_factor: candidates sampled per training circuit; the
            training set is the subset whose noiseless values spread most
            evenly. ``1`` keeps the first candidates.
        training_settings: settings each training circuit measures, drawn
            without replacement by the size of their noiseless contribution
            to it; ``None`` measures all of them, like the target.
        training_shots: shots per training circuit and noise level, split over
            its settings.
        calibrate_measurement: calibrate the measurement layer, attenuation
            and offset of every qubit in every basis, and undo it on every
            shot.
        calibration_shots: shots per calibration circuit, six in all.
    """

    noise_levels: tuple = (1, 3, 5)
    n_training_circuits: int = 16
    replacement_probability: float = 0.7
    replacement_method: str = "closest"
    candidate_factor: int = 4
    training_settings: int | None = 8
    training_shots: int = 2000
    calibrate_measurement: bool = True
    calibration_shots: int = 4000

    def __post_init__(self):
        levels = tuple(self.noise_levels)
        if levels[0] != 1 or any(
            level < 1 or level % 2 == 0 or level != int(level) for level in levels
        ):
            raise ValueError(
                f"noise_levels must be odd integers starting at 1, got {levels}."
            )
        object.__setattr__(self, "noise_levels", tuple(int(level) for level in levels))

    def build_plan(
        self,
        model,
        observable,
        cut_index: int,
        n_shots: int | None = None,
        epsilon: float | None = None,
        method: PauliSampling | None = None,
        max_bond_dimension=-1,
        seed: int | None = None,
    ) -> MitigationPlan:
        """
        The target plan of ``model`` at ``cut_index``, folded to every noise
        level, plus the training and calibration circuits.

        Args:
            model: the :class:`~mpstab.evolutors.hqsmpo.HQSMPO`.
            observable: as in :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.plan`.
            cut_index: the split point.
            n_shots: shots of the target plan at each noise level; exactly one
                of this and ``epsilon``.
            epsilon: target standard error sizing those shots instead.
            method: the ``"pauli"`` measurement method, with its options.
            max_bond_dimension: bond cap of the tail fold, as in the plan.
            seed: seed of the plan, the training circuits and the settings.
        """
        method = method or PauliSampling()
        if not isinstance(method, PauliSampling):
            raise NotImplementedError("Mitigation plans support the 'pauli' method.")
        rng = np.random.default_rng(seed)
        target = model.plan(
            observable,
            cut_index,
            method=method,
            n_shots=n_shots,
            epsilon=epsilon,
            max_bond_dimension=max_bond_dimension,
            seed=int(rng.integers(2**32)),
            out_folder=False,
        )
        head = model.resynthesize_head(
            cut_index, observable=observable if model.light_cone else None
        ).circuit
        groups, coefficients = target.recombination
        nqubits = head.nqubits

        training, exact, group_exact = self._training_set(
            head, groups, coefficients, model, int(rng.integers(2**32))
        )
        circuits, shots, measurements = [], [], []

        def add(preparation, basis, n, measurement, prebuilt=None):
            circuits.append(prebuilt or preparation + basis_measurement(basis, nqubits))
            shots.append(int(n))
            measurements.append(measurement)

        for level in self.noise_levels:
            folded = fold_circuit(head, level)
            for g, group in enumerate(groups):
                add(
                    folded,
                    group.setting,
                    target.shots[g],
                    _Measurement("target", level, group=g),
                    prebuilt=target.circuits[g] if level == 1 else None,
                )

        for i, (circuit, draws) in enumerate(
            zip(training, self._training_settings(target, group_exact, rng))
        ):
            for level in self.noise_levels:
                folded = fold_circuit(circuit, level)
                for g, n in draws:
                    add(
                        folded,
                        groups[g].setting,
                        n,
                        _Measurement("training", level, i, g),
                    )

        if self.calibrate_measurement:
            for basis in "XYZ":
                for eigenvalue in (1, -1):
                    add(
                        _eigenstates(basis, nqubits, eigenvalue),
                        basis * nqubits,
                        self.calibration_shots,
                        _Measurement("calibration", basis=basis, eigenvalue=eigenvalue),
                    )

        return MitigationPlan(
            target=target,
            noise_levels=self.noise_levels,
            circuits=tuple(circuits),
            shots=tuple(shots),
            measurements=tuple(measurements),
            training_exact=exact,
            training_group_exact=group_exact,
            training_circuits=tuple(training),
        )

    def _training_set(self, head, groups, coefficients, model, seed):
        """
        ``n_training_circuits`` low-magic variants of ``head`` whose noiseless
        values of ``sum_P c_P P`` spread most evenly over their range, out of
        ``candidate_factor`` times as many candidates, with those values and
        each setting's share of them.
        """
        rng = random.Random(seed)
        ansatz = CircuitAnsatz(qibo_circuit=head)
        n_candidates = self.n_training_circuits * max(1, self.candidate_factor)
        candidates, group_values, seen = [], [], set()
        for _ in range(n_candidates):
            circuit = ansatz.partitionate_circuit(
                self.replacement_probability, self.replacement_method, rng=rng
            )[1]
            key = tuple(
                (gate.name, gate.qubits, tuple(np.ravel(gate.parameters)))
                for gate in circuit.queue
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(circuit)
            surrogate = HSMPO(
                circuit, max_bond_dimension=model.max_bond_dimension, progress=False
            )
            paulis = _pauli_values(surrogate, coefficients)
            group_values.append(
                [
                    sum(coefficients[m] * paulis[m] for m in group.members)
                    for group in groups
                ]
            )
        if len(candidates) < self.n_training_circuits:
            raise ValueError(
                f"Only {len(candidates)} distinct training circuits out of "
                f"{n_candidates} candidates; lower replacement_probability or "
                "use replacement_method='random'."
            )
        group_values = np.asarray(group_values)
        values = group_values.sum(axis=1)
        chosen = _spread(values, self.n_training_circuits)
        return [candidates[k] for k in chosen], values[chosen], group_values[chosen]

    def _training_settings(self, target, group_exact, rng) -> list:
        """
        ``(group, shots)`` of each training circuit's measurements: every
        setting with Neyman shots, or ``training_settings`` settings drawn
        without replacement by the size of their noiseless contribution to
        that circuit, sharing ``training_shots`` evenly.
        """
        groups, coefficients = target.recombination
        n = self.n_training_circuits
        if self.training_settings is None or self.training_settings >= len(groups):
            shots = allocate_shots_by_variance(
                PauliMeasurementPlan(groups, coefficients), self.training_shots
            )
            return [list(enumerate(shots))] * n
        draws = []
        for contribution in np.abs(group_exact):
            informative = np.flatnonzero(contribution > 1e-12)
            m = min(self.training_settings, len(informative))
            if m == 0:  # nothing to learn from: measure the largest settings
                informative, contribution = np.arange(len(groups)), np.ones(len(groups))
                m = self.training_settings
            weights = contribution[informative]
            chosen = rng.choice(
                informative, size=m, replace=False, p=weights / weights.sum()
            )
            per_setting = max(1, self.training_shots // m)
            draws.append([(int(g), per_setting) for g in sorted(chosen)])
        return draws


def mitigated_expectation_at_cut(
    model,
    observable,
    cut_index: int,
    mitigation: VnCDRMitigation | None = None,
    n_shots: int | None = None,
    epsilon: float | None = None,
    backend=None,
    method: PauliSampling | None = None,
    max_bond_dimension=-1,
    seed: int | None = None,
) -> MitigatedResult:
    """
    Plan, run and mitigate in one call, like
    :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.expectation_at_cut`. On a device,
    call :meth:`VnCDRMitigation.build_plan`, run its circuits, then
    :meth:`MitigationPlan.estimate`.

    Args:
        model: the :class:`~mpstab.evolutors.hqsmpo.HQSMPO`.
        observable: as in :meth:`~mpstab.evolutors.hqsmpo.HQSMPO.plan`.
        cut_index: the split point.
        mitigation: the options, :class:`VnCDRMitigation` defaults if ``None``.
        n_shots: shots of the target plan at each noise level.
        epsilon: target standard error sizing them instead.
        backend: anything with ``execute_circuits(circuits, nshots)``, e.g.
            :class:`~mpstab.qibo_tools.utilized_backends.density_matrix_noise_simulator.DensityMatrixNoiseSimulator`.
        method: the ``"pauli"`` measurement method, with its options.
        max_bond_dimension: bond cap of the tail fold.
        seed: seed of the plan.
    """
    plan = (mitigation or VnCDRMitigation()).build_plan(
        model,
        observable,
        cut_index,
        n_shots=n_shots,
        epsilon=epsilon,
        method=method,
        max_bond_dimension=max_bond_dimension,
        seed=seed,
    )
    return plan.estimate(execute_plan(backend or StatevectorExactSimulator(), plan))


def _pauli_values(surrogate, coefficients) -> dict:
    """``{P: <P>}`` on ``surrogate``'s state, for every string of ``coefficients``."""
    strings = list(coefficients)
    backpropagated = surrogate.stab_engine.backpropagate_many(
        dict.fromkeys(strings, 1.0), surrogate.clifford_circuit
    )
    values = surrogate.tn_engine.pauli_expvals(
        surrogate.original_circuit_mps, [pauli for pauli, _ in backpropagated]
    )
    return {
        string: sign * float(np.real(value))
        for string, (_, sign), value in zip(strings, backpropagated, values)
    }


def _spread(values: np.ndarray, n: int) -> list:
    """
    Indices of ``n`` of ``values`` closest to ``n`` evenly spaced targets
    across their range, each used once.
    """
    targets = np.linspace(values.min(), values.max(), n)
    available = list(range(len(values)))
    chosen = []
    for target in targets:
        best = min(available, key=lambda k: abs(values[k] - target))
        chosen.append(best)
        available.remove(best)
    return chosen


def _eigenstates(basis: str, nqubits: int, eigenvalue: int = 1) -> Circuit:
    """The ``eigenvalue`` eigenstate of the Pauli ``basis`` on every qubit."""
    circuit = Circuit(nqubits)
    for q in range(nqubits):
        if eigenvalue == -1:
            circuit.add(gates.X(q))
        for gate in _EIGENSTATE_PREPARATION[basis]:
            circuit.add(gate(q))
    return circuit


def _calibration(measurements, frequencies, nqubits: int):
    """
    ``{basis: (c, a)}``, the attenuation and offset of each qubit measured in
    each basis, from the mean readings ``c + a`` and ``-c + a`` of its two
    eigenstates; ``None`` without calibration circuits. ``c`` is floored at
    0.05 so that a broken qubit cannot blow up the correction.
    """
    readings: dict = {}
    for measurement, freq in zip(measurements, frequencies):
        if measurement.kind != "calibration":
            continue
        total = sum(freq.values())
        mean = np.zeros(nqubits)
        for bitstring, count in freq.items():
            mean += count * (1 - 2 * np.array([int(b) for b in bitstring]))
        readings[measurement.basis, measurement.eigenvalue] = mean / total
    if not readings:
        return None
    return {
        basis: (
            np.clip((readings[basis, 1] - readings[basis, -1]) / 2, 0.05, None),
            (readings[basis, 1] + readings[basis, -1]) / 2,
        )
        for basis in "XYZ"
    }


def _group_value(group, coefficients, freq, calibration) -> tuple:
    """
    Mean and variance over one setting's shots of ``sum_P c_P prod_q z_q``,
    each reading ``z_q = +-1`` of the support of ``P`` corrected to
    ``(z_q - a_q) / c_q`` with the calibration of its basis (left as is
    without calibration).
    """
    bitstrings = list(freq)
    readings = 1 - 2 * np.array([[int(b) for b in s] for s in bitstrings], dtype=float)
    values = np.zeros(len(bitstrings))
    for member in group.members:
        product = np.full(len(bitstrings), coefficients[member], dtype=float)
        for q, label in enumerate(member):
            if label == "I":
                continue
            if calibration is None:
                product *= readings[:, q]
            else:
                c, a = calibration[label]
                product *= (readings[:, q] - a[q]) / c[q]
        values += product
    counts = list(freq.values())
    mean, per_shot_variance = weighted_mean_and_variance(values, counts)
    shots = sum(counts)
    return mean, (per_shot_variance / shots if shots > 1 else 0.0)


__all__ = [
    "MitigatedResult",
    "MitigationPlan",
    "VnCDRMitigation",
    "mitigated_expectation_at_cut",
]
