"""
Hybrid stabilizer-MPO simulation of quantum circuits.

The objects of a typical workflow are importable from here: the simulators
(:class:`HSMPO`, :class:`HQSMPO`), noise models and their errors, error
mitigation, and the backends running measurement plans. Ansatze are in
:mod:`mpstab.models.ansatze`, measurement and synthesis options in
:mod:`mpstab.qpu`, engines in :mod:`mpstab.engines`.
"""

from mpstab.evolutors.hqsmpo import HQSMPO
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.hamiltonians import pauli_string_to_hamiltonian, pauli_terms
from mpstab.models.error_mitigation import (
    VnCDRMitigation,
    cdr,
    fold_circuit,
    mitigated_expectation_at_cut,
    noisy_value,
    training_circuits,
    vncdr,
)
from mpstab.models.noise import (
    AmplitudeDampingError,
    CoherentError,
    DepolarizingError,
    KrausError,
    NoiseModel,
    PauliError,
    ReadoutError,
    StatePreparationError,
    ThermalRelaxationError,
)
from mpstab.qibo_tools import (
    DensityMatrixNoiseSimulator,
    MetaBackend,
    StatevectorExactSimulator,
)
from mpstab.qpu.execution import execute_plan

__all__ = [
    # simulators
    "HSMPO",
    "HQSMPO",
    # observables
    "pauli_string_to_hamiltonian",
    "pauli_terms",
    # noise
    "NoiseModel",
    "AmplitudeDampingError",
    "CoherentError",
    "DepolarizingError",
    "KrausError",
    "PauliError",
    "ReadoutError",
    "StatePreparationError",
    "ThermalRelaxationError",
    # error mitigation
    "VnCDRMitigation",
    "cdr",
    "fold_circuit",
    "mitigated_expectation_at_cut",
    "noisy_value",
    "training_circuits",
    "vncdr",
    # running measurement plans
    "execute_plan",
    "DensityMatrixNoiseSimulator",
    "StatevectorExactSimulator",
    # mpstab as a qibo backend
    "MetaBackend",
]
