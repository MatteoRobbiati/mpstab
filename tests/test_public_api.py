"""Tests for the flat public API: the objects of a typical workflow from ``mpstab``."""

import importlib

import pytest

import mpstab

HOMES = {
    "mpstab.evolutors.hsmpo": ["HSMPO"],
    "mpstab.evolutors.hqsmpo": ["HQSMPO"],
    "mpstab.models.noise": [
        "AmplitudeDampingError",
        "CoherentError",
        "DepolarizingError",
        "KrausError",
        "NoiseModel",
        "PauliError",
        "ReadoutError",
        "StatePreparationError",
        "ThermalRelaxationError",
    ],
    "mpstab.models.error_mitigation": [
        "VnCDRMitigation",
        "cdr",
        "fold_circuit",
        "mitigated_expectation_at_cut",
        "noisy_value",
        "training_circuits",
        "vncdr",
    ],
    "mpstab.models.topology": ["Topology"],
    "mpstab.qpu.execution.backend": ["run_plan"],
    "mpstab.qibo_tools": ["DensityMatrixNoiseSimulator", "StatevectorExactSimulator"],
}
NAMES = [(home, name) for home, names in HOMES.items() for name in names]


@pytest.mark.parametrize("home, name", NAMES, ids=[name for _, name in NAMES])
def test_name_is_exported_from_mpstab(home, name):
    assert name in mpstab.__all__
    assert getattr(mpstab, name) is getattr(importlib.import_module(home), name)


def test_every_exported_name_exists():
    for name in mpstab.__all__:
        assert hasattr(mpstab, name), name
