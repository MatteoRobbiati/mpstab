"""Tests for the deprecated import paths kept as aliases of the moved modules."""

import importlib

import pytest

from mpstab import qpu
from mpstab.qibo_tools import StatevectorExactSimulator
from mpstab.qpu import execution
from mpstab.qpu.execution import execute

# The first import of an alias warns outside pytest.warns; each test checks it.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
MOVED_MODULES = [
    (
        "mpstab.noise",
        "mpstab.models.noise",
        [
            "AmplitudeDampingError",
            "CoherentError",
            "DepolarizingError",
            "KrausError",
            "NoiseModel",
            "NoisyExpectation",
            "PauliError",
            "ReadoutError",
            "StatePreparationError",
            "ThermalRelaxationError",
        ],
    ),
    (
        "mpstab.noise.compile",
        "mpstab.models.noise.compile",
        ["CompiledNoise", "compile_noise"],
    ),
    (
        "mpstab.noise.events",
        "mpstab.models.noise.events",
        ["CoherentRotation", "KrausChannel", "NoiseEvent", "PauliChannel"],
    ),
    (
        "mpstab.noise.mc_noise_estimator",
        "mpstab.models.noise.mc_noise_estimator",
        ["NoisyExpectation", "noisy_expectation"],
    ),
    (
        "mpstab.noise.model",
        "mpstab.models.noise.model",
        ["NoiseModel", "DepolarizingError"],
    ),
    (
        "mpstab.qibo_backend",
        "mpstab.qibo_tools.provided_backends",
        ["MPStabBackend", "MetaBackend"],
    ),
    (
        "mpstab.qibo_backend.mpstab",
        "mpstab.qibo_tools.provided_backends.mpstab_backend",
        ["MPStabBackend"],
    ),
]


@pytest.mark.parametrize(
    "old_path, new_path, names", MOVED_MODULES, ids=[m[0] for m in MOVED_MODULES]
)
def test_moved_module_alias(old_path, new_path, names):
    new_module = importlib.import_module(new_path)
    alias = importlib.import_module(old_path)
    with pytest.warns(DeprecationWarning, match=new_path):
        # A module runs once per process: reload it to see its warning again.
        importlib.reload(alias)
    for name in names:
        assert getattr(alias, name) is getattr(new_module, name)


def test_simulator_module_alias():
    alias = importlib.import_module("mpstab.qpu.execution.simulator")
    with pytest.warns(DeprecationWarning, match="StatevectorExactSimulator"):
        importlib.reload(alias)
    assert alias.QiboSimulator is StatevectorExactSimulator
    assert alias.execute_plan is execute.execute_plan


@pytest.mark.parametrize("package", [qpu, execution], ids=lambda p: p.__name__)
def test_qibo_simulator_alias(package):
    with pytest.warns(DeprecationWarning, match="StatevectorExactSimulator"):
        assert package.QiboSimulator is StatevectorExactSimulator


def test_unknown_attribute_still_raises():
    with pytest.raises(AttributeError):
        qpu.NotAnAttribute
