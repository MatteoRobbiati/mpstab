"""
Checks for the parameter annotations of the ansätze: every name listed in
``magic_parameters`` or ``entanglement_parameters`` is a constructor field, and
the magic knobs of the kicked Ising family really control the magic gates.
"""

import dataclasses
import inspect

import numpy as np
import pytest
from qibo import Circuit, gates

from mpstab.models import ansatze
from mpstab.models.ansatze import (
    Ansatz,
    ParametricKickedIsing,
    RampedKickedIsing,
    TranspiledAnsatz,
)
from mpstab.models.utils import hardware_compatible_circuit


def _ansatz_classes():
    return [
        cls
        for _, cls in inspect.getmembers(ansatze, inspect.isclass)
        if issubclass(cls, Ansatz) and cls is not Ansatz and not cls.__name__.startswith("_")
    ]


@pytest.mark.parametrize("cls", _ansatz_classes(), ids=lambda c: c.__name__)
def test_listed_parameters_are_fields(cls):
    fields = {f.name for f in dataclasses.fields(cls) if f.init}
    for name in cls.magic_parameters + cls.entanglement_parameters:
        assert name in fields, f"{cls.__name__}.{name} is not a constructor field"


def _magic_gates(ansatz):
    return [g for g in ansatz.circuit.queue if not g.clifford]


@pytest.mark.parametrize("cls", [ParametricKickedIsing, RampedKickedIsing])
def test_zero_magic_parameters_give_a_clifford_circuit(cls):
    """With every magic knob at zero only the Clifford skeleton is left."""
    ansatz = cls(nqubits=5, n_steps=3, delta_h=0.0, delta_j=0.0, delta_z=0.0)
    assert _magic_gates(ansatz) == []


def test_entanglement_parameters_do_not_change_the_magic():
    """Changing the Clifford skeleton leaves the magic gates untouched."""
    reference = ParametricKickedIsing(nqubits=5, n_steps=2, delta_h=0.2)
    other = ParametricKickedIsing(
        nqubits=5, n_steps=2, delta_h=0.2, clifford_h=np.pi, clifford_j=0.0
    )
    params = lambda a: [(g.name, g.parameters) for g in _magic_gates(a)]
    assert params(reference) == params(other)


def test_transpiled_ansatz_uses_the_given_native_gates():
    circuit = Circuit(2)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.RZ(1, theta=0.3))
    native = [gates.GPI2, gates.RZ, gates.Z, gates.CNOT]
    ansatz = TranspiledAnsatz(original_circuit=circuit, native_gates=native)
    expected = hardware_compatible_circuit(circuit, native)
    assert [g.name for g in ansatz.circuit.queue] == [g.name for g in expected.queue]
