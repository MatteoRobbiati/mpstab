"""Home-made stabilizers backend.""" """Home-made stabilizers engine."""

from dataclasses import dataclass

from qibo import Circuit

from mpstab.backends.stabilizers.abstract import StabilizersEngine
from mpstab.evolutors.stabilizer import tableaus
from mpstab.evolutors.stabilizer.pauli_string import Pauli
from mpstab.evolutors.utils import gate2tableau


@dataclass
class NativeStabilizersEngine(StabilizersEngine):

    def backpropagate(self, observable: str, circuit: Circuit) -> str:
        pauli = Pauli(observable)
        # Apply gates in reverse order for Heisenberg backpropagation
        for gate in reversed(circuit.queue):
            tableau_name = gate2tableau.get(gate.name.lower())
            if tableau_name:
                tab_cls = getattr(tableaus, tableau_name)
                # Note: This logic assumes a method to apply tableau to Pauli
                pauli = tab_cls(*gate.qubits).apply(pauli)
        return pauli.to_string()
