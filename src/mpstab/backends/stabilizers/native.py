from dataclasses import dataclass

from qibo.models import Circuit

from mpstab.backends.stabilizers.abstract import StabilizersEngine
from mpstab.evolutors.stabilizer import tableaus
from mpstab.evolutors.stabilizer.pauli_string import Pauli
from mpstab.evolutors.utils import gate2tableau


@dataclass
class NativeStabilizersEngine(StabilizersEngine):

    def backpropagate(self, observable: str, clifford_circuit: Circuit) -> str:
        """
        Process a given Pauli string by applying the inverse of the circuit
        gates in reverse order (Heisenberg picture).
        """
        # Start with the observable as a Pauli object
        propagator = Pauli(observable)

        # Apply gates in reverse order for Heisenberg backpropagation
        for gate in reversed(clifford_circuit.queue):

            # Invert the gate logic (dagger)
            inverted_gate = gate.dagger()

            # Retrieve the corresponding tableau class
            tableau_name = gate2tableau.get(inverted_gate.name.lower())
            if tableau_name:
                tab_cls = getattr(tableaus, tableau_name)

                # Instantiate the tableau with qubits and optional parameters
                if len(inverted_gate.parameters) != 0:
                    # Handles parameterized gates like rotations
                    angle = inverted_gate.parameters[0]
                    tab_obj = tab_cls(*inverted_gate.qubits, angle=angle)
                else:
                    # Handles standard Clifford gates (H, CNOT, S, etc.)
                    tab_obj = tab_cls(*inverted_gate.qubits)

                # Apply the transformation rules to the Pauli string
                # This follows your logic: propagator.apply(tableau_instance)
                propagator.apply(tab_obj)

        return propagator.to_string()
