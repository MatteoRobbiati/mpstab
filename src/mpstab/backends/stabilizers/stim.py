"""Stim-based stabilizers engine."""

from dataclasses import dataclass

import numpy as np
import stim
from qibo import Circuit

from mpstab.backends.stabilizers.abstract import StabilizersEngine


@dataclass
class StimEngine(StabilizersEngine):
    """Stabilizers engine powered by Stim."""

    def backpropagate(self, observable: str, clifford_circuit: Circuit) -> str:
        """
        Evolve `observable` using the logic from the benchmark script.
        We use a TableauSimulator to find the equivalent backpropagated Pauli string.
        """
        # 1. Convert Qibo Clifford circuit to Stim (using your script's gate logic)
        stim_circuit = self._qibo_to_stim(clifford_circuit)

        # 2. Use TableauSimulator as done in the script
        sim = stim.TableauSimulator()
        sim.do(stim_circuit)

        # 3. Compute the conjugate of the Pauli string manually.
        # Since 'tableau * pauli_string' is failing, we use the
        # Tableau to transform the Pauli components.
        pauli_to_evolve = stim.PauliString(observable)

        # In Stim, the backpropagated Pauli U† O U is equivalent to
        # rotating the Pauli by the inverse of the circuit's tableau.
        # We use current_inverse_tableau() which is U†.
        inv_tableau = sim.current_inverse_tableau()

        # Manual conjugation: transform the PauliString using the inverse tableau
        # If 'conjugated_by' failed previously, we use the tableau to
        # map each individual Pauli component.
        n = len(pauli_to_evolve)
        result_pauli = stim.PauliString(n)

        # We find the new Pauli by looking at how the inverse tableau maps the original
        # This is a robust fallback for the __mul__ and conjugated_by errors.
        for i in range(n):
            p_val = pauli_to_evolve[i]
            if p_val == 1:  # X
                result_pauli *= inv_tableau.x_output(i)
            elif p_val == 2:  # Y
                result_pauli *= inv_tableau.y_output(i)
            elif p_val == 3:  # Z
                result_pauli *= inv_tableau.z_output(i)

        # Account for the original sign of the Pauli string
        if pauli_to_evolve.sign == -1:
            result_pauli *= -1

        return str(result_pauli)

    def _qibo_to_stim(self, circuit: Circuit) -> stim.Circuit:
        """Helper to convert a Qibo circuit into a Stim circuit (from script logic)."""
        stim_c = stim.Circuit()

        def is_approx(val, target, atol=1e-5):
            return np.isclose(
                val % (2 * np.pi), target % (2 * np.pi), atol=atol
            ) or np.isclose(
                val % (2 * np.pi), (target % (2 * np.pi)) + 2 * np.pi, atol=atol
            )

        for g in circuit.queue:
            q_indices = g.qubits
            name = g.name.lower()

            if name == "h":
                stim_c.append("H", q_indices)
            elif name in ["x", "y", "z"]:
                stim_c.append(name.upper(), q_indices)
            elif name in ["cx", "cnot"]:
                stim_c.append("CNOT", q_indices)
            elif name == "cz":
                stim_c.append("CZ", q_indices)
            elif name == "swap":
                stim_c.append("SWAP", q_indices)
            elif name == "s":
                stim_c.append("S", q_indices)
            elif name == "sdg":
                stim_c.append("S_DAG", q_indices)
            elif name in ["rx", "ry", "rz"]:
                theta = g.parameters[0]
                axis = name[1].upper()
                if is_approx(theta, 0):
                    continue
                elif is_approx(theta, np.pi) or is_approx(theta, -np.pi):
                    stim_c.append(axis, q_indices)
                elif is_approx(theta, np.pi / 2):
                    stim_c.append(f"SQRT_{axis}", q_indices)
                elif is_approx(theta, -np.pi / 2):
                    stim_c.append(f"SQRT_{axis}_DAG", q_indices)
                else:
                    raise ValueError(f"Gate {g} is not Clifford.")
        return stim_c
