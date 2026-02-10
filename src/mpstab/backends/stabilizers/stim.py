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
        Evolve `observable` using stim while correctly handling the sign and string padding.
        """
        # 1. Determine total number of qubits from the observable string
        n_obs = len(observable)

        # 2. Convert Qibo circuit to Stim
        stim_circuit = self._qibo_to_stim(clifford_circuit)

        # 3. Use TableauSimulator to get the inverse transformation
        sim = stim.TableauSimulator()

        # Pad with identity on the highest qubit to ensure tableau matches observable size
        padded_circuit = stim.Circuit()
        padded_circuit.append("I", [n_obs - 1])
        padded_circuit += stim_circuit
        sim.do(padded_circuit)

        # current_inverse_tableau returns the inverse tableau U†
        inv_tableau = sim.current_inverse_tableau()

        # 4. Transform the PauliString manually to handle polyfill environments
        pauli_to_evolve = stim.PauliString(observable)
        result_pauli = stim.PauliString(n_obs)
        for i in range(n_obs):
            p_val = pauli_to_evolve[i]
            if p_val == 1:  # X
                result_pauli *= inv_tableau.x_output(i)
            elif p_val == 2:  # Y
                result_pauli *= inv_tableau.y_output(i)
            elif p_val == 3:  # Z
                result_pauli *= inv_tableau.z_output(i)

        # Apply original sign of the input Pauli string
        if pauli_to_evolve.sign == -1:
            result_pauli *= -1

        # 5. FIX: Adapt the Stim string format for mpstab
        # stim str() returns "+Z_X_". mpstab needs "ZIXI" or "+ZIXI"
        # Since mpstab's string_to_xz fails on '+', we strip it.
        # mpstab.Pauli constructor handles phases separately from the raw XZ string.
        res_str = str(result_pauli).replace("_", "I")

        # If the string has a sign, mpstab.Pauli's __init__ will parse it correctly
        # only if we don't break string_to_xz inside it.
        # mpstab.Pauli handles signs by checking description[0] in phase_to_xz.keys()c
        return res_str[1:]

    def _qibo_to_stim(self, circuit: Circuit) -> stim.Circuit:
        """Helper to convert a Qibo circuit into a Stim circuit."""
        stim_c = stim.Circuit()

        def is_approx(val, target, atol=1e-5):
            return np.isclose(
                val % (2 * np.pi), target % (2 * np.pi), atol=atol
            ) or np.isclose(
                val % (2 * np.pi), (target % (2 * np.pi)) + 2 * np.pi, atol=atol
            )

        for g in circuit.queue:
            q = g.qubits
            name = g.name.lower()

            if name == "h":
                stim_c.append("H", q)
            elif name in ["x", "y", "z"]:
                stim_c.append(name.upper(), q)
            elif name in ["cx", "cnot"]:
                stim_c.append("CNOT", q)
            elif name == "cz":
                stim_c.append("CZ", q)
            elif name == "swap":
                stim_c.append("SWAP", q)
            elif name == "s":
                stim_c.append("S", q)
            elif name == "sdg":
                stim_c.append("S_DAG", q)
            elif name in ["rx", "ry", "rz"]:
                theta = g.parameters[0]
                axis = name[1].upper()
                if is_approx(theta, 0):
                    continue
                elif is_approx(theta, np.pi) or is_approx(theta, -np.pi):
                    stim_c.append(axis, q)
                elif is_approx(theta, np.pi / 2):
                    stim_c.append(f"SQRT_{axis}", q)
                elif is_approx(theta, -np.pi / 2):
                    stim_c.append(f"SQRT_{axis}_DAG", q)
                else:
                    raise ValueError(f"Gate {g} is not Clifford.")
        return stim_c
