"""Stim-based stabilizers engine."""

from dataclasses import dataclass

import numpy as np
import stim
from qibo import Circuit

from mpstab.engines.stabilizers.abstract import StabilizersEngine


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
        # mpstab.Pauli handles signs by checking description[0] in phase_to_xz.keys()
        return res_str[1:], -1 if res_str.startswith("-") else 1

    def fold_pauli_through_tableau(
        self, pauli_str: str, tableau: stim.Tableau, sign: float = 1.0
    ):
        """
        Reabsorb a Clifford ``tableau`` into a Pauli observable: returns
        ``M = tableau^dag . P . tableau`` as a (signed) Pauli, so that applying
        ``tableau`` last to a state ``psi`` gives
        ``<psi|tableau^dag . P . tableau|psi> = <psi|M|psi>``.

        Used to exactly reabsorb a pure-Clifford residual (e.g. the tail
        produced by
        :func:`mpstab.evolutors.quantum_hardware.rustiq_synthesis.build_head_and_residual`)
        into an observable, with no tensor-network truncation involved.

        Args:
            pauli_str: Pauli string (qubit-0-leftmost), e.g. "XZIZ".
            tableau: The Clifford tableau to reabsorb (applied last).
            sign: +/-1 prefactor on the input Pauli.

        Returns:
            (pauli_str, sign): the folded observable (qubit-0-leftmost).
        """
        p = stim.PauliString(pauli_str)
        if sign < 0:
            p = -p
        out = tableau.inverse()(p)
        out_sign = float(out.sign.real if hasattr(out.sign, "real") else out.sign)
        label = str(out)
        label = (label[1:] if label and label[0] in "+-" else label).replace("_", "I")
        return label, out_sign

    def _qibo_to_stim(self, circuit: Circuit) -> stim.Circuit:
        """Helper to convert a Qibo circuit into a Stim circuit."""
        stim_c = stim.Circuit()

        def is_approx(val, target, atol=1e-5):
            # Use regular numpy operations (which work with JAX arrays too)
            val_norm = val % (2 * np.pi)
            target_norm = target % (2 * np.pi)
            diff1 = np.abs(val_norm - target_norm)
            diff2 = np.abs(val_norm - (target_norm + 2 * np.pi))
            return (diff1 < atol) or (diff2 < atol)

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
            elif name == "gpi2":
                # GPI2(phi) is a pi/2 rotation about the axis (cos phi, sin phi, 0).
                # It is Clifford only at multiples of pi/2, mapping to stim as:
                #   0 -> SQRT_X, pi/2 -> SQRT_Y, pi -> SQRT_X_DAG, 3pi/2 -> SQRT_Y_DAG
                # (verified via U P U^dag against stim's named-gate tableaux).
                phi = g.parameters[0]
                if is_approx(phi, 0):
                    stim_c.append("SQRT_X", q)
                elif is_approx(phi, np.pi / 2):
                    stim_c.append("SQRT_Y", q)
                elif is_approx(phi, np.pi) or is_approx(phi, -np.pi):
                    stim_c.append("SQRT_X_DAG", q)
                elif is_approx(phi, -np.pi / 2) or is_approx(phi, 3 * np.pi / 2):
                    stim_c.append("SQRT_Y_DAG", q)
                else:
                    raise ValueError(f"Gate {g} is not Clifford.")
            elif name in ["rx", "ry", "rz"]:
                theta = g.parameters[0]
                axis = name[1].upper()
                try:
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
                except (TypeError, ValueError) as e:
                    # During JAX traced execution, conditionals with traced values will fail
                    # In this case, just skip the gate and let the circuit continue
                    if "not Clifford" in str(e):
                        raise
                    # Otherwise silently handle the traced case
            elif name in ["measure", "id", "barrier"]:
                # Non-unitary / no-op on the Pauli frame: nothing to backpropagate.
                continue
            else:
                # Fail loudly rather than silently dropping a gate from the
                # Clifford backpropagation (a silent skip yields wrong results).
                raise ValueError(
                    f"Gate '{name}' is not supported by the stim backpropagation "
                    "engine. Ensure the Clifford part uses supported gates."
                )
        return stim_c
