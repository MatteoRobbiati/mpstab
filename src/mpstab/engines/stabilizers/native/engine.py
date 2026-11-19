"""Pure-Python stabilizers engine."""

from qibo.models import Circuit

from mpstab.engines.stabilizers.abstract import StabilizersEngine
from mpstab.engines.stabilizers.native import tableaus
from mpstab.engines.stabilizers.native.pauli_string import Pauli

#: qibo gate names mapped to the tableau class in
#: :mod:`~mpstab.engines.stabilizers.native.tableaus` that implements them.
GATE_TABLEAUS = {
    "cx": "CNOT",
    "h": "H",
    "s": "S",
    "sdg": "Sdg",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "swap": "SWAP",
    "cz": "CZ",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "gpi2": "GPI2",
}


class NativeStabilizersEngine(StabilizersEngine):
    """
    Clifford backpropagation by composing per-gate tableaus.

    A pure-Python example of the :class:`StabilizersEngine` interface, written
    to be read rather than to be fast: use :class:`~mpstab.engines.StimEngine`
    for real workloads.
    """

    def backpropagate(self, observable: str, clifford_circuit: Circuit):
        """
        Evolve ``observable`` back through ``clifford_circuit``, applying each
        gate's inverse in reverse order.

        Returns:
            ``(pauli_string, sign)`` for ``U^dag . O . U``, with ``U`` the circuit.
        """
        propagator = Pauli(observable)

        for gate in reversed(clifford_circuit.queue):
            inverted = gate.dagger()
            tableau_name = GATE_TABLEAUS.get(inverted.name.lower())
            if tableau_name is None:
                continue

            tableau_class = getattr(tableaus, tableau_name)
            if inverted.parameters:
                tableau = tableau_class(*inverted.qubits, angle=inverted.parameters[0])
            else:
                tableau = tableau_class(*inverted.qubits)
            propagator.apply(tableau)

        result = propagator.to_string()
        if result[0] not in "IXYZ":  # a leading sign or phase character
            return result[1:], -1 if result.startswith("-") else 1
        return result, 1.0

    def backpropagate_at(self, observables, breakpoints, clifford_circuit: Circuit):
        """
        Backpropagate each observable through its own prefix of
        ``clifford_circuit``, one :meth:`backpropagate` per observable.

        Each prefix is replayed from scratch, so the cost grows with the
        circuit length times the number of observables: the plain version of
        what a tableau simulator does in a single pass.
        """
        results = []
        for observable, breakpoint_index in zip(observables, breakpoints):
            prefix = Circuit(clifford_circuit.nqubits)
            for gate in clifford_circuit.queue[:breakpoint_index]:
                prefix.add(gate)
            results.append(self.backpropagate(observable, prefix))
        return results
