import random
from typing import Callable, List, Tuple

import stim
from qibo import Circuit, gates

gate2generator = {
    "rx": "X",
    "ry": "Y",
    "rz": "Z",
    "t": "Z",
}

gate2tableau = {
    "cx": "CNOT",
    "h": "H",
    "s": "S",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "swap": "SWAP",
    "cz": "CZ",
    "rz": "RZ",
    "ry": "RY",
    "rx": "RX",
    "gpi2": "GPI2",
    "sdg": "Sdg",
}

one_qubit_cliff = "HXYZ"


def sample_random_pauli_gate(qubit):
    """
    Sample a random one-qubit gate applyed to a given qubit.
    """
    random_letter = random.choice(one_qubit_cliff)
    return getattr(gates, random_letter)(q=qubit)


def _link_to_dummy(tn, dummy, tensor, tensor_direction, edge_id="v_link"):

    T, d, e, data = list(tn.tensornet.in_edges(dummy, data=True, keys=True))[0]
    dummy_direction = data["directions"][0]
    tn.remove_edge(T, d, e)
    tn.add_edge(T, tensor, edge_id, (dummy_direction, tensor_direction))
    tn.tensornet.remove_node(dummy)


def validate_pauli_observable(observable: str, nqubits: int) -> None:
    """
    Validate that a Pauli observable string is well-formed.

    Args:
        observable: Pauli observable string (e.g., "ZZZZZ", "XYZIX")
        nqubits: Number of qubits in the system

    Raises:
        ValueError: If observable contains invalid characters or has incorrect length
    """
    # Validate observable string contains only Pauli operators
    valid_paulis = set("IXYZ")
    invalid_chars = set(observable) - valid_paulis
    if invalid_chars:
        raise ValueError(
            f"Observable string contains invalid characters: {invalid_chars}. "
            f"Observable strings should only contain Pauli operators: I, X, Y, Z. "
            f"Do not include signs, coefficients, or other characters. "
            f"Examples: 'ZZZZZ' or 'XYZIX', not '2*ZZZZZ' or '-ZZ'."
        )

    # Validate observable string length matches number of qubits
    if len(observable) != nqubits:
        raise ValueError(
            f"Observable string length ({len(observable)}) does not match "
            f"the number of qubits ({nqubits}). "
            f"Expected a Pauli string of length {nqubits}, "
            f"e.g., '{'Z'*nqubits}' for measuring Z operators on all qubits."
        )


def dressed_rotations(
    nqubits: int,
    stab_engine,
    magic_gates: List[Tuple[int, object]],
    clifford_circuit: Circuit,
    gate_angle: Callable,
) -> List[Tuple[str, float]]:
    """
    Return every magic gate's dressed ``(generator, signed_angle)``, in
    circuit order: each magic gate's generator backpropagated through the
    Clifford sub-circuit preceding it (Heisenberg picture), scaled by its
    rotation angle.

    Single pass: the Clifford circuit is simulated ONCE into a running
    ``stim.TableauSimulator``, and each magic gate's generator is conjugated
    at its breakpoint using the simulator's current inverse tableau -- instead
    of re-simulating the Clifford prefix from scratch for every magic gate.

    Requires a ``StimEngine`` (raised as ``NotImplementedError`` otherwise,
    only when there is actual magic-gate work to do -- a Clifford-only
    circuit has no dressed rotations regardless of the stabilizers engine in
    use).
    """
    if not magic_gates:
        return []

    from mpstab.engines.stabilizers.stim import StimEngine

    if not isinstance(stab_engine, StimEngine):
        raise NotImplementedError(
            "Fast dressed-rotation extraction requires StimEngine (this "
            f"circuit has {len(magic_gates)} non-Clifford magic gate(s)). Call "
            "set_engines(stab_engine=StimEngine()) to enable it."
        )

    sim = stim.TableauSimulator()
    sim.do(stim.Circuit(f"I {nqubits - 1}"))

    def _conjugate(gate):
        gen = "".join(
            gate2generator[gate.name] if q in gate.target_qubits else "I"
            for q in range(nqubits)
        )
        inv = sim.current_inverse_tableau()
        p = stim.PauliString(gen)
        res = stim.PauliString(nqubits)
        for i in range(nqubits):
            v = p[i]
            if v == 1:
                res *= inv.x_output(i)
            elif v == 2:
                res *= inv.y_output(i)
            elif v == 3:
                res *= inv.z_output(i)
        s = str(res)
        sign = -1 if s.startswith("-") else 1
        return s.replace("_", "I").lstrip("+-"), sign

    rotations = []
    mi, nmagic, count = 0, len(magic_gates), 0
    while mi < nmagic and magic_gates[mi][0] == count:
        g = magic_gates[mi][1]
        gen, sign = _conjugate(g)
        rotations.append((gen, gate_angle(g) * sign))
        mi += 1
    for gate in clifford_circuit.queue:
        one = Circuit(nqubits)
        one.add(gate)
        sim.do(stab_engine._qibo_to_stim(one))
        count += 1
        while mi < nmagic and magic_gates[mi][0] == count:
            g = magic_gates[mi][1]
            gen, sign = _conjugate(g)
            rotations.append((gen, gate_angle(g) * sign))
            mi += 1
    return rotations
