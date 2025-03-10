"""Simple test to check whether the stabilizer simulator is working."""

from qibo import Circuit, gates, hamiltonians, symbols

from tncdr.evolutors.stabilizer.pauli_string import Pauli
from tncdr.evolutors.stabilizer import tableaus
from tncdr.evolutors.utils import gate2tableau

nqubits = 4

# Circuit with only Clifford gates
stab_circ = Circuit(nqubits)
[stab_circ.add(gates.H(q)) for q in range(nqubits)]
[stab_circ.add(gates.CNOT(q % nqubits, (q + 1) % nqubits)) for q in range(nqubits)]
stab_circ.add(gates.Z(2))
# [stab_circ.add(gates.H(q)) for q in range(nqubits)]


# Empty circuit
init_circ = Circuit(nqubits)
[init_circ.add(gates.RY(q=q, theta=0.2 + q)) for q in range(nqubits)]
[init_circ.add(gates.RZ(q=q, theta=0.4 + q)) for q in range(nqubits)]

# Full circ
circ = init_circ + stab_circ

init_circ.draw()
print("\n\n")
circ.draw()

# Pauli string
obs_str = "ZZZY"
obs_form = symbols.Z(0) * symbols.Z(1)  * symbols.Z(2) * symbols.Y(3)
ham = hamiltonians.SymbolicHamiltonian(form=obs_form)

p = Pauli(obs_str)
for gate in stab_circ.invert().queue:
    if len(gate.parameters) != 0:
        params = {"angle": gate.parameters[0]}
    else:
        params = {}
    print(gate.name)
    p.apply(getattr(tableaus, gate2tableau[gate.name])(*gate.qubits, **params))

p = p.__repr__()
print(p)

if p[0] == "-":
    p = p[1:]
    sign = -1.
else:
    sign = 1.

new_obs_form = sign
for i, pauli in enumerate(p):
    new_obs_form *= getattr(symbols, pauli)(i)

print(new_obs_form)

new_ham = hamiltonians.SymbolicHamiltonian(new_obs_form)

print(f"Circuit {ham.expectation(circ().state())}")
print(f"Empty circ on our string: {new_ham.expectation(empty_circ().state())}")