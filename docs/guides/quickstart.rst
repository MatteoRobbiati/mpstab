Quick Start Guide
=================


Your First MPSTAB Program
~~~~~~~~~~~~~~~~~~~~~~~~~

Here's the minimal code to simulate a quantum circuit with MPSTAB::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    # Create a simple quantum circuit
    circuit = Circuit(5)
    for q in range(5):
        circuit.add(gates.H(q))
        circuit.add(gates.RY(q, theta=0.5))
    circuit.add(gates.CNOT(0, 1))

    # Create HSMPO simulator
    simulator = HSMPO(ansatz=circuit)

    # Compute expectation value of observable
    result = simulator.expectation("ZIIIZ")
    print(f"<ZIIIZ> = {result}")


Simulate with Bond Dimension Limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Limit MPO bond dimension for faster computation::

    simulator = HSMPO(ansatz=circuit, max_bond_dimension=16)
    result = simulator.expectation("ZIIIZ")

    # Check fidelity lower bound
    print(f"Fidelity: {simulator.fidelity_lower_bound}")

Use an Ansatz
~~~~~~~~~~~~~~

Pre-built quantum circuit patterns::

    from mpstab.models.ansatze import HardwareEfficient

    # Use a pre-built quantum circuit pattern
    ansatz = HardwareEfficient(nqubits=5, nlayers=3)
    simulator = HSMPO(ansatz=ansatz)
    result = simulator.expectation("ZZZZZ")

Measure Complex Observables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Beyond simple Pauli strings, you can measure complex observables using Qibo's symbolic Hamiltonians::

    from qibo import hamiltonians
    from qibo.symbols import X, Y, Z

    # Define a multi-term observable
    # Example: 0.5 * X(0)*Z(1) + 0.3 * Y(1)*Y(2)
    observable = 0.5 * X(0) * Z(1) + 0.3 * Y(1) * Y(2)

    # Create a symbolic Hamiltonian
    ham = hamiltonians.SymbolicHamiltonian(form=observable)

    # Use HSMPO to compute the expectation value
    simulator = HSMPO(ansatz=circuit)
    result = simulator.expectation(ham)
    print(f"<H> = {result}")

Build observables from individual Pauli operators::

    from qibo.symbols import X, Y, Z, I

    # Construct from qubit-by-qubit: X ⊗ Y ⊗ Z on qubits 0, 1, 2
    pauli_string = X(0) * Y(1) * Z(2)
    ham = hamiltonians.SymbolicHamiltonian(form=pauli_string)

    # Multi-term Hamiltonian with coefficients
    H = 2.0 * Z(0) * Z(1) + 1.5 * X(1) * X(2) + 0.5 * Z(2)
    ham = hamiltonians.SymbolicHamiltonian(form=H)

    exp_val = simulator.expectation(ham)

Simple Pauli strings still work::

    # Quick measurement of a single Pauli string
    result = simulator.expectation("XYZIX")


Key Concepts
------------

**Bond Dimension**
    Controls MPO accuracy vs efficiency trade-off

**Fidelity Lower Bound**
    Indicates approximation error due to truncation

**Clifford Circuits**
    Simulated exactly and efficiently, even for large systems

**Observable**
    Pauli string defining the measurement (e.g., "ZXIY")

Next Steps
----------

- Read the :doc:`../examples/introduction` example for more detailed explanations
- Explore :doc:`working_with_engines` for backend options
- Learn about :doc:`using_ansatze` for circuit design
- Check the :doc:`../api/evolutors` for complete documentation
