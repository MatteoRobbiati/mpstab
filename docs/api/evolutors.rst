Evolutors
=========

Evolutors are the high-level managers of quantum circuit simulation in MPSTAB. They orchestrate the flow through a quantum circuit, deciding when to use stabilizer methods versus tensor network methods for maximum efficiency.

**Key Concept**: Evolutors handle the temporal evolution of quantum states through circuit gates, managing the hybrid representation.

HSMPO: Hybrid Stabilizer MPO
-----------------------------

The main HSMPO class combines stabilizer states with tensor network representations.

.. automodule:: mpstab.evolutors.hsmpo
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods**:

- ``__init__(ansatz, max_bond_dimension, initial_state)``: Initialize the simulator
- ``expectation(observable, return_fidelity=False)``: Compute expectation value of a Pauli observable
- ``truncation_fidelity()``: Compute truncation fidelity of current state
- ``set_engines()``: Configure stabilizer and tensor network engines

**Usage Example**::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    # Create a circuit
    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    # Simulate with HSMPO
    simulator = HSMPO(ansatz=circuit, max_bond_dimension=32)
    result, fidelity = simulator.expectation("Z" * 5, return_fidelity=True)
    print(f"Expectation value: {result}")
    print(f"Fidelity: {fidelity}")

Utilities
---------

Helper functions for evolutor operations.

.. automodule:: mpstab.evolutors.utils
   :members:
   :undoc-members:
   :show-inheritance:

Stabilizer Evolution Tools
--------------------------

Low-level stabilizer state management and evolution.

.. automodule:: mpstab.evolutors.stabilizer
   :members:
   :undoc-members:
   :show-inheritance:

Tensor Network Evolution Tools
------------------------------

Low-level tensor network (MPS/MPO) management and operations.

.. automodule:: mpstab.evolutors.tensor_network.tensor_network
   :members:
   :undoc-members:
   :show-inheritance:

For more practical examples, see:

- :doc:`../guides/quickstart`
- :doc:`../guides/fidelity_and_approximation`
- :doc:`../examples/introduction`
