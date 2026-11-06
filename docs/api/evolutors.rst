Evolutors
=========

Evolutors are the high-level managers of a simulation. They partition a circuit
into a Clifford part and the magic gates that resist it, then drive the
stabilizer and tensor-network engines through the result.

HSMPO: Hybrid Stabilizer MPO
-----------------------------

The Clifford part is absorbed into the observable and the magic gates are applied
to an MPS as dressed Pauli rotations.

.. automodule:: mpstab.evolutors.hsmpo
   :members:
   :undoc-members:
   :show-inheritance:

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

HSynthSMPO: Head/Tail Split
---------------------------

Splits the dressed-rotation chain in two: a head resynthesized into a circuit a
device can run, and a tail folded into the observable. See
:doc:`../guides/rustiq_resynthesis`.

.. automodule:: mpstab.evolutors.hsynthsmpo
   :members:
   :undoc-members:
   :show-inheritance:

Optimization
------------

.. automodule:: mpstab.evolutors.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: mpstab.evolutors.utils
   :members:
   :undoc-members:
   :show-inheritance:

For more practical examples, see:

- :doc:`../guides/quickstart`
- :doc:`../guides/fidelity_and_approximation`
- :doc:`../examples/introduction`
