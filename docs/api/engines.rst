Engines
=======

Engines are the computational backends that perform low-level state operations. Each kind has an abstract interface, so a custom engine can be swapped in.

**Two Engine Types**:

1. **Stabilizer Engines**: Handle Clifford gates efficiently
2. **Tensor Network Engines**: Manage non-Clifford effects via MPO

Stabilizer Engine
-----------------

.. automodule:: mpstab.engines.stabilizers.stim
   :members:
   :undoc-members:
   :show-inheritance:

**Stim Engine**: Clifford backpropagation of Pauli observables with the C++
`Stim <https://github.com/quantumlib/Stim>`_ simulator.

Tensor Network Engine
---------------------

.. automodule:: mpstab.engines.tensor_networks.quimb
   :members:
   :undoc-members:
   :show-inheritance:

**Quimb Engine**: MPS evolution, MPO expectation values and operator
conjugation with `quimb <https://quimb.readthedocs.io/>`_, on the numpy, jax or
torch backend.

Selecting Engines
-----------------

Engines are automatically selected based on your circuit type and parameters::

    from qibo import Circuit, gates
    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    clifford_circuit = Circuit(nqubits=10)
    # Add only Clifford gates
    for q in range(10):
         clifford_circuit.add(gates.H(q))
         clifford_circuit.add(gates.CNOT(q0=q%10, q1=(q+1)%10))

    # Clifford circuit - uses stabilizer engine only
    simulator = HSMPO(ansatz=clifford_circuit)

    # Constructing a circuit with also magic gates
    mixed_circuit = HardwareEfficient(
         nqubits=10,
         nlayers=3,
    )

    # Mixed circuit with bond limit - uses both engines
    simulator = HSMPO(
        ansatz=mixed_circuit,
        max_bond_dimension=32  # Enables tensor network engine
    )

For detailed guidance, see :doc:`../guides/working_with_engines`

Engine Interface
----------------

All engines implement standardized interfaces for compatibility.

Base Engine Classes::

    from mpstab.engines import StabilizersEngine, TensorNetworkEngine

These abstract base classes define the required methods that any engine must implement.

See Also
--------

- :doc:`../guides/working_with_engines` - Engine selection guide
- :doc:`../tutorials/introduction` - Practical examples
