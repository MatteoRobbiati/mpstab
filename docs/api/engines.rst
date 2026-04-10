Engines
=======

Engines are the computational backends that perform low-level state operations. MPSTAB uses a modular engine architecture allowing different implementations to be swapped for specific use cases.

**Two Engine Types**:

1. **Stabilizer Engines**: Handle Clifford gates efficiently
2. **Tensor Network Engines**: Manage non-Clifford effects via MPO

Stabilizer Engines
------------------

Handle pure stabilizer/Clifford circuit simulation.

.. automodule:: mpstab.engines.stabilizers.native
   :members:
   :undoc-members:
   :show-inheritance:

**Native Stabilizer Engine**:
- Pure Python implementation
- Highest compatibility
- Suitable for all Clifford circuits
- Can handle hundreds of qubits

.. automodule:: mpstab.engines.stabilizers.stim
   :members:
   :undoc-members:
   :show-inheritance:

**Stim Engine**:
- Fast, C++-backed stabilizer simulator
- Requires Stim library installation
- Best for large Clifford circuits
- Seamless integration with MPSTAB

Tensor Network Engines
----------------------

Manage non-Clifford effects through Matrix Product Operator (MPO) representations.

.. automodule:: mpstab.engines.tensor_networks.native
   :members:
   :undoc-members:
   :show-inheritance:

**Native Tensor Network Engine**:
- Pure Python implementation
- Direct control over MPS/MPO operations
- Suitable for understanding tensor network mechanics
- Good for small to medium systems

.. automodule:: mpstab.engines.tensor_networks.quimb
   :members:
   :undoc-members:
   :show-inheritance:

**Quimb Engine**:
- Advanced tensor network library integration
- Optimized contraction paths via Cotengra
- Excellent for larger systems with manageable bond dimensions
- Automatic optimization of tensor operations

Engine Comparison
-----------------

+-------------------+----------+-------------------+---------------+
| Engine            | Speed    | Best For          | Max Qubits*   |
+===================+==========+===================+===============+
| Native Stabilizer | Medium   | Clifford only     | 1000+         |
+-------------------+----------+-------------------+---------------+
| Stim              | Very Fast| Clifford, large   | 1000+         |
+-------------------+----------+-------------------+---------------+
| Native TN         | Moderate | Understanding TN  | ~20           |
+-------------------+----------+-------------------+---------------+
| Quimb             | Fast     | Mixed circuits    | ~50           |
+-------------------+----------+-------------------+---------------+

Approximate limits depend on bond dimension and circuit complexity. Performance scales with system size.

Selecting Engines
-----------------

Engines are automatically selected based on your circuit type and parameters::

    from mpstab import HSMPO

    # Clifford circuit - uses stabilizer engine only
    simulator = HSMPO(ansatz=clifford_circuit)

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

Performance Tuning
------------------

Tips for optimal engine performance:

1. **Stabilizer Engines**: Can handle very large systems efficiently
2. **Tensor Network Engines**: Performance highly dependent on bond dimension
3. **Quimb Engine**: Automatic optimization may have startup overhead
4. **Native TN Engine**: Lower overhead but potentially slower for large systems

See Also
--------

- :doc:`../guides/working_with_engines` - Engine selection guide
- :doc:`../examples/introduction` - Practical examples
