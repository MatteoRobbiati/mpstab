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

Native Engines
--------------

Pure-Python implementations of the two interfaces, kept as readable examples
of what an engine does rather than for performance. They give the same
expectation values as stim and quimb, but are slower and cover less: the
native tensor-network engine has no operator-side conjugation (so no head/tail
split in :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`) and no DMRG.

.. automodule:: mpstab.engines.stabilizers.native.engine
   :members:
   :show-inheritance:

**Native Stabilizers Engine**: backpropagates a Pauli string by composing one
tableau per gate, on the XZ-encoded Pauli arithmetic of
:mod:`mpstab.engines.stabilizers.native.pauli_string`.

.. automodule:: mpstab.engines.tensor_networks.native.engine
   :members:
   :show-inheritance:

**Native Tensor Network Engine**: an MPS in Vidal canonical form on a small
in-package tensor-network layer
(:mod:`mpstab.engines.tensor_networks.native.circuit_mps`), evolved by
applying MPOs and re-splitting by SVD.

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

The evolutors never look inside an engine's states or operators: everything
they need goes through these methods, so a new tensor-network engine only has
to implement them. ``tests/test_engine_agnostic.py`` enforces this with an
engine whose states and operators expose nothing.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``TensorNetworkEngine`` methods
     - Needed for
   * - ``build_circuit_mps``, ``pauli_mpo``, ``expval``, ``pauli_rot``,
       ``squared_norm``
     - Required: building and evolving the MPS, expectation values and
       truncation fidelities.
   * - ``pauli_expvals``, ``rotation_operator``, ``apply_rotation``,
       ``copy_state``, ``bond_dimension``
     - Optional, with a generic default an engine may override for speed.
   * - ``conjugate_operator``, ``operator_site_arrays``
     - Optional: the head/tail split of ``HSynthSMPO`` and its measurement
       routes. Operators leave the engine only as numpy site arrays.
   * - ``pauli_sum_mpo``, ``minimize_energy``
     - Optional: DMRG (``HSMPO.minimize_expectation``).
   * - ``circuit_truncation_fidelity``
     - Optional: ``HSMPO.truncation_fidelity_pure_tn``.

An optional method an engine does not implement raises ``NotImplementedError``
only when the feature that needs it is used.

A stabilizers engine is where the Clifford part of the circuit is handled; the
evolutors and the noise compiler only ask it for backpropagated Pauli strings.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``StabilizersEngine`` methods
     - Needed for
   * - ``backpropagate``
     - Required: dressing observables with the Clifford circuit.
   * - ``backpropagate_many``
     - Optional, with a generic default an engine may override to share work
       across observables.
   * - ``backpropagate_at``
     - Dressing each magic gate and noise event with the Clifford gates before
       it. The default replays each prefix through ``backpropagate``;
       ``StimEngine`` overrides it with a single tableau pass.

See Also
--------

- :doc:`../guides/working_with_engines` - Engine selection guide
- :doc:`../examples/introduction` - Practical examples
