Working with Engines
====================

MPSTAB runs on two computational backends (engines): one for the stabilizer
part of the circuit, one for the tensor network.

By default HSMPO uses Stim for the stabilizer part and Quimb for the tensor
network. Engines are passed at construction, or swapped later with
``set_engines``::

    from mpstab import HSMPO
    from mpstab.engines import QuimbEngine
    from qibo import Circuit, gates

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    simulator = HSMPO(ansatz=circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    # ...or later, which rebuilds the cached MPS:
    simulator.set_engines(tn_engine=QuimbEngine())

Available Engines
-----------------

Stabilizer Engines
~~~~~~~~~~~~~~~~~~~

Engines that handle the Clifford (stabilizer) part of the circuit:

**Stim Engine**
  - Speed: State-of-art Clifford simulator
  - Coverage: Pure Clifford circuits
  - Best for: High-performance stabilizer simulation
  - External library: `Stim <https://github.com/quantumlib/Stim>`_

::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    simulator = HSMPO(ansatz=circuit)
    # By default we use Stim

Tensor Network Engines
~~~~~~~~~~~~~~~~~~~~~~

Engines that handle the non-Clifford effects via MPO:

**Quimb Engine**
  - Speed: Fast (optimized contractions)
  - Memory: Efficient for large bond dimensions
  - Best for: Large systems with moderate bond dimensions
  - Backends: ``QuimbEngine(backend="jax")`` or ``"torch"`` for autodiff
  - External library: `Quimb <https://quimb.readthedocs.io/>`_

::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.RY(0, theta=0.5))

    simulator = HSMPO(ansatz=circuit, max_bond_dimension=32)
    # Uses Quimb tensor network engine by default

Native Engines
~~~~~~~~~~~~~~

``NativeStabilizersEngine`` and ``NativeTensorNetworkEngine`` are pure-Python
implementations of the same interfaces, meant to be read: they show what a
stabilizers engine and a tensor-network engine actually do, without stim or
quimb in the way. They give the same results, more slowly, and the native
tensor-network engine supports neither DMRG nor the head/tail split of
``HSynthSMPO``.

::

    from mpstab import HSMPO
    from mpstab.engines import NativeStabilizersEngine, NativeTensorNetworkEngine
    from mpstab.models.ansatze import HardwareEfficient

    simulator = HSMPO(
        ansatz=HardwareEfficient(nqubits=5, nlayers=2),
        stab_engine=NativeStabilizersEngine(),
        tn_engine=NativeTensorNetworkEngine(),
    )
    print(simulator.expectation("ZZZZZ"))

Engine Configuration
--------------------

::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Basic HSMPO with the default engines
    simulator = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=32
    )

    # Access engine information
    print(f"Stabilizer engine: {simulator.stab_engine}")
    print(f"TN engine: {simulator.tn_engine}")

Tips for Efficient Simulation
------------------------------

1. **Use Clifford circuits when possible** - they're orders of magnitude faster
2. **Start with low bond dimensions** - increase only if accuracy is insufficient
3. **Profile your circuits** - understand what gates are expensive

Example: Adaptive Bond Dimension
---------------------------------

::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient
    from qibo import gates

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Start with low bond dimension
    for max_bd in [2, 4, 8, 16, 32]:
        simulator = HSMPO(ansatz=ansatz, max_bond_dimension=max_bd)
        result, fidelity = simulator.expectation("Z" * 10, return_fidelity=True)

        print(f"χ_max={max_bd}: result={result:.4f}, fidelity={fidelity:.4f}")

        # Stop if accuracy is sufficient
        if fidelity > 0.99:
            break

Further Reading
---------------

- `Quimb Documentation <https://quimb.readthedocs.io/>`_
- `Stim Documentation <https://quantumlib.org/reference/python/stim/>`_
- :doc:`../api/engines`
