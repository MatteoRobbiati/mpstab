Working with Engines
====================

MPSTAB supports multiple computational backends (engines) for different simulation scenarios.

Once an HSMPO is defined, the engines can be customized using the ``hsmpo.set_engines`` method. By default we set Stim and Quimb respectively for stabilizers and tensor network simulations::

    simulator = HSMPO(ansatz=circuit)
    # Here we can pass the desired engines
    simulator.set_engines()

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

    simulator = HSMPO(ansatz=circuit)
    # By default we use Stim

**Native Stabilizer Engine**
  - Speed: Lightweight, fast on single thread CPU
  - Coverage: Pure Clifford circuits only
  - Best for: Large Clifford circuits (thousands of qubits)
  - Implementation: In-house Clifford simulator

::

    from mpstab import HSMPO
    simulator = HSMPO(ansatz=circuit)
    simulator.set_engines(stab_engine=NativeStabilizerEngine)

Tensor Network Engines
~~~~~~~~~~~~~~~~~~~~~~

Engines that handle the non-Clifford effects via MPO:

**Quimb Engine**
  - Speed: Fast (optimized contractions)
  - Memory: Efficient for large bond dimensions
  - Best for: Large systems with moderate bond dimensions
  - Features: Automatic contraction path optimization
  - External library: `Quimb <https://quimb.readthedocs.io/>`_

::

    simulator = HSMPO(ansatz=circuit, max_bond_dimension=32)
    # Uses Quimb tensor network engine by default

**Native Tensor Network Engine**
  - Speed: Moderate
  - Memory: Scales with bond dimension
  - Best for: Small systems with fine controlled network structure


Engine Configuration
--------------------

::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Basic HSMPO with automatic engine selection
    simulator = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=32
    )

    # Access engine information
    simulator.set_engines()  # Configure engines
    print(f"Stabilizer engine: {simulator.st_engine}")
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

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Start with low bond dimension
    for max_bd in [2, 4, 8, 16, 32]:
        simulator = HSMPO(ansatz=ansatz, max_bond_dimension=max_bd)
        result = simulator.expectation("Z" * 10)
        fidelity = simulator.fidelity_lower_bound

        print(f"χ_max={max_bd}: result={result:.4f}, fidelity={fidelity:.4f}")

        # Stop if accuracy is sufficient
        if fidelity > 0.99:
            break

Further Reading
---------------

- `Quimb Documentation <https://quimb.readthedocs.io/>`_
- `Stim Documentation <https://quantumlib.org/reference/python/stim/>`_
- :doc:`../api/engines`
