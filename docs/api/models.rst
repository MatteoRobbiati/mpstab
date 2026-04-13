Models & Methods
================

The models subpackage contains quantum circuit templates (ansätze), error mitigation techniques, and analysis tools.

Ansätze: Quantum Circuit Templates
----------------------------------

Ansätze are pre-built quantum circuit patterns used as templates for quantum algorithms.

.. automodule:: mpstab.models.ansatze
   :members:
   :undoc-members:
   :show-inheritance:

**Available Ansätze**:

- ``Ansatz``: Abstract base class for all ansätze
- ``CircuitAnsatz``: Wrapper for Qibo circuits
- ``HardwareEfficient``: Industry-standard pattern for NISQ devices

**Key Methods**:

- ``execute(nshots, initial_state, with_noise)``: Execute the circuit
- ``update_noise_model(noise_model)``: Add noise to simulation
- ``partitionate_circuit(replacement_probability, replacement_method)``: Partition circuit
- ``nparams``: Number of circuit parameters
- ``circuit``: Access the underlying QiboCircuit

**Example Usage**::

    from mpstab.models.ansatze import HardwareEfficient
    from mpstab import HSMPO

    # Create ansatz
    ansatz = HardwareEfficient(nqubits=5, nlayers=3)

    # Use with HSMPO
    simulator = HSMPO(ansatz=ansatz, max_bond_dimension=32)
    result = simulator.expectation("ZZZZZ")

See :doc:`../guides/using_ansatze` for detailed examples.

Error Mitigation
----------------

Techniques for reducing approximation errors and improving simulation accuracy.

.. automodule:: mpstab.models.mitigation_methods
   :members:
   :undoc-members:
   :show-inheritance:

Entanglement Metrics & Analysis
--------------------------------

Tools for analyzing entanglement and state properties.

.. automodule:: mpstab.models.entropies
   :members:
   :undoc-members:
   :show-inheritance:

**Available Metrics**:

- ``EntanglementEntropy``: Von Neumann entanglement entropy
- ``MagicMeasure``: Non-stabilizer resource (magic) measure
- ``EntanglementSpectrum``: Singular values from Schmidt decomposition

**Key Methods**:

- ``compute(state)``: Compute metric for a state
- ``across_cut(subsystem_a, subsystem_b)``: Entropy across a partition

**Example Usage**::

    from mpstab.models.entropies import EntanglementEntropy
    from mpstab import HSMPO

    simulator = HSMPO(ansatz=my_circuit)

    # Compute entanglement entropy across qubits 0-2 vs 3-4
    ee = EntanglementEntropy.across_cut(
        mps=simulator.mps,
        subsystem_a=[0, 1, 2],
        subsystem_b=[3, 4]
    )
    print(f"Entanglement entropy: {ee:.4f} bits")

Additional Utilities
--------------------

.. automodule:: mpstab.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

====================  ===============================================
Class/Function        Purpose
====================  ===============================================
HardwareEfficient     Standard VQA circuit pattern
CircuitAnsatz         Wrap Qibo circuits as ansätze
ErrorMitigation       Apply various mitigation techniques
EntanglementEntropy   Measure entanglement
MagicMeasure          Quantify non-stabilizer resources
====================  ===============================================

Related Documentation
-----------------------

- :doc:`../guides/using_ansatze` - Ansätze usage guide
- :doc:`../guides/fidelity_and_approximation` - Understanding approximations
- :doc:`../examples/introduction` - Practical examples
