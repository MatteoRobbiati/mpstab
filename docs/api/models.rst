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

*Wrappers & structural:*

- ``Ansatz``: Abstract base class for all ansätze
- ``CircuitAnsatz``: Wrapper for Qibo circuits
- ``TranspiledAnsatz``: Wrap a circuit and transpile it into a device's native gates

*Variational patterns:*

- ``HardwareEfficient``: Industry-standard RY + CZ pattern for NISQ devices
- ``HardwareEfficientBrickwork``: Brickwork RY/RZ + CZ pattern with a ring-closing bond
- ``HammingWeightPreserving``: RBS-based Hamming-weight-preserving ansatz
- ``FloquetAnsatz``: Floquet echo circuit with tunable magic

*Circuit library (textbook algorithms, for HSMPO/HSynthSMPO benchmarks):*

- ``QFT``: Quantum Fourier Transform
- ``QFTPhaseKernel``: QFT followed by a diagonal RZ phase kernel
- ``Grover``: Grover search (oracle + diffuser iterations)
- ``QPE``: Quantum Phase Estimation
- ``QAE``: Quantum Amplitude Estimation
- ``TrotterIsing``: First-order Trotter evolution of the transverse-field Ising model

The circuit-library ansätze are written with high-level gates (``CU1``, ``SWAP``,
``TOFFOLI``, ``RZZ``, …) and, by default, transpiled to a native gate set on
construction so every non-Clifford gate becomes an ``rz`` rotation that HSMPO
understands — no manual decomposition needed. Pass ``transpile=False`` to keep the
raw high-level circuit (e.g. for inspection); HSMPO itself requires the transpiled
form.

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

**Available Functions**:

- ``stabilizer_renyi_entropy(state, alpha)``: Stabilizer Renyi entropy for a given state and order


Additional Utilities
--------------------

.. automodule:: mpstab.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

==========================  ==================================================
Class/Function              Purpose
==========================  ==================================================
HardwareEfficient           Standard VQA circuit pattern
CircuitAnsatz               Wrap Qibo circuits as ansätze
TranspiledAnsatz            Wrap + transpile to native gates
QFT                         Quantum Fourier Transform
QFTPhaseKernel              QFT + diagonal RZ phase kernel
Grover                      Grover search
QPE                         Quantum Phase Estimation
QAE                         Quantum Amplitude Estimation
TrotterIsing                Trotterized transverse-field Ising evolution
stabilizer_renyi_entropy    Calculate Renyi entropy for stabilizer states
==========================  ==================================================

Related Documentation
-----------------------

- :doc:`../guides/using_ansatze` - Ansätze usage guide
- :doc:`../guides/fidelity_and_approximation` - Understanding approximations
- :doc:`../examples/introduction` - Practical examples
