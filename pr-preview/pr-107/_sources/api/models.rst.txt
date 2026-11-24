Models & Methods
================

The models subpackage contains quantum circuit templates (ansätze) and analysis tools.

Ansätze: Quantum Circuit Templates
----------------------------------

Ansätze are pre-built quantum circuit patterns used as templates for quantum algorithms.

.. automodule:: mpstab.models.ansatze
   :members:
   :undoc-members:
   :show-inheritance:

**Available Ansätze**: see :doc:`../guides/ansatze_catalogue` for the full list,
with the physical description of each ansatz and the parameters that tune its
magic and its entanglement (also exposed as the ``magic_parameters`` and
``entanglement_parameters`` class attributes).

The circuit-library ansätze (``QFT``, ``QFTPhaseKernel``, ``Grover``, ``QPE``,
``QAE``, ``TrotterIsing``) are written with high-level gates (``CU1``, ``SWAP``,
``TOFFOLI``, ``RZZ``, …) and, by default, transpiled to a native gate set on
construction so every non-Clifford gate becomes an ``rz`` rotation that HSMPO
understands — no manual decomposition needed. Pass ``transpile=False`` to keep the
raw high-level circuit (e.g. for inspection); HSMPO itself requires the transpiled
form.

**Key Methods**:

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

Coupling Topologies
-------------------

Which pairs of qubits an ansatz couples: named lattices, any graph, or the
coupling map of a device. Every ansatz with two-qubit couplings takes one
through its ``graph`` field.

.. automodule:: mpstab.models.topology
   :members:
   :show-inheritance:

Entanglement Metrics & Analysis
--------------------------------

Tools for analyzing entanglement and state properties.

.. automodule:: mpstab.models.metrics
   :members:
   :undoc-members:
   :show-inheritance:

**Available Functions**:

- ``stabilizer_renyi_entropy_mps(hsmpo, alpha, n_samples, seed)``: sampled stabilizer Renyi
  entropy of the state an HSMPO represents
- ``meyer_wallach_mps(hsmpo)``: Meyer-Wallach global entanglement of the state an HSMPO represents

For a dense statevector, use Qibo's ``qibo.quantum_info.stabilizer_renyi_entropy`` and
``qibo.quantum_info.meyer_wallach_entanglement``; the HSMPO versions follow the same conventions.


Additional Utilities
--------------------

.. automodule:: mpstab.models.utils
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

============================  ==================================================
Class/Function                Purpose
============================  ==================================================
HardwareEfficient             Standard VQA circuit pattern
CircuitAnsatz                 Wrap Qibo circuits as ansätze
TranspiledAnsatz              Wrap + transpile to native gates
QFT                           Quantum Fourier Transform
QFTPhaseKernel                QFT + diagonal RZ phase kernel
Grover                        Grover search
QPE                           Quantum Phase Estimation
QAE                           Quantum Amplitude Estimation
TrotterIsing                  Trotterized transverse-field Ising evolution
Topology                      Which pairs of qubits an ansatz couples
meyer_wallach_mps             Meyer-Wallach entanglement of an HSMPO state
stabilizer_renyi_entropy_mps  Sampled stabilizer Renyi entropy of an HSMPO
                              state
============================  ==================================================

Related Documentation
-----------------------

- :doc:`../guides/using_ansatze` - Ansätze usage guide
- :doc:`../guides/fidelity_and_approximation` - Understanding approximations
- :doc:`../tutorials/introduction` - Practical examples
