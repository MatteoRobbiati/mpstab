Using Ansätze
==============

Ansätze are pre-built quantum circuit patterns that serve as templates for quantum algorithms.
Out HSMPO surrogate works around a provided ansatz, but, as you know from previous examples, a simple Qibo circuit works too.
This is because we internally take care of constructing a proper ansatz around the provided circuit.

Available Ansätze
------------------

In MPSTAB, we provide a few precomputed ansätze, for example Hardware Efficient or Floquet Dynamics inspired.
More remarkably, we provide two, more abstract, models: a first general wrapper, which can be used to transform
any Qibo circuit into an Mpstab ansatz (CircuitAnsatz), and a more complicated ansatz, which allows to provide
more information about the quantum hardware and, then, recover a compiled and transpiled ansatz (TranspiledAnsatz). This second
model is particularly relevant in contexts where one wants to have control of the number of gates executed
on a real quantum device.

CircuitAnsatz
~~~~~~~~~~~~~~

Wrap any existing Qibo circuit as an ansatz::

    from mpstab import HSMPO
    from qibo import Circuit, gates
    from mpstab.models.ansatze import CircuitAnsatz

    # Create a custom circuit
    circuit = Circuit(5)
    for q in range(5):
        circuit.add(gates.H(q))
        circuit.add(gates.RY(q, theta=0.5))
    circuit.add(gates.CNOT(0, 1))

    # Wrap as ansatz
    ansatz = CircuitAnsatz(qibo_circuit=circuit)
    simulator = HSMPO(ansatz=ansatz)


Building Custom Ansätze
------------------------

Extend the ``Ansatz`` class for custom patterns::

    from mpstab import HSMPO
    from mpstab.models.ansatze import Ansatz
    from qibo import Circuit, gates
    import numpy as np

    class MyCustomAnsatz(Ansatz):
        """Custom ansatz for specific problem."""

        def __init__(self, nqubits, pattern="ring"):
            super().__init__(nqubits)
            self.pattern = pattern
            self._build_circuit()

        def _build_circuit(self):
            # Your circuit construction logic
            self.circuit = Circuit(self.nqubits)

            # Add gates based on pattern
            if self.pattern == "ring":
                for q in range(self.nqubits):
                    self.circuit.add(gates.H(q))
                for q in range(self.nqubits):
                    next_q = (q + 1) % self.nqubits
                    self.circuit.add(gates.CNOT(q, next_q))

    # Use with HSMPO
    ansatz = MyCustomAnsatz(nqubits=5, pattern="ring")
    simulator = HSMPO(ansatz=ansatz)

The ansatz partitioning
-----------------------

An HSMPO surrogate is constructed grouping all the Clifford operations in the circuit and
rewriting the representation of the unitary in terms of new - global - rotations. All this
procedure, that is well described in the original HSMPO paper, requires a fine partitioning
of the original quantum circuit, and a reconstruction of a series of operations alternately
handled by the stabilizers and the tensor network engines. Our ansätze provide a built int
method to partitionate and recover all these features of the provided circuit.


Further Reading
---------------

- `Qibo Circuit Documentation <https://qibo.science/>`_
- :doc:`../api/models`
- Research papers on ansatz design and VQA algorithms
