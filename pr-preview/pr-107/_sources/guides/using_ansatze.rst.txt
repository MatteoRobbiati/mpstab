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

The :doc:`ansatze_catalogue` lists every ansatz with its physical description and
the parameters that tune its magic and its entanglement.

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


Circuit-library ansätze
~~~~~~~~~~~~~~~~~~~~~~~~~

MPSTAB ships a small library of textbook circuits — ``QFT``, ``QFTPhaseKernel``,
``Grover``, ``QPE``, ``QAE`` and ``TrotterIsing`` — useful as HSMPO/HQSMPO
benchmarks. They are written with high-level gates (``CU1``, ``SWAP``,
``TOFFOLI``, ``RZZ``, …) and, by default, transpiled to a native gate set on
construction, so every non-Clifford gate becomes an ``rz`` rotation that HSMPO
can consume — you never hand-decompose anything::

    from mpstab import HSMPO
    from mpstab.models.ansatze import QFT, TrotterIsing, Grover

    # Quantum Fourier Transform on 6 qubits (transpiled by default)
    ansatz = QFT(nqubits=6)
    simulator = HSMPO(ansatz=ansatz)
    simulator.expectation("Z" * 6)

    # Trotterized transverse-field Ising evolution
    ising = TrotterIsing(nqubits=8, n_steps=4, dt=0.2, J=1.0, h=0.5)

    # Grover search; QPE/QAE take ``n_counting`` and add one work qubit
    grover = Grover(nqubits=4)

Pass ``transpile=False`` to keep the raw high-level circuit for inspection
(HSMPO itself requires the transpiled form). These pair naturally with
``HQSMPO`` and its cut-index split — see :doc:`../api/engines`.


Choosing the couplings
----------------------

Every ansatz with two-qubit gates takes a ``graph`` field: the pairs of qubits
it couples, as a :class:`.Topology` or anything that describes one, such as a
name, a list of edges or a ``networkx`` graph. The single-qubit part of the
ansatz stays the same; the two-qubit gates go on the edges, a layer of
disjoint edges at a time. ``ansatz.topology`` reads back the pairs the circuit
couples::

    from mpstab.models.ansatze import HardwareEfficient, RampedKickedIsing

    # the same kicked Ising ansatz on a 3 x 4 lattice instead of a chain
    lattice = RampedKickedIsing(nqubits=12, n_steps=4, delta_h=0.2, graph="grid")
    print(lattice.topology.depth)  # 4 layers of disjoint bonds per step

    # any list of edges
    pairs = HardwareEfficient(nqubits=6, nlayers=2, graph=[(0, 3), (1, 4), (2, 5)])
    print(pairs.topology.edges)  # ((0, 3), (1, 4), (2, 5))

To run the circuit on a device without routing, build the ansatz on the
coupling map of the device, or on part of it, and check it with
:meth:`.Topology.is_subgraph_of`. A provider lists each coupled pair in both
directions; :meth:`.Topology.from_device` reads it::

    from mpstab import Topology
    from mpstab.models.ansatze import RampedKickedIsing

    coupling_map = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1)]
    device = Topology.from_device(coupling_map)

    ansatz = RampedKickedIsing(
        nqubits=device.nqubits, n_steps=3, delta_h=0.2, graph=device
    )
    assert ansatz.topology.is_subgraph_of(device)

The :doc:`ansatze_catalogue` lists the default couplings of each ansatz, and
the tutorial :doc:`../tutorials/topology` shows how the choice of the graph
changes the cost of HSMPO at fixed magic.

Building Custom Ansätze
------------------------

Extend the ``Ansatz`` class for custom patterns. Let the ansatz take a
``graph`` and place its two-qubit gates on the layers of the corresponding
:class:`.Topology`: the same class then runs on a ring, a lattice or the
coupling map of a device::

    from dataclasses import dataclass

    import numpy as np
    from qibo import gates

    from mpstab import HSMPO, Topology
    from mpstab.models.ansatze import Ansatz


    @dataclass
    class LayeredRY(Ansatz):
        """Random RY rotations on every qubit, then a CZ on every coupled pair."""

        nlayers: int = 2
        graph: object = "ring"  # anything Topology.coerce accepts

        def __post_init__(self):
            super().__post_init__()  # creates the empty circuit
            topology = Topology.coerce(self.graph, self.nqubits)
            for _ in range(self.nlayers):
                for q in range(self.nqubits):
                    self.circuit.add(gates.RY(q, theta=np.random.uniform(-np.pi, np.pi)))
                for layer in topology.layers:
                    for a, b in layer:
                        self.circuit.add(gates.CZ(a, b))


    ansatz = LayeredRY(nqubits=6, graph="grid")
    simulator = HSMPO(ansatz=ansatz)
    print(simulator.expectation("ZZIIII"))

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
