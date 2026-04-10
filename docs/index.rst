mpstab: a cutting-edge quantum circuit simulator
================================================

**mpstab** is an open-source Python library designed for high-performance simulation of quantum circuits using a **hybrid Stabilizer-Matrix Product Operator (HSMPO)** formalism.

By combining the efficiency of Clifford stabilizers with the expressiveness of Tensor Networks, ``mpstab`` allows researchers to simulate circuits that are "mostly Clifford" but contain non-trivial non-stabilizer resources. The hybrid stabilizer-MPO representation was introduced in `this work from Mello, Santini and Collura  <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150604>`_.

Core Features
-------------

The HSMPO formalism is implemented here exploiting, as operative engines, the state-of-art stabilizers and tensor network simulators.
Among our core features:

* **Hybrid Evolution**: Seamlessly combine stabilizer and tensor network engines for efficient simulation.
* **Qibo Integration**: Native custom backend support, allowing you to run Qibo circuits using ``mpstab`` algorithms. Quantum circuit and observables definitions from Qibo are supported as well.
* **Scalable Clifford Simulation**: Simulate pure Clifford circuits with hundreds of qubits.
* **Advanced Metrics**: Built-in tools for computing entanglement entropies, fidelity bounds, and circuit analysis.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   guides/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials & Examples

   examples/introduction
   guides/working_with_engines
   guides/using_ansatze
   guides/fidelity_and_approximation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/evolutors
   api/engines
   api/models

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
