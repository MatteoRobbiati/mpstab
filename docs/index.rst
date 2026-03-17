mpstab: Hybrid Quantum Circuit Simulator
========================================

**mpstab** is an open-source Python library designed for high-performance simulation of quantum circuits using a **hybrid Stabilizer-Matrix Product Operator (MPO)** formalism.

By combining the efficiency of Clifford stabilizers with the expressiveness of Tensor Networks, ``mpstab`` allows researchers to simulate circuits that are "mostly Clifford" but contain non-trivial non-stabilizer resources.

Core Features
-------------
* **Hybrid Evolution**: Seamlessly switch between stabilizer and tensor network engines during a single simulation.
* **Qibo Integration**: Native custom backend support, allowing you to run Qibo circuits using ``mpstab`` algorithms.
* **Advanced Metrics**: Built-in tools for computing entanglement entropies, magic measures, and circuit fidelities.
* **Error Mitigation**: Specialized modules for quantum error mitigation within the hybrid formalism.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   examples/introduction

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   examples/introduction

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
