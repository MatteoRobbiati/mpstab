mpstab: a cutting-edge quantum circuit simulator
================================================

.. container:: hero

   .. container:: hero-lead

      **mpstab** is an open-source Python library for high-performance
      simulation of quantum circuits with a **hybrid Stabilizer-Matrix
      Product Operator (HSMPO)** formalism.

   By combining the efficiency of Clifford stabilizers with the expressiveness
   of tensor networks, ``mpstab`` simulates circuits that are "mostly Clifford"
   but contain non-trivial non-stabilizer resources. The hybrid stabilizer-MPO
   representation was introduced in `this work from Mello, Santini and Collura
   <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.133.150604>`_.

   .. raw:: html

      <div class="hero-buttons">
        <a class="hero-button primary" href="quickstart.html">Quick start</a>
        <a class="hero-button" href="installation.html">Installation</a>
        <a class="hero-button" href="https://github.com/MatteoRobbiati/mpstab">GitHub</a>
      </div>

.. themed-figure:: stab_ent.svg
   :align: center
   :width: 520px
   :alt: Where mpstab sits among classical simulation methods.

   Where classical simulators reach. Statevector methods are limited by the
   number of qubits, stabilizer methods by non-stabilizerness (magic), and
   tensor networks by entanglement. The HSMPO formalism of mpstab covers states
   that are both highly entangled and moderately non-stabilizer.

Core features
-------------

The HSMPO formalism is implemented on top of state-of-the-art stabilizer and
tensor network simulators, used as operative engines.

.. container:: feature-grid

   .. container:: feature-card

      **Hybrid evolution**

      Combine stabilizer and tensor network engines for efficient simulation.

   .. container:: feature-card

      **Qibo integration**

      A native custom backend runs Qibo circuits with ``mpstab`` algorithms;
      Qibo circuits and observables are supported as well.

   .. container:: feature-card

      **Scalable Clifford simulation**

      Simulate pure Clifford circuits with hundreds of qubits.

   .. container:: feature-card

      **Advanced metrics**

      Built-in tools for entanglement entropies, fidelity bounds and circuit
      analysis.

Explore the documentation
-------------------------

.. raw:: html

   <details class="nav-section">
   <summary><span class="nav-title">Getting started</span>
   <span class="nav-hint">Install mpstab and run a first simulation</span></summary>

* :doc:`installation`
* :doc:`quickstart`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-title">Methods</span>
   <span class="nav-hint">The theory behind the HSMPO formalism</span></summary>

* :doc:`methods/theory_background`
* :doc:`methods/fidelity_and_approximation`
* :doc:`methods/hqsmpo/index`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-title">User guide</span>
   <span class="nav-hint">Backends, engines, ansätze and optimization</span></summary>

* :doc:`user_guide/qibo_backend_provider`
* :doc:`user_guide/working_with_engines`
* :doc:`user_guide/using_ansatze`
* :doc:`user_guide/ansatze_catalogue`
* :doc:`user_guide/dmrg_optimization`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-title">Tutorials</span>
   <span class="nav-hint">Runnable notebooks, one idea at a time</span></summary>

* :doc:`tutorials/index`
* :doc:`tutorials/introduction`
* :doc:`tutorials/noise`
* :doc:`tutorials/advanced_noise`
* :doc:`tutorials/error_mitigation`
* :doc:`tutorials/ground_states`
* :doc:`tutorials/state_metrics`
* :doc:`tutorials/vqe_gradients`
* :doc:`tutorials/echo`
* :doc:`tutorials/free_fermions`
* :doc:`tutorials/frameworks`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-title">API reference</span>
   <span class="nav-hint">Every public class and function</span></summary>

* :doc:`api/evolutors`
* :doc:`api/engines`
* :doc:`api/qpu`
* :doc:`api/noise`
* :doc:`api/error_mitigation`
* :doc:`api/pauli`
* :doc:`api/qibo_tools`
* :doc:`api/models`
* :ref:`genindex`
* :ref:`modindex`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-title">About</span>
   <span class="nav-hint">Publications and source code</span></summary>

* :doc:`publications`
* `GitHub repository <https://github.com/MatteoRobbiati/mpstab>`_: issues and
  contributions are welcome.

.. raw:: html

   </details>

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Methods

   methods/theory_background
   methods/fidelity_and_approximation
   methods/hqsmpo/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User guide

   user_guide/qibo_backend_provider
   user_guide/working_with_engines
   user_guide/using_ansatze
   user_guide/ansatze_catalogue
   user_guide/dmrg_optimization

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/evolutors
   api/engines
   api/qpu
   api/noise
   api/error_mitigation
   api/pauli
   api/qibo_tools
   api/models

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: About

   publications
   GitHub repository <https://github.com/MatteoRobbiati/mpstab>
