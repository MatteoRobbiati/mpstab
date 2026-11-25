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
        <a class="hero-button primary" href="quickstart.html">
          <span>Quick start</span>
          <svg class="hero-icon arrow" viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8.5 4.5 12 8l-3.5 3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </a>
        <a class="hero-button" href="installation.html">
          <svg class="hero-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5v7.5M4.8 7 8 10.2 11.2 7M3 13h10" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
          <span>Installation</span>
        </a>
        <a class="hero-button" href="https://github.com/MatteoRobbiati/mpstab">
          <svg class="hero-icon" viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
          <span>GitHub</span>
        </a>
      </div>

.. themed-figure:: stab_ent.svg
   :align: center
   :width: 420px
   :alt: Where mpstab sits among classical simulation methods.

   Where classical simulators reach. Statevector methods are limited by the
   number of qubits, stabilizer methods by non-stabilizerness (magic), and
   tensor networks by entanglement. The HSMPO formalism of mpstab covers states
   that are both highly entangled and moderately non-stabilizer.

Core features
-------------

The HSMPO formalism is implemented on top of state-of-the-art stabilizer and
tensor network simulators, used as operative engines.

.. raw:: html

   <div class="feature-grid">
     <div class="feature-card">
       <span class="feature-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3 3 7.5 12 12l9-4.5L12 3Z"/><path d="m3 12 9 4.5 9-4.5"/><path d="m3 16.5 9 4.5 9-4.5"/></svg></span>
       <p class="feature-title">Hybrid evolution</p>
       <p>Combine stabilizer and tensor network engines for efficient simulation.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3v5M15 3v5"/><path d="M6 8h12v3a6 6 0 0 1-12 0V8Z"/><path d="M12 17v4"/></svg></span>
       <p class="feature-title">Qibo integration</p>
       <p>A native custom backend runs Qibo circuits with <code>mpstab</code> algorithms; Qibo circuits and observables are supported as well.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></span>
       <p class="feature-title">Scalable Clifford simulation</p>
       <p>Simulate pure Clifford circuits with hundreds of qubits.</p>
     </div>
     <div class="feature-card">
       <span class="feature-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 18a8 8 0 1 1 16 0"/><path d="m12 18 4-6"/><path d="M4 21h16"/></svg></span>
       <p class="feature-title">Advanced metrics</p>
       <p>Built-in tools for entanglement entropies, fidelity bounds and circuit analysis.</p>
     </div>
   </div>

Explore the documentation
-------------------------

.. raw:: html

   <details class="nav-section">
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15c-1.5 1.5-2 4-2 6 2 0 4.5-.5 6-2"/><path d="M9 15 6 12c1-4 4.5-8.5 12-9 0 7.5-5 11-9 12Z"/><circle cx="14.5" cy="9.5" r="1.5"/></svg></span><span class="nav-title">Getting started</span>
   <span class="nav-hint">Install mpstab and run a first simulation</span></summary>

* :doc:`installation`
* :doc:`quickstart`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5v-15Z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20"/><path d="M9 8h7"/></svg></span><span class="nav-title">Methods</span>
   <span class="nav-hint">The theory behind the HSMPO formalism</span></summary>

* :doc:`methods/theory_background`
* :doc:`methods/fidelity_and_approximation`
* :doc:`methods/hqsmpo/index`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="m15.5 8.5-2 5-5 2 2-5 5-2Z"/></svg></span><span class="nav-title">User guide</span>
   <span class="nav-hint">Backends, engines, ansätze and optimization</span></summary>

* :doc:`user_guide/qibo_backend_provider`
* :doc:`user_guide/working_with_engines`
* :doc:`user_guide/using_ansatze`
* :doc:`user_guide/ansatze_catalogue`
* :doc:`user_guide/dmrg_optimization`

.. raw:: html

   </details>
   <details class="nav-section">
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="m9 9 2.5 2.5L9 14"/><path d="M13 15h3"/></svg></span><span class="nav-title">Tutorials</span>
   <span class="nav-hint">Runnable notebooks, one idea at a time</span></summary>

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
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="m8 7-5 5 5 5"/><path d="m16 7 5 5-5 5"/><path d="m14 4-4 16"/></svg></span><span class="nav-title">API reference</span>
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
   <summary><span class="nav-icon"><svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 7.5h.01"/></svg></span><span class="nav-title">About</span>
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
