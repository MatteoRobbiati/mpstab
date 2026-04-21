.. _dmrg_optimization:

DMRG Optimization of Quantum Hamiltonians
==========================================

.. raw:: html

   <div class="admonition note">
   <p class="admonition-title">Advanced Topic</p>
   <p>
   This guide covers DMRG-based optimization for finding ground states of quantum Hamiltonians.
   For an introduction to mpstab, see the <a href="quickstart.html">Quickstart</a> guide.
   </p>
   </div>

Overview
--------

**Density Matrix Renormalization Group (DMRG)** is one of the most powerful algorithms for finding ground states of quantum systems. Unlike circuit-based approaches like VQE that optimize circuit parameters, DMRG directly optimizes the quantum state representation as a **Matrix Product State (MPS)**.

In **mpstab**, we leverage DMRG through the Hybrid Stabilizer-MPO (HSMPO) formalism, which:

1. Decomposes quantum circuits into Clifford stabilizers and magic gates
2. Represents the state efficiently as an MPS
3. Uses DMRG to optimize the MPS tensors directly

This approach dramatically improves computational efficiency compared to traditional VQE, allowing us to solve larger systems with higher accuracy.

Key Advantages
--------------

- **Scalability**: Efficiently handle 10-20+ qubit systems
- **Accuracy**: Near-exact ground state energies (controlled by MPS bond dimension)
- **Efficiency**: Converges in just a few sweeps vs. thousands of VQE iterations
- **Versatility**: Works for any Hamiltonian, not limited to hardware-native gates

Use Cases
---------

DMRG is particularly valuable for:

- :doc:`Quantum chemistry <using_ansatze>` simulations (molecular ground states)
- Condensed matter systems (Heisenberg, Ising, Hubbard models)
- Optimization benchmarking and algorithm comparison
- Finding ground states for quantum phase transitions
- Training quantum circuits for VQE initialization

The Heisenberg Model Example
-----------------------------

The **Heisenberg XXX model** is an ideal system for demonstrating DMRG:

.. math::

   H = \sum_{i=1}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})

This represents magnetic interactions between nearest-neighbor spins. The model has:

- Known analytical solutions for small systems (benchmarking)
- Rich physical properties (phase transitions, entanglement)
- Growing complexity with system size (testing scalability)


Complete Tutorial Notebook
---------------------------

Below is an interactive notebook demonstrating DMRG optimization on the Heisenberg model for systems up to 12 qubits:

.. nbinput:: python
   :linenos:

   import numpy as np
   import matplotlib.pyplot as plt
   from qibo import hamiltonians
   from qibo.symbols import X, Y, Z
   from mpstab.models.ansatze import HardwareEfficient
   from mpstab.evolutors.hsmpo import HSMPO
   from mpstab.engines import QuimbEngine, StimEngine

.. toctree::
   :maxdepth: 1
   :caption: Interactive Examples

   ../examples/dmrg_optimization_heisenberg

Getting Started with DMRG
-------------------------

Basic Workflow
~~~~~~~~~~~~~~

Here's the basic workflow for optimizing any Hamiltonian with DMRG:

.. code-block:: python

   from mpstab.models.ansatze import HardwareEfficient
   from mpstab.evolutors.hsmpo import HSMPO
   from mpstab.engines import QuimbEngine, StimEngine

   # 1. Create ansatz and HSMPO
   ansatz = HardwareEfficient(nqubits=8, nlayers=3)
   hsmpo = HSMPO(ansatz=ansatz)

   # 2. Set engines
   hsmpo.set_engines(
       stab_engine=StimEngine(),
       tn_engine=QuimbEngine(backend="numpy")
   )

   # 3. Define Hamiltonian (any observable)
   from qibo.hamiltonians import XXZ
   hamiltonian = XXZ(nqubits=8, delta=0.5)

   # 4. Run DMRG optimization
   result = hsmpo.minimize_expectation(
       observables=hamiltonian,
       method="dmrg",
       bond_dims=[10, 20, 50],
       max_sweeps=20,
       verbosity=1
   )

   # 5. Extract results
   ground_state_energy = result['energy']
   ground_state_mps = result['ground_state']
   converged = result['converged']

Key Parameters
~~~~~~~~~~~~~~

**Bond Dimensions** (``bond_dims``)
   Controls the expressiveness of the MPS:

   - Larger dimensions → better accuracy, more computation
   - Typical values: ``[10, 20, 50, 100]`` for growing schedule
   - Exponential scaling with dimension (algorithm complexity is :math:`O(D^3)`)

**DMRG Sweeps** (``max_sweeps``)
   Number of left-to-right and right-to-left optimization passes:

   - Usually converges in 5-20 sweeps
   - More sweeps for harder problems or higher accuracy
   - Diminishing returns after ~10-15 sweeps typically

**Convergence Tolerance** (``tol``)
   Energy change threshold for convergence:

   - Default: ``1e-6``
   - Tighter tolerance (``1e-8``) for higher accuracy
   - Looser tolerance (``1e-4``) for faster (approximate) results

Advanced Topics
---------------

State Initialization
~~~~~~~~~~~~~~~~~~~~

For better convergence, initialize DMRG with a good ansatz:

.. code-block:: python

   # Custom initialization with more layers
   ansatz = HardwareEfficient(nqubits=8, nlayers=5)
   hsmpo = HSMPO(ansatz=ansatz)

   # DMRG will start from the state prepared by this ansatz
   result = hsmpo.minimize_expectation(observables=hamiltonian, ...)

And one can also use the `initial_state` argument to directly provide
an initial state to place before the given ansatz. For example, when dealing with
special symmetry preserving Hamiltonians, one can initialize the state in the
correct symmetry sector of the Hilbert space, which can significantly speed up
convergence and improve accuracy.

Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize linear combinations of observables:

.. code-block:: python

   # Minimize H1 + 0.5*H2
   result = hsmpo.minimize_expectation(
       observables={
           "ZZZZ": 1.0,
           "XXXX": 0.5,
           "YYYY": 0.5
       },
       method="dmrg",
       ...
   )

See Also
--------

- :doc:`Working with Engines <working_with_engines>`
- :doc:`Using Ansatze <using_ansatze>`
- :doc:`Theory Background <theory_background>`
