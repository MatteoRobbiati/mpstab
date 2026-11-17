HSynthSMPO on quantum hardware
==============================

:class:`~mpstab.HSynthSMPO` splits the chain of dressed Pauli rotations at a
cut: the *head*, next to the initial state, is resynthesized into native
gates and run on a device, while the *tail*, next to the observable, is folded
into the observable as an MPO.

.. image:: ../../../examples/scripts/quantum_hardware/figures/hsynthsmpo_head_tail_split_concept_en.png
   :alt: The head of the rotation chain runs on hardware, the tail is folded into the observable MPO.

The folded observable is then measured along one of three routes:

.. image:: ../../../examples/scripts/quantum_hardware/figures/hsynthsmpo_three_measurement_routes_en.png
   :alt: The three measurement routes of HSynthSMPO.

.. image:: ../../../examples/scripts/quantum_hardware/figures/hsynthsmpo_qite_cost_analysis.png
   :alt: Cost analysis of the head/tail split.

With the optional rustiq extra (``pip install "mpstab[rustiq]"``) the head is
gate-count optimized; otherwise both scripts fall back to a dependency-free
decomposition.

Approximation from folding the tail
-----------------------------------

Sweeps the cut at a fixed bond-dimension cap and shows the error of the folded
MPO growing as more rotations are pushed into the tail.

.. code-block:: bash

   uv run python examples/scripts/quantum_hardware/hsynthsmpo_mpo_approximation.py

.. literalinclude:: ../../../examples/scripts/quantum_hardware/hsynthsmpo_mpo_approximation.py
   :language: python

Measuring a resynthesized head with finite shots
------------------------------------------------

Resynthesizes every dressed rotation, measures the circuit with a finite shot
budget through the ``"pauli"`` route and checks the estimate against the
classical simulation.

.. code-block:: bash

   uv run python examples/scripts/quantum_hardware/hsynthsmpo_rustiq_fold.py

.. literalinclude:: ../../../examples/scripts/quantum_hardware/hsynthsmpo_rustiq_fold.py
   :language: python
