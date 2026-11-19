Noisy simulation against density matrices
=========================================

A kicked Ising circuit under depolarizing noise, correlated :math:`ZZ`
dephasing and a coherent :math:`ZZ` error on every CNOT. For several noise
strengths the script prints the noiseless value, the noisy value from
:meth:`~mpstab.HSMPO.noisy_expectation` and the exact value from a qibo
density-matrix simulation, with timings. The :doc:`../noise` and
:doc:`../advanced_noise` notebooks walk through the same tools step by step.

Run it from the repository root with

.. code-block:: bash

   uv run python examples/scripts/noise/noisy_simulation.py

.. literalinclude:: ../../../examples/scripts/noise/noisy_simulation.py
   :language: python
