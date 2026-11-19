VQE trained with the analytic gradient
======================================

The transverse-field Ising ring trained with Adam on a hardware-efficient
ansatz twice, from the same initial parameters: once with
:meth:`~mpstab.HSMPO.gradient` and once with the parameter-shift rule. The
script prints the time per step and the final energy of both, and plots the
energy error against the step and against the wall time. The
:doc:`../vqe_gradients` notebook walks through the same training step by step.

Run it from the repository root with

.. code-block:: bash

   uv run python examples/scripts/gradient/vqe_training.py

or, beyond exact methods, with
``--nqubits 40 --nlayers 4 --max-bond 32 --psr-steps 10``.

.. literalinclude:: ../../../examples/scripts/gradient/vqe_training.py
   :language: python
