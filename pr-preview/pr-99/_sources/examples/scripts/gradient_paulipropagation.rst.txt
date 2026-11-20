Gradients against PauliPropagation.jl
=====================================

The adjoint gradient of :meth:`~mpstab.HSMPO.value_and_gradient` against
PauliPropagation.jl differentiated with a compiled ReverseDiff tape, on the
transverse-field Ising ring and a hardware-efficient ansatz. For each width it
prints the time of one gradient and its error against an exact reference, and,
for PauliPropagation, the one-time cost of compiling the tape, with and without
its ``max_weight`` and ``max_freq`` truncations.

Run it from the repository root with

.. code-block:: bash

   uv run python examples/scripts/gradient/gradient_paulipropagation.py

It installs PauliPropagation.jl and ReverseDiff.jl through juliapkg the first
time.

.. literalinclude:: ../../../examples/scripts/gradient/gradient_paulipropagation.py
   :language: python
