DMRG: pure quimb against HSMPO
==============================

Ground-state search for a local Ising-like Hamiltonian made highly entangling
by a deep Clifford scrambler. The script runs quimb's DMRG directly and on the
HSMPO surrogate, where the Clifford part is absorbed by the stabilizer frame,
and prints the two energies. See also :doc:`../../guides/dmrg_optimization`.

Run it from the repository root with

.. code-block:: bash

   uv run python examples/scripts/dmrg/dmrg_j1j2_comparison.py

.. literalinclude:: ../../../examples/scripts/dmrg/dmrg_j1j2_comparison.py
   :language: python
