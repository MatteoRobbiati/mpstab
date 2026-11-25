Qibo tools
==========

Tools connecting mpstab and `qibo <https://qibo.science/>`_, in both
directions.

Provided backends
-----------------

What mpstab gives qibo: mpstab as a qibo backend, see
:doc:`../user_guide/qibo_backend_provider`.

.. automodule:: mpstab.qibo_tools.provided_backends.mpstab_backend
   :members: MPStabBackend, MetaBackend

Utilized backends
-----------------

What mpstab gets from qibo: its simulators, running the circuits of a
measurement plan in place of a device. On a device, run the plan's circuits
there instead.

.. automodule:: mpstab.qibo_tools.utilized_backends.statevector_exact_simulator
   :members: StatevectorExactSimulator

.. automodule:: mpstab.qibo_tools.utilized_backends.density_matrix_noise_simulator
   :members: DensityMatrixNoiseSimulator
