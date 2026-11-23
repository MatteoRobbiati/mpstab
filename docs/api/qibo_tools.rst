Qibo tools
==========

Tools connecting mpstab and `qibo <https://qibo.science/>`_, in both
directions.

Provided backends
-----------------

What mpstab gives qibo: mpstab as a qibo backend, see
:doc:`../guides/qibo_backend_provider`.

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

Migration
---------

These modules moved in this release. The old paths still import, as aliases
of the new objects, and emit a ``DeprecationWarning`` naming the new path.

.. TODO: name the version that removes these aliases.

.. list-table::
   :header-rows: 1

   * - Old path
     - New path
   * - ``mpstab.qibo_backend``
     - :mod:`mpstab.qibo_tools.provided_backends`
   * - ``mpstab.qibo_backend.mpstab``
     - :mod:`mpstab.qibo_tools.provided_backends.mpstab_backend`
   * - ``mpstab.qpu.QiboSimulator``, ``mpstab.qpu.execution.QiboSimulator``,
       ``mpstab.qpu.execution.simulator.QiboSimulator``
     - :class:`mpstab.qibo_tools.StatevectorExactSimulator`
   * - ``mpstab.qpu.execution.simulator.execute_plan``
     - :func:`mpstab.qpu.execution.execute.execute_plan`
