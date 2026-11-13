QPU
===

Everything needed to run an :class:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO`
head/tail split on a quantum device. The pipeline is three steps, one
subpackage each, and every circuit is decided classically before anything
reaches a backend.

.. automodule:: mpstab.qpu

Synthesis
---------

Turns a head's dressed rotations into a runnable circuit plus a pure-Clifford
residual.

.. automodule:: mpstab.qpu.synthesis
   :members: resolve_synthesis

.. automodule:: mpstab.qpu.synthesis.abstract
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.synthesis.rustiq
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.synthesis.extraction
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.synthesis.cnot_ladder
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.synthesis.utils
   :members:
   :show-inheritance:

Measurement
-----------

Fixes the circuits and shots that measure the tail-folded observable, and
recombines the measured frequencies into an expectation value with its error
bars. One module per method.

.. automodule:: mpstab.qpu.measurement
   :members: estimate, resolve_method

.. automodule:: mpstab.qpu.measurement.abstract
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.measurement.pauli
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.measurement.shadows
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.measurement.tnice
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.measurement.pauli_expansion
   :members:
   :show-inheritance:

Execution
---------

Runs a plan's circuits on a backend, or exports them as OpenQASM to run on a
device.

.. automodule:: mpstab.qpu.execution

.. automodule:: mpstab.qpu.execution.simulator
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.execution.qasm
   :members:
   :show-inheritance:

.. automodule:: mpstab.qpu.execution.storage
   :members:
   :show-inheritance:
