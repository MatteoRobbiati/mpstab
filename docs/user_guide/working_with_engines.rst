Working with Engines
====================

MPSTAB runs on two computational backends (engines): one for the stabilizer
part of the circuit, one for the tensor network.

By default HSMPO uses Stim for the stabilizer part and Quimb for the tensor
network. Engines are passed at construction, or swapped later with
``set_engines``::

    from mpstab import HSMPO
    from mpstab.engines import QuimbEngine
    from qibo import Circuit, gates

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    simulator = HSMPO(ansatz=circuit, tn_engine=QuimbEngine(svd_cutoff=0.0))
    # ...or later, which rebuilds the cached MPS:
    simulator.set_engines(tn_engine=QuimbEngine())

Available Engines
-----------------

Stabilizer Engines
~~~~~~~~~~~~~~~~~~~

Engines that handle the Clifford (stabilizer) part of the circuit:

**Stim Engine**
  - Speed: State-of-art Clifford simulator
  - Coverage: Pure Clifford circuits
  - Best for: High-performance stabilizer simulation
  - External library: `Stim <https://github.com/quantumlib/Stim>`_

::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 1))

    simulator = HSMPO(ansatz=circuit)
    # By default we use Stim

Tensor Network Engines
~~~~~~~~~~~~~~~~~~~~~~

Engines that handle the non-Clifford effects via MPO:

**Quimb Engine**
  - Speed: Fast (optimized contractions)
  - Memory: Efficient for large bond dimensions
  - Best for: Large systems with moderate bond dimensions
  - Backends: ``QuimbEngine(backend="jax")`` or ``"torch"`` for autodiff
  - External library: `Quimb <https://quimb.readthedocs.io/>`_

::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    circuit = Circuit(5)
    circuit.add(gates.H(0))
    circuit.add(gates.RY(0, theta=0.5))

    simulator = HSMPO(ansatz=circuit, max_bond_dimension=32)
    # Uses Quimb tensor network engine by default

**cuTensorNet Engine**
  - Runs on: an NVIDIA GPU, through CuPy and
    `cuTensorNet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/>`_
  - Same algorithms and truncation rule as the Quimb engine, so the same
    results up to roundoff; ``dtype="complex64"`` trades accuracy for speed
  - Meant for: large bond dimensions, where the QR and SVD of the bond-doubled
    windows dominate the run time. At small bond dimensions the cost of each
    GPU call dominates instead and quimb on the CPU is likely faster; the
    crossover has not been benchmarked yet.
  - Not covered: DMRG, ``truncation_fidelity_pure_tn``, the adjoint gradient
    (``gradient(method="flips")`` works), entangled initial states
  - Install: ``pip install "mpstab[cutensornet]"`` (CUDA 12)

.. code-block:: python

    from mpstab import HQSMPO, HSMPO
    from mpstab.engines import CuTensorNetEngine
    from mpstab.models.ansatze import HardwareEfficient

    ansatz = HardwareEfficient(nqubits=40, nlayers=6)
    simulator = HSMPO(
        ansatz=ansatz, max_bond_dimension=256, tn_engine=CuTensorNetEngine()
    )
    print(simulator.expectation("Z" * 40))

    # The head/tail split runs on the GPU too.
    split = HQSMPO(ansatz, tn_engine=CuTensorNetEngine(svd_cutoff=0.0))

The engine runs on CuPy's current device; ``cupy.cuda.Device(1).use()``
selects another GPU.

Native Engines
~~~~~~~~~~~~~~

``NativeStabilizersEngine`` and ``NativeTensorNetworkEngine`` are pure-Python
implementations of the same interfaces, meant to be read: they show what a
stabilizers engine and a tensor-network engine actually do, without stim or
quimb in the way. They give the same results, more slowly, and the native
tensor-network engine supports neither DMRG nor the head/tail split of
``HQSMPO``.

::

    from mpstab import HSMPO
    from mpstab.engines import NativeStabilizersEngine, NativeTensorNetworkEngine
    from mpstab.models.ansatze import HardwareEfficient

    simulator = HSMPO(
        ansatz=HardwareEfficient(nqubits=5, nlayers=2),
        stab_engine=NativeStabilizersEngine(),
        tn_engine=NativeTensorNetworkEngine(),
    )
    print(simulator.expectation("ZZZZZ"))

Engine Configuration
--------------------

::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Basic HSMPO with the default engines
    simulator = HSMPO(
        ansatz=ansatz,
        max_bond_dimension=32
    )

    # Access engine information
    print(f"Stabilizer engine: {simulator.stab_engine}")
    print(f"TN engine: {simulator.tn_engine}")

Writing a Tensor-Network Engine
-------------------------------

A new tensor-network engine describes how it stores states and operators;
the MPS algorithms come from :class:`~mpstab.engines.TensorNetworkEngine`.
It implements:

- ``build_circuit_mps`` and ``pauli_mpo``;
- ``state_sites``: the state as a sequence of ``(left, phys, right)`` arrays
  in mixed canonical form, with a ``center`` attribute and a ``commit()``
  that writes changed sites back;
- ``operator_sites`` / ``operator_from_sites`` / ``operator_site_arrays`` for
  the operator side (the head/tail split of ``HQSMPO``);
- ``linalg``, if its arrays are not NumPy: a :class:`~mpstab.engines.Linalg`
  with the array module and the ``qr`` and ``svd`` the kernels call.

An engine that keeps its own arrays can store its states as
:class:`~mpstab.engines.SiteMPS`, which is already a site view (its
``state_sites`` returns the state itself), and its operators as
:class:`~mpstab.engines.SiteMPO`.

``CuTensorNetEngine`` is a complete example: it implements only these hooks
and inherits every algorithm.

An engine that runs its own algorithms instead, as the native one does, can
skip the site view: it then implements ``expval``, ``pauli_rot`` and
``squared_norm`` itself (and ``conjugate_operator`` for ``HQSMPO``), and the
base class uses them in place of the shared kernels
(``has_site_view`` / ``has_operator_sites`` are then ``False``).

Tips for Efficient Simulation
------------------------------

1. **Use Clifford circuits when possible** - they're orders of magnitude faster
2. **Start with low bond dimensions** - increase only if accuracy is insufficient
3. **Profile your circuits** - understand what gates are expensive

Example: Adaptive Bond Dimension
---------------------------------

::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient
    from qibo import gates

    # Define an ansatz or a Qibo circuit
    ansatz = HardwareEfficient(nqubits=10, nlayers=3)

    # Start with low bond dimension
    for max_bd in [2, 4, 8, 16, 32]:
        simulator = HSMPO(ansatz=ansatz, max_bond_dimension=max_bd)
        result, fidelity = simulator.expectation("Z" * 10, return_fidelity=True)

        print(f"χ_max={max_bd}: result={result:.4f}, fidelity={fidelity:.4f}")

        # Stop if accuracy is sufficient
        if fidelity > 0.99:
            break

Further Reading
---------------

- `Quimb Documentation <https://quimb.readthedocs.io/>`_
- `cuTensorNet Documentation <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/>`_
- `Stim Documentation <https://quantumlib.org/reference/python/stim/>`_
- :doc:`../api/engines`
