Engines
=======

Engines are the computational backends that perform low-level state operations. Each kind has an abstract interface, so a custom engine can be swapped in.

**Two Engine Types**:

1. **Stabilizer Engines**: Handle Clifford gates efficiently
2. **Tensor Network Engines**: Manage non-Clifford effects via MPO

Stabilizer Engine
-----------------

.. automodule:: mpstab.engines.stabilizers.stim
   :members:
   :undoc-members:
   :show-inheritance:

**Stim Engine**: Clifford backpropagation of Pauli observables with the C++
`Stim <https://github.com/quantumlib/Stim>`_ simulator.

Tensor Network Engine
---------------------

.. automodule:: mpstab.engines.tensor_networks.quimb
   :members:
   :undoc-members:
   :show-inheritance:

**Quimb Engine**: MPS states and MPO operators as
`quimb <https://quimb.readthedocs.io/>`_ objects, on the numpy, jax or torch
backend. On numpy it runs the shared algorithms of
:class:`~mpstab.engines.TensorNetworkEngine`; on jax and torch, quimb's own
differentiable contractions. Its MPS also feeds quimb's DMRG.

.. autoclass:: mpstab.engines.Linalg
   :members:

.. autoclass:: mpstab.engines.SiteMPS
   :members:

.. autoclass:: mpstab.engines.SiteMPO
   :members:

**Building blocks for a new engine**: the array backend the shared kernels run
on, and a ready-made site-array state and operator.

.. automodule:: mpstab.engines.tensor_networks.cutensornet
   :members: CuTensorNetEngine, CuTensorNetLinalg, CuQuantumAPI
   :show-inheritance:

**cuTensorNet Engine**: the same algorithms on an NVIDIA GPU, with CuPy site
arrays and the QR, SVD and network contractions of NVIDIA's
`cuTensorNet <https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/>`_.
Install with ``pip install "mpstab[cutensornet]"``. No DMRG and no adjoint
gradient (``method="flips"`` works), and initial states
must be product states.

Native Engines
--------------

Pure-Python implementations of the two interfaces, kept as readable examples
of what an engine does rather than for performance. They give the same
expectation values as stim and quimb, but are slower and cover less: the
native tensor-network engine has no operator-side conjugation (so no head/tail
split in :class:`~mpstab.evolutors.hqsmpo.HQSMPO`) and no DMRG.

.. automodule:: mpstab.engines.stabilizers.native.engine
   :members:
   :show-inheritance:

**Native Stabilizers Engine**: backpropagates a Pauli string by composing one
tableau per gate, on the XZ-encoded Pauli arithmetic of
:mod:`mpstab.engines.stabilizers.native.pauli_string`.

.. automodule:: mpstab.engines.tensor_networks.native.engine
   :members:
   :show-inheritance:

**Native Tensor Network Engine**: an MPS in Vidal canonical form on a small
in-package tensor-network layer
(:mod:`mpstab.engines.tensor_networks.native.circuit_mps`), evolved by
applying MPOs and re-splitting by SVD.

Selecting Engines
-----------------

Engines are automatically selected based on your circuit type and parameters::

    from qibo import Circuit, gates
    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    clifford_circuit = Circuit(nqubits=10)
    # Add only Clifford gates
    for q in range(10):
         clifford_circuit.add(gates.H(q))
         clifford_circuit.add(gates.CNOT(q0=q%10, q1=(q+1)%10))

    # Clifford circuit - uses stabilizer engine only
    simulator = HSMPO(ansatz=clifford_circuit)

    # Constructing a circuit with also magic gates
    mixed_circuit = HardwareEfficient(
         nqubits=10,
         nlayers=3,
    )

    # Mixed circuit with bond limit - uses both engines
    simulator = HSMPO(
        ansatz=mixed_circuit,
        max_bond_dimension=32  # Enables tensor network engine
    )

For detailed guidance, see :doc:`../user_guide/working_with_engines`

Engine Interface
----------------

All engines implement standardized interfaces for compatibility.

Base Engine Classes::

    from mpstab.engines import StabilizersEngine, TensorNetworkEngine

The evolutors never look inside an engine's states or operators: everything
they need goes through these methods. ``tests/test_engine_agnostic.py``
enforces this with an engine whose states and operators expose nothing.

A tensor-network engine splits into a *representation*, which is
engine-specific, and the *algorithms*, which are implemented once in
:class:`~mpstab.engines.TensorNetworkEngine` as concrete methods. The
algorithms (windowed Pauli rotations, expectation values, operator
conjugation, transition amplitudes) are the kernels of
``mpstab.engines.tensor_networks._mps_kernels``. They run on the site arrays
the representation exposes, with the array module and factorizations of the
engine's :attr:`~mpstab.engines.TensorNetworkEngine.linalg`: NumPy and LAPACK
for quimb, CuPy and cuTensorNet for the GPU engine. Every engine therefore
truncates with the same ``"rsum2"`` rule
(:attr:`~mpstab.engines.TensorNetworkEngine.svd_cutoff`).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``TensorNetworkEngine`` methods
     - Role
   * - ``build_circuit_mps``, ``pauli_mpo``
     - Abstract: build a state and a Pauli-string operator.
   * - ``state_sites``, ``state_from_sites``
     - Representation: the state as ``(left, phys, right)`` arrays in mixed
       canonical form, which every state-side algorithm runs on, and back
       from such arrays.
   * - ``operator_sites``, ``operator_from_sites``, ``operator_site_arrays``,
       ``pauli_string_of``
     - Representation: operators as ``(left, 4, right)`` arrays and back, and
       out of the engine as numpy ``(left, right, out, in)`` arrays (the
       head/tail split of ``HQSMPO`` and its measurement routes).
   * - ``linalg``, ``copy_state``, ``bond_dimension``
     - Representation, with defaults: NumPy arrays, a deep copy, no bond
       report.
   * - ``svd_cutoff``, ``truncation_cutoff``
     - The ``"rsum2"`` cutoff every shared algorithm truncates with
       (``None`` keeps quimb's default, ``1e-10``).
   * - ``process_start_method``, ``worker_context``
     - How the processes of ``noisy_expectation`` and ``gradient`` start:
       ``"spawn"`` for an engine holding a GPU context.
   * - ``has_site_view``, ``has_operator_sites``, ``has_pauli_sum``
     - Whether the shared algorithms apply; by default, whether the hooks
       above are implemented. An engine without them keeps its own
       ``expval`` and ``conjugate_operator``. With a site view and Pauli sums,
       the adjoint gradient runs fused on the shared kernels.
   * - ``expval``, ``pauli_expvals``, ``pauli_rot``, ``squared_norm``,
       ``pauli_transitions``, ``rotation_operator``, ``apply_rotation``,
       ``conjugate_operator``, ``conjugate_operator_many``, ``site_arrays``
     - Concrete, shared: implemented on the site view. An engine may still
       override them, as ``QuimbEngine`` does on jax and torch to stay
       differentiable, and the native engine does to run its own Vidal-form
       algorithms.
   * - ``apply_pauli_sum``
     - Optional: the adjoint gradient (``HSMPO.gradient(method="adjoint")``);
       ``method="flips"`` does without it. ``QuimbEngine`` only.
   * - ``pauli_sum_mpo``, ``minimize_energy``
     - Optional: DMRG (``HSMPO.minimize_expectation``). ``QuimbEngine`` only.
   * - ``circuit_truncation_fidelity``
     - Optional: ``HSMPO.truncation_fidelity_pure_tn``. ``QuimbEngine`` only.

An optional method an engine does not implement raises ``NotImplementedError``
only when the feature that needs it is used.

A stabilizers engine is where every Clifford circuit is simulated: the Clifford
part of the circuit, and the residual a head resynthesis leaves behind. The
evolutors, the noise compiler and the measurement routes only ask it for
backpropagated Pauli strings, so ``stim`` is imported by ``StimEngine`` alone.
Clifford circuits travel as qibo circuits (e.g.
:attr:`~mpstab.evolutors.hqsmpo.ResynthesizedHead.residual_circuit`), and
signed Pauli products are plain Pauli algebra
(:func:`mpstab.pauli.pauli_product`).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ``StabilizersEngine`` methods
     - Needed for
   * - ``backpropagate``
     - Required: dressing observables with the Clifford circuit.
   * - ``backpropagate_many``
     - Optional, with a generic default an engine may override to share work
       across observables.
   * - ``backpropagate_at``
     - Dressing each magic gate and noise event with the Clifford gates before
       it. The default replays each prefix through ``backpropagate``;
       ``StimEngine`` overrides it with a single tableau pass.
   * - ``clifford_images``, ``is_identity``
     - Concrete, built on ``backpropagate_many``: comparing Clifford circuits,
       e.g. checking whether the resynthesis residual of ``HQSMPO`` is
       trivial.

See Also
--------

- :doc:`../user_guide/working_with_engines` - Engine selection guide
- :doc:`../tutorials/introduction` - Practical examples
