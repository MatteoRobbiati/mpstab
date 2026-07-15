Hardware Resynthesis with rustiq
=================================

``HSynthSMPO`` can resynthesize its dressed Pauli rotations into a circuit
made of hardware-native gates, using the low-level `rustiq
<https://github.com/smartiel/rustiq>`_ Pauli-network synthesis API. This is
the circuit you would actually run on a real device; it is never used for the
classical expectation value, which always applies the exact rotations (to the
state MPS, or -- for this feature -- to a statevector for verification).

Install the optional dependency (built from source; it ships no PyPI wheels)::

    pip install "mpstab[rustiq]"

Why "foldable"?
---------------

``rustiq.pauli_network_synthesis`` can synthesize a sequence of Pauli
rotations two ways:

- ``fix_clifford=False``: a circuit that implements the target unitary **up
  to a trailing Clifford** (cheaper);
- ``fix_clifford=True``: the exact circuit, with no trailing ambiguity.

The first is always a gate-for-gate *prefix* of the second, so the exact
circuit is ``head + tail``, where ``tail`` is a **pure-Clifford** correction.
Unlike Qiskit's ``rustiq`` high-level-synthesis plugin (which resynthesizes
the exact and up-to-Clifford circuits independently, leaving a residual that
is generally *not* Clifford), this property lets ``tail`` be reabsorbed
*exactly* into an observable with a Clifford simulator (``stim``, via
``StimEngine``) -- no dense matrices, no tensor-network truncation:

.. code-block:: python

    target = tail . head                 # tail applied last
    <target|O|target> = <head 0| tail^dagger . O . tail | head 0>

so only ``head`` -- the hardware-native circuit -- needs to actually run.

Basic usage
-----------

.. code-block:: python

    from mpstab.evolutors.hsynthsmpo import HSynthSMPO
    from mpstab.models.ansatze import HardwareEfficient

    ansatz = HardwareEfficient(nqubits=6, nlayers=2)

    # rotations_only skips the eager MPS precompute -- only the dressed
    # rotations are needed here, so construction stays O(1) in system size.
    hs = HSynthSMPO.rotations_only(ansatz)

    # Resynthesize ALL dressed rotations and fold the resulting pure-Clifford
    # tail exactly into the observable -- no approximation involved.
    expval = hs.expectation_from_rustiq_fold("Z" * 6)

    # The actual hardware-native circuit you would run:
    head_circuit = hs.foldable_head_circuit(cut_index=len(hs.magic_gates))

Partial cuts: gate-count profiling
-----------------------------------

Resynthesizing only a prefix of the dressed rotations (``cut_index <
len(magic_gates)``) is useful for gate-count profiling -- e.g. to see how
resynthesis cost grows as more of the circuit is included:

.. code-block:: python

    for cut in range(0, len(hs.magic_gates) + 1, 5):
        counts = hs.foldable_head_gate_counts(cut)
        print(counts["n_head_rotations"], counts["synthesized_head_2q_gates"])

A partial cut is **not** combined with an exact expectation value: the
rotations left out of the resynthesis are still non-Clifford, and their
Heisenberg conjugation is generally not a single Pauli -- which is exactly
why the (approximate) MPO-tail path exists, see
:doc:`fidelity_and_approximation` and ``HSynthSMPO.mpo_tail_approximation``.

See ``examples/hsynthsmpo_rustiq_fold.py`` for a complete, runnable script.
