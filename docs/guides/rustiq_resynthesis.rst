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
    hs = HSynthSMPO(ansatz)

    # Resynthesize ALL dressed rotations into a hardware-native circuit, and
    # measure it with a finite shot budget -- there is no exact route.
    result = hs.expectation_at_cut("Z" * 6, cut_index=len(hs.magic_gates), n_shots=20000)

    # The actual hardware-native circuit that gets run:
    resynth = hs.resynthesize_head(cut_index=len(hs.magic_gates))
    head_circuit = resynth.circuit

Partial cuts: gate-count profiling
-----------------------------------

Resynthesizing only a prefix of the dressed rotations (``cut_index <
len(magic_gates)``) is useful for gate-count profiling -- e.g. to see how
resynthesis cost grows as more of the circuit is included:

.. code-block:: python

    for cut in range(0, len(hs.magic_gates) + 1, 5):
        resynth = hs.resynthesize_head(cut)
        print(resynth.n_gates, resynth.n_two_qubit_gates)

A partial cut leaves a pure-Clifford residual (``resynth.tail_tableau``): the
``"pauli"`` route of :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.expectation_at_cut`
folds it into the sampled Pauli terms automatically before grouping (see the
module docstring of :mod:`mpstab.evolutors.hsynthsmpo`); the ``"shadows"``
route instead raises unless ``tail_handling="append"`` is passed, since
folding it into the tail MPO would destroy the product structure its
contraction relies on. Rotations left out of the resynthesis entirely are
still non-Clifford and get folded approximately into the tail MPO -- see
:doc:`fidelity_and_approximation` and ``HSynthSMPO.tail_truncation``.
