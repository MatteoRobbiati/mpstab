Head Synthesis by Clifford Extraction
=====================================

``HSynthSMPO`` splits a circuit into a *head*, run on the device, and a
*tail*, folded classically into the observable. The head is a list of dressed
Pauli rotations, and has to become a circuit. This page explains the
``"extraction"`` synthesis, which needs no extra dependency and can be told
the connectivity of the device.

The problem
-----------

A dressed rotation is ``exp(-i theta P / 2)`` with ``P`` a Pauli string, for
instance ``Z Z X I``. A device only rotates single qubits, so the string has
to be brought onto one qubit first.

The textbook circuit does it in three steps:

1. single-qubit gates turn every letter of ``P`` into ``Z``;
2. a chain of CNOTs collects the parity of the support on one qubit: each CNOT
   removes one site, so a string of weight ``w`` needs ``w - 1`` of them;
3. an ``RZ(theta)`` on that qubit, then steps 2 and 1 **undone**.

The undoing costs another ``w - 1`` CNOTs, for every rotation.

The idea: do not undo
---------------------

The gates of steps 1 and 2 are Clifford gates. A Clifford gate can be moved
past a Pauli rotation: the rotation stays a Pauli rotation, only its string
changes. So instead of undoing them, we leave them where they are and
**update the strings of the rotations still to come**.

.. code-block:: text

    textbook      V1  RZ  V1^-1   V2  RZ  V2^-1   V3  RZ  V3^-1
    extraction    V1  RZ          V2' RZ          V3' RZ           (V1 V2' V3')^-1

    V  = the Clifford that brings a string onto one qubit
    V' = the same, for the string as changed by the Cliffords before it

What is left at the end is one accumulated Clifford: the *residual*. It is
never run. A Clifford turns a Pauli observable into another Pauli observable,
so the residual is absorbed classically into what is measured, exactly as the
Clifford part of the original circuit is.

This is the idea of QuCLEAR (Liu et al., `arXiv:2408.13316
<https://arxiv.org/abs/2408.13316>`_): Clifford *extraction* (push them to the
end) and *absorption* (handle them classically).

Which CNOTs
-----------

Bringing a string of weight ``w`` onto one qubit takes ``w - 1`` CNOTs, but
there are many ways to choose them, and each choice changes the coming strings
differently. A good choice makes them lighter, a bad one heavier. This choice
is the whole difference between a short head and a long one.

The rule used here is greedy. For the rotation in turn, among all the pairs
(control, target) in its support, take the CNOT after which the next few
strings touching those two qubits are

- **lightest**: fewer sites, hence fewer CNOTs later;
- **most local**: with a device connectivity, sites that are close on the
  chip, so that later CNOTs need no routing. A CNOT between distant qubits is
  itself charged the SWAPs that would route it.

Two more degrees of freedom are used:

- rotations that commute may be reordered, the lightest first;
- no single setting of the rule is best on every circuit, so a handful are
  tried (:data:`~mpstab.qpu.synthesis.extraction.EXTRACTION_STRATEGIES`) and
  the cheapest head is kept. A synthesis takes a fraction of a second.

Usage
-----

.. code-block:: python

    from mpstab.evolutors.hsynthsmpo import HSynthSMPO
    from mpstab.models.ansatze import ParametricKickedIsing

    hs = HSynthSMPO(
        ansatz=ParametricKickedIsing(nqubits=6, n_steps=3, delta_h=0.2),
        head_synthesis="extraction",   # "rustiq" (default) | "extraction" | "cnot_ladder"
        head_connectivity="line",      # None (all-to-all) | "line" | [(q0, q1), ...]
    )
    cut_index = len(hs.magic_gates) // 2

    head = hs.resynthesize_head(cut_index)
    print(head.method, head.n_two_qubit_gates)   # the circuit is head.circuit

    # every hardware route uses that head and folds its residual
    result = hs.expectation_at_cut("ZZIIII", cut_index, method="pauli", n_shots=20000, seed=0)
    exact = hs.expectation_at_cut("ZZIIII", cut_index, method="simulation").value
    assert abs(result.value - exact) <= 5 * result.total_error + 1e-6

The light cone of the observable
--------------------------------

A second saving does not depend on the synthesis. Going backwards from the
observable, a rotation matters only if its string anticommutes with the
observable, or with a later rotation that matters. Every other rotation
cancels against its own inverse in ``<psi| U^dag O U |psi>``: it can be removed
from the head, and need not be folded in the tail. This is exact, and costs
only commutations of Pauli strings.

.. code-block:: python

    hs.light_cone = True                      # used by every hardware route
    kept = hs.relevant_rotations("IIZIII")    # indices of the rotations that matter
    head = hs.resynthesize_head(cut_index, observable="IIZIII")
    assert len(kept) < len(hs.magic_gates)

It pays for local observables and shallow circuits. For a sum of terms the
head keeps the union of their light cones, so an observable covering the
whole chain (a sum of ``Z Z`` over every bond) removes nothing.

What to expect
--------------

Two-qubit gates after compilation to a line of CZ-native qubits, 16 qubits,
head = 80% of the rotations (in brackets, the whole original circuit). The
last column also drops the rotations outside the light cone of a ``Z`` on the
central qubit:

.. code-block:: text

                                        rustiq  original,  QuCLEAR  extraction  extraction
                                                truncated                       + light cone
    kicked Ising, 8 steps       (240)     303      180       149        81          50
    parametric kicked Ising     (240)     630      180       173       164          92
    Trotter Ising, 8 steps      (240)     413      203       206       188         129
    random brickwork, 6 layers  (390)     954      282       306       399         291

- It pays where the circuit has a Clifford structure to extract (the kicked
  Ising chain near its Clifford point): the head is far shorter than the
  original circuit cut at the same place.
- Where every gate is already a rotation (Trotter), there is nothing to
  extract, and the head is about as long as the original, slightly shorter.
- On a random brickwork the rule is still worse than the original circuit:
  compare the candidates and keep the shortest.
- The light cone helps most on shallow circuits: at 20 qubits and 4 steps of
  the kicked Ising chain, the same head goes from 42 to 10 two-qubit gates.

"Original, truncated" is the original circuit up to the cut, which is itself a
valid head. It is the reference a synthesis has to beat. The light cone could
be applied to it as well: the comparison of the last column with the others
is not like for like.
