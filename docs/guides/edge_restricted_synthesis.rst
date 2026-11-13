Head Synthesis on the Device Couplers
=====================================

This page documents two more head synthesisers, ``"steiner"`` and
``"edge_greedy"``, whose CNOTs are all on the device couplers. It assumes the
:doc:`clifford_extraction_synthesis` guide, and spells out where the new
synthesisers agree with extraction and where they differ.

The idea they share with extraction
-----------------------------------

All four non-trivial synthesisers of :mod:`mpstab.qpu.synthesis` (rustiq,
extraction, steiner, edge_greedy) do the same thing at the top level: build a
Clifford *skeleton* that brings every dressed rotation onto one qubit, never
undo it, place the single-qubit rotations into it
(:func:`~mpstab.qpu.synthesis.utils.place_rotations`, shared) and leave the
inverse of the skeleton as the *residual*, absorbed into the observable. This
is QuCLEAR's extraction and absorption (Liu et al., `arXiv:2408.13316
<https://arxiv.org/abs/2408.13316>`_). They only differ in **which Clifford
gates make up the skeleton**.

The difference: every CNOT on a coupler
---------------------------------------

Rustiq ignores the device: its CNOTs can join any two qubits, and a router
adds SWAPs afterwards. Extraction is told the device connectivity, but still
picks its CNOTs between two sites of the rotation's own support: a pair that
is far apart on the chip is *charged* the SWAPs a router would add
(``1 + 3 (d - 1)``), so it is avoided when possible, but it can be chosen, and
then the head needs routing.

``steiner`` and ``edge_greedy`` only ever emit a CNOT **on a coupler**. Their
head needs no routing at all: compiling it is unrolling to the native gates,
with the placement the connectivity was given in.

What makes this possible is that the residual is free. A CNOT may move a
string's ``Z`` onto a qubit **outside its support**, a Steiner node between
two distant sites, and nothing has to undo it on the device. Extraction never
does this; it is the main algorithmic difference.

``steiner``
~~~~~~~~~~~

:class:`~mpstab.qpu.synthesis.steiner.SteinerSynthesis`. One rotation at a
time, in circuit order:

1. turn every site of the string to ``Z`` (``H`` or ``SqrtX``);
2. build an approximate Steiner tree of its support on the coupling graph
   (Kou, Markowsky and Berman: minimum spanning tree of shortest-path
   distances, expanded into paths, pruned; :meth:`CouplingGraph.steiner_tree
   <mpstab.qpu.synthesis.coupling.CouplingGraph.steiner_tree>`);
3. collect the parity from the leaves to a root, one CNOT per tree edge; on a
   tree node the string does not touch, one more CNOT first moves the ``Z``
   there.

No look-ahead and no cost function: the tree is the cost. It is the
coupler-restricted analogue of one extraction step without its choice rule.

``edge_greedy``
~~~~~~~~~~~~~~~

:class:`~mpstab.qpu.synthesis.edge_greedy.EdgeGreedySynthesis`. Rustiq's way
of building a skeleton (one gate at a time, each chosen for its effect on the
pending strings), restricted to the couplers:

- the candidates are every coupler in both orientations, each with one of the
  nine local basis changes ``{-, H, SqrtX}`` on its two qubits;
- the cost of a string is the size of the Steiner tree joining its support on
  the device (:meth:`CouplingGraph.tree_cost
  <mpstab.qpu.synthesis.coupling.CouplingGraph.tree_cost>`): the CNOTs it
  still needs, with routing already counted;
- a candidate is judged on all the rotations of the *front* (those that can
  be placed next, :class:`~mpstab.qpu.synthesis.coupling.PauliFrame`) with
  weight 1, plus the next ``lookahead`` pending ones with weight
  ``lam * gamma**k``;
- the best gate that lowers the cost is applied; on a plateau one ``steiner``
  step reduces the cheapest front rotation, so the loop terminates.

Side by side
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``extraction``
     - ``steiner``
     - ``edge_greedy``
   * - unit of work
     - one rotation at a time (a commuting one may overtake, cheapest first)
     - one rotation at a time, in order
     - one gate at a time, for all the front rotations at once
   * - basis change
     - fixed: every site to ``Z`` first
     - fixed: every site to ``Z`` first
     - chosen with each CNOT, nine options per coupler
   * - CNOT candidates
     - pairs inside the rotation's support, any distance
     - the edges of a Steiner tree of the support, Steiner nodes included
     - every coupler touching a front string
   * - cost
     - weight (and spread) of the next strings touching the pair, discounted;
       a distant pair charged its SWAPs
     - none, the tree
     - Steiner-tree size of front strings, plus discounted look-ahead
   * - CNOTs off the couplers
     - possible, then routed
     - never
     - never
   * - settings tried
     - 5 (:data:`~mpstab.qpu.synthesis.extraction.EXTRACTION_STRATEGIES`),
       cheapest kept
     - 1
     - 1 (``lookahead=10, gamma=0.7, lam=0.5``, not tuned)
   * - residual
     - inverse of the skeleton (``H``, ``SqrtX`` -> ``SqrtXd``, ``CNOT``)
     - same
     - same

What is shared in code: the gate-list format, ``place_rotations``,
``CLIFFORD_TABLEAUS`` and the convention ``target = residual . head``; the
connectivity argument (``None``, ``"line"`` or a list of edges in the qubit
numbering of the strings) is the same as extraction's, and
:class:`~mpstab.qpu.synthesis.coupling.CouplingGraph` reuses
:func:`~mpstab.qpu.synthesis.extraction.connectivity_distances`.

Placement is the caller's job: the connectivity is given in the numbering of
the strings. On a grid, number the qubits along a snake path, so that
consecutive qubits of a chain are neighbours (``examples/compare_head_syntheses.py``
has a ``snake_grid``).

Usage
-----

.. code-block:: python

    from mpstab.evolutors.hsynthsmpo import HSynthSMPO
    from mpstab.models.ansatze import ParametricKickedIsing
    from mpstab.qpu import EdgeGreedySynthesis, SteinerSynthesis

    n = 8
    edges = [(q, q + 1) for q in range(n - 1)]   # the device, in the strings' numbering
    hs = HSynthSMPO(
        ansatz=ParametricKickedIsing(nqubits=n, n_steps=4, delta_h=0.2),
        head_synthesis=EdgeGreedySynthesis(connectivity=edges),   # or "edge_greedy"
    )
    cut_index = len(hs.magic_gates) // 2
    head = hs.resynthesize_head(cut_index)        # every CNOT of head.circuit on an edge
    other = hs.resynthesize_head(cut_index, synthesis=SteinerSynthesis(connectivity=edges))

Every route of :meth:`~mpstab.evolutors.hsynthsmpo.HSynthSMPO.expectation_at_cut`
uses the head and folds its residual as for the other synthesisers.

No synthesis wins on every circuit (see below): with a compiler for the
device, synthesise with several, compile each head, and keep the cheapest.
mpstab leaves that choice to the caller.

How they are tested
-------------------

``tests/test_edge_restricted_synthesis.py`` checks, on random rotations and
three connectivities, that ``residual . head`` is the product of the rotations
on the state, that every two-qubit gate is on a coupler, and that the folded
observable on the head gives the exact value. It also checks it at the Pauli
level, with no state vector: every rotation of the head is pushed through the
Clifford gates before it, and the result must be the given rotations (in order
up to swaps of commuting neighbours, with their angles) while the residual
undoes the head's Clifford. The same check passes for rustiq, extraction and
the CNOT ladder.

What to expect
--------------

``examples/compare_head_syntheses.py``: 16 qubits, 8 steps, head = 80% of the
rotations, every head compiled by Qiskit at optimization level 3 to R + CZ on
the device. "Original" is the original circuit up to the same cut, the head
with no resynthesis (computed in the example). The original, rustiq and
extraction heads are laid out and routed by Qiskit; the coupler-restricted
heads keep their placement. CZ / depth counting CZ only:

.. code-block:: text

    line                       original   rustiq     extraction  steiner    edge_greedy
    kicked Ising                90 / 12   265 / 134   81 / 23    106 / 29    81 / 29
    parametric kicked Ising     90 / 12   526 / 266  153 / 56    248 / 63    82 / 35
    Trotter Ising              203 / 50   346 / 163  186 / 43    188 / 39   199 / 74
    brickwork                  240 / 39   716 / 277  247 / 46    841 / 347  347 / 154

    grid (snake placement)
    kicked Ising                90 / 12   200 / 102   81 / 23    221 / 86    81 / 29
    parametric kicked Ising     90 / 12   348 / 190  185 / 67    234 / 103   82 / 35
    Trotter Ising              203 / 50   312 / 146  191 / 47    188 / 39   232 / 71
    brickwork                   72 / 9    407 / 172   72 / 9    2252 / 764  128 / 61

Where they agree, and where they do not:

- **kicked Ising**: extraction and edge_greedy reach the same 81 CNOTs on both
  devices. Extraction's CNOTs happen to be nearest-neighbour here, so being
  restricted to couplers costs nothing and gains nothing.
- **parametric kicked Ising** (a Clifford circuit plus small deviations, whose
  dressed strings are heavy): edge_greedy needs about half of extraction's
  CZs (82 against 153 on the line, 185 on the grid), the case where routing
  extraction's distant CNOTs costs most.
- **Trotter Ising**: there is little Clifford structure to extract; extraction
  and steiner are slightly below the original, edge_greedy is not.
- **random brickwork**: the original circuit is best; steiner is very poor
  (no look-ahead on high-weight strings) and is only worth trying among
  others.
- **rustiq** is the most expensive everywhere once routed.
- Against the **original circuit compiled by Qiskit**, the best head has about
  the same number of CZs on the Ising circuits (81-82 against 90) but a two to
  three times larger CZ depth: the skeleton is built gate by gate, with no
  goal of parallel layers. A depth-aware cost is the obvious next step.

Notes from the hybrex experiments
---------------------------------

Collected while developing these synthesisers (hybrex, 10 to 30 qubits, line,
grid and heavy-hex, three seeds, Qiskit O3 and qibo); they are context, not
part of the tests.

- **Compiler matters for the reference.** Qiskit O3 compiles the original
  kicked Ising circuit to half the CZs qibo's unroller gives (it re-synthesises
  ``CNOT RZ CNOT`` blocks). Comparisons against the original circuit should use
  a strong compiler, or they flatter every synthesis.
- **Head against the original prefix (Qiskit O3, median over sizes and
  seeds).** Kicked and parametric kicked Ising: the cheapest synthesis ties in
  CZs (ratio 1.00-1.04) with a 3-6x larger CZ depth. Trotter Ising: ties in
  CZs with 22-36% less CZ depth. Brickwork: 1.2-1.7x more CZs. Rustiq: 1.2-6x
  more CZs and 6-25x more CZ depth. On circuits that are already
  hardware-native, resynthesis beats rustiq but not the original circuit; the
  room for it is in circuits that are not (long-range or 2D rotations).
- **Where the residual is folded matters, and not because of the synthesis.**
  The ``"pauli"`` route folds the tail into an MPO at the bond cap and applies
  the residual to the sampled strings afterwards. Passing the residual inside
  the fold instead (``build_post_head_mpo(..., residual=...)``), before the
  truncation, changes the truncation bias. With rustiq's residual it is far
  worse (biases of order one where the usual route is below 1e-6). With a
  local residual it is often far better, e.g. parametric kicked Ising, 10
  qubits, 6 steps, chi 4, 75% cut: 6.5e-3 the usual way, 1.3e-3 with
  edge_greedy's residual folded in, 2e-15 with the residual of the original
  circuit up to the cut (the inverse of its Clifford gates). Not
  always: on kicked Ising at chi 4 and a 50% cut it is worse (1.2 against
  0.41). Over 72 cases (three ansatze, 4-8 steps, chi 4 and 8, two
  observables, two cuts) the original circuit's residual is as good as
  edge_greedy's within a few percent in most and clearly better in several:
  the gain comes from folding in a *local* Clifford frame, not from the
  synthesis. Folding the residual in first, when the estimated fold fidelity
  says so, is a candidate improvement of the tail fold for every head.
- **The equivalence under noise.** With depolarising noise on the device
  (density matrix, 10 qubits), the edge_greedy head and the original circuit
  up to the cut give hybrid estimates within 10-20% of each other in bias, at
  equal CZ count; the original has the lower depth.

Limitations
-----------

- The edge_greedy parameters are untuned; its cost ignores depth.
- Steiner trees are approximate (within twice the optimum), and steiner has no
  look-ahead: it is a fallback and one candidate among others, not a default.
- Choosing the cheapest head among syntheses, after compiling them, is left
  to the caller (hybrex does it).
- Placement (which physical qubit each string index sits on) is not chosen
  here.
