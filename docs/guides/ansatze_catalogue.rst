Ansatz Catalogue
================

This page lists every ansatz in :mod:`mpstab.models.ansatze`. For each one it
gives the physical system or algorithm the circuit models, what the circuit
does, and which parameters set its two resources:

- **magic**: how far the circuit is from a Clifford circuit, i.e. how many
  magic gates HSMPO keeps and how large their angles are. It sets the cost of
  the tensor-network part of HSMPO.
- **entanglement**: which qubits interact and for how long, i.e. how fast the
  bond dimension of the state can grow.

The same information is available in code. Every ansatz class has two tuples,
``magic_parameters`` and ``entanglement_parameters``, with the names of the
constructor fields that act as magic and entanglement knobs::

    from mpstab.models.ansatze import RampedKickedIsing

    RampedKickedIsing.magic_parameters
    # ('delta_h', 'delta_j', 'delta_z', 'delta_h_last', 'delta_z_last', 'disorder')
    RampedKickedIsing.entanglement_parameters
    # ('n_steps', 'clifford_h', 'clifford_j', 'graph', 'bond_density')

A field may be in both lists. A generic rotation angle makes a magic gate and,
through the couplings, also changes the entangling power. The
:class:`~mpstab.models.ansatze.ParametricKickedIsing` family is built so that
the two groups of knobs are separate.

Each section shows a short piece of the circuit the class builds, on a few
qubits and for one step or layer. Clifford gates are yellow and magic gates
blue; CNOT, CZ and SWAP have their usual symbols. Brackets above the wires
mark the parts of the circuit, and angles set by a constructor field carry the
field's symbol (the colour legend is under the first figure). The figures are
drawn by ``docs/scripts/draw_ansatze.py``.

Summary
-------

.. list-table::
   :header-rows: 1
   :widths: 24 30 23 23

   * - Ansatz
     - Physical picture
     - Magic knobs
     - Entanglement knobs
   * - ``HardwareEfficient``
     - Generic variational circuit
     - ``nlayers`` (random angles)
     - ``nlayers``, ``entangling``, ``graph``
   * - ``HardwareEfficientBrickwork``
     - Brickwork variational circuit
     - ``nlayers`` (random angles)
     - ``nlayers``, ``graph``
   * - ``HammingWeightPreserving``
     - Particle-number-conserving circuit
     - ``nlayers`` (random angles)
     - ``nlayers``, ``graph``
   * - ``FloquetAnsatz``
     - Floquet echo of a kicked chain
     - ``b``, ``theta``
     - ``nlayers``, ``b``, ``graph``
   * - ``KickedIsing``
     - Kicked Ising chain
     - ``theta_h``, ``theta_z``, ``theta_j``
     - ``n_steps``, ``theta_h``, ``theta_j``, ``graph``
   * - ``ParametricKickedIsing``
     - Kicked Ising chain near a Clifford point
     - ``delta_h``, ``delta_j``, ``delta_z``
     - ``n_steps``, ``clifford_h``, ``clifford_j``, ``graph``
   * - ``RampedKickedIsing``
     - Kicked Ising chain with a time-dependent drive
     - as above, plus ``delta_h_last``, ``delta_z_last``, ``disorder``
     - as above, plus ``bond_density``
   * - ``StarKickedIsing``
     - Kicked central-spin model
     - as ``RampedKickedIsing``
     - as ``RampedKickedIsing``
   * - ``TrotterIsing``
     - Transverse-field Ising time evolution
     - ``dt``, ``J``, ``h``
     - ``n_steps``, ``dt``, ``J``, ``graph``
   * - ``QFT``
     - Quantum Fourier transform
     - none (fixed)
     - none (fixed)
   * - ``QFTPhaseKernel``
     - QFT followed by a phase kernel
     - ``coeffs``
     - none (fixed)
   * - ``Grover``
     - Grover search
     - none (fixed)
     - ``n_iterations``
   * - ``QPE``
     - Quantum phase estimation
     - ``phase``
     - ``n_counting``
   * - ``QAE``
     - Quantum amplitude estimation
     - ``theta_A``
     - ``n_counting``
   * - ``CircuitAnsatz``, ``TranspiledAnsatz``
     - Any user circuit
     - set by the circuit
     - set by the circuit

``nqubits`` scales both resources for every ansatz and is not listed.

Coupling topology
-----------------

Every ansatz with two-qubit couplings takes a ``graph`` field: the pairs of
qubits it couples, as anything
:meth:`.Topology.coerce` accepts — a
name (``"chain"``, ``"ring"``, ``"star"``, ``"grid"``, ``"grid:RxC"``,
``"complete"``), a list of edges, a ``networkx`` or ``rustworkx`` graph, or a
:class:`.Topology`. The entangling gates go on its
edges, layer of disjoint bonds after layer; the single-qubit part of each
ansatz is unchanged.

``graph`` says where the ansatz puts its gates. To run the circuit on a
device without routing, every pair of ``graph`` must be coupled on the chip
(:meth:`.Topology.is_subgraph_of`): the simplest way is to build the ansatz on
the coupling map of the device, or on part of it. A device whose qubits couple
only through a shared resonator has no qubit-qubit couplings: use
``graph="star"``, with a qubit standing for the hub (``StarKickedIsing``)::

    from mpstab.models.ansatze import HardwareEfficient, RampedKickedIsing
    from mpstab import Topology

    RampedKickedIsing(nqubits=20, n_steps=6, delta_h=0.4, graph="grid")  # 4 x 5

    # a device coupling map lists each pair in both directions
    coupling_map = [(0, 1), (1, 0), (1, 2), (2, 1), (1, 3), (3, 1)]
    device = Topology.from_device(coupling_map)
    HardwareEfficient(nqubits=device.nqubits, nlayers=3, graph=device)

The graph is an entanglement knob at fixed magic: at the Clifford point of a
kicked Ising ansatz, one step prepares the graph state of ``graph`` up to local
Cliffords. The defaults keep each ansatz's historical pattern, gate for gate:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Ansatz
     - ``graph`` default
   * - kicked Ising family, ``FloquetAnsatz``
     - ``"chain"`` (``StarKickedIsing``: ``"star"``)
   * - ``HardwareEfficient``
     - ``None``: the ring ``CZ(q, q+1 mod n)`` in qubit order
   * - ``HardwareEfficientBrickwork``
     - ``None``: even bonds, then odd bonds plus ``CZ(0, n-1)``
   * - ``HammingWeightPreserving``
     - ``None``: RBS gates at distances 1 and 2, periodic
   * - ``TrotterIsing``
     - ``None``: the open chain, bonds in order

The algorithm ansätze (``QFT``, ``Grover``, ``QPE``, ``QAE``) have no coupling
pattern to choose; their ``connectivity`` field, used to route the transpiled
circuit, accepts the same inputs as ``graph``.

Class hierarchy
---------------

::

    Ansatz
    ├── HardwareEfficient
    ├── HardwareEfficientBrickwork
    ├── HammingWeightPreserving
    ├── FloquetAnsatz
    ├── CircuitAnsatz
    ├── _KickedIsingBase              step loop: kicks, then couplings layer by layer
    │   ├── KickedIsing               whole angles as gates
    │   └── ParametricKickedIsing     Clifford angle + deviation
    │       └── RampedKickedIsing     deviations ramped over the steps, disorder, dilution
    │           └── StarKickedIsing   graph="star" by default
    └── _CompiledAnsatz               build a high-level circuit, transpile it
        ├── TranspiledAnsatz
        ├── QFT, QFTPhaseKernel
        ├── Grover, QPE, QAE
        └── TrotterIsing

The kicked Ising ansätze share the loop of ``_KickedIsingBase``, which couples
the edges of ``graph`` layer by layer. A subclass only says how a qubit is
kicked (``_kick``), how a bond is coupled (``_coupling``) and whether a bond is
kept (``_keep_bond``). ``StarKickedIsing`` only changes the default ``graph``. The circuit-library ansätze share ``_CompiledAnsatz``:
they only implement ``_build_circuit``, and the transpilation to native gates
is done once in the base class.

Variational circuits
--------------------

HardwareEfficient
~~~~~~~~~~~~~~~~~

The standard ansatz of variational algorithms on NISQ devices. Each of the
``nlayers`` layers applies an ``RY`` rotation to every qubit, then a ring of
``CZ`` gates between neighbours ``(q, q+1 mod n)``. The angles are drawn
uniformly in :math:`[-\pi, \pi)` from numpy's global generator, so a generic
rotation is a magic gate. There are ``nqubits * nlayers`` magic gates, which
``nlayers`` sets. With ``entangling=False`` the circuit makes a product state.

.. themed-figure:: ../_static/ansatze/hardware_efficient.svg
   :align: center

   ``HardwareEfficient(nqubits=4, nlayers=1)``: one layer.

.. themed-figure:: ../_static/ansatze/legend.svg
   :align: center

HardwareEfficientBrickwork
~~~~~~~~~~~~~~~~~~~~~~~~~~

A denser variational circuit. Each layer has two sublayers of random ``RY`` and
``RZ`` rotations. The first is followed by ``CZ`` gates on the even bonds, the
second by ``CZ`` gates on the odd bonds plus a ``CZ(0, n-1)`` that closes the
ring and creates long-range entanglement. A final layer of ``RY`` rotations
ends the circuit. ``nlayers`` sets both the number of random rotations (magic)
and the depth of the brickwork (entanglement).

.. themed-figure:: ../_static/ansatze/hardware_efficient_brickwork.svg
   :align: center

   ``HardwareEfficientBrickwork(nqubits=4, nlayers=1)``: the whole circuit.

HammingWeightPreserving
~~~~~~~~~~~~~~~~~~~~~~~

A particle-number-conserving circuit, as used for fermionic systems in quantum
chemistry. It starts from the half-filling state
:math:`|1\dots10\dots0\rangle`. Each layer then applies reconfigurable beam
splitter (RBS) gates, Givens rotations in the :math:`\{|01\rangle, |10\rangle\}`
subspace of a pair, between nearest and next-nearest neighbours on a ring.
Every RBS is decomposed into ``CNOT`` gates and two ``RY`` rotations of random
angle :math:`\pm\theta`, the magic gates. The state stays in the
sector with Hamming weight ``nqubits // 2``. ``nqubits`` must be even.

.. themed-figure:: ../_static/ansatze/hamming_weight_preserving.svg
   :align: center

   ``HammingWeightPreserving(nqubits=4)``: the half-filling state and the
   first two RBS gates of the layer.

Floquet and kicked dynamics
---------------------------

FloquetAnsatz
~~~~~~~~~~~~~

A Floquet echo, the circuit of a Loschmidt-echo or out-of-time-order
experiment:

.. math::
    U = (F^{t})^\dagger \, R_Z(\theta)_q \, F^{t} \, H_q,

where :math:`F` is one Floquet layer of a kicked chain: on each bond,
:math:`R_Z(\pi/4)` and :math:`R_X(b)` on both qubits, then
:math:`R_{ZZ}(\pi/2)`, first on the even bonds and then on the odd ones. The
perturbation :math:`R_Z(\theta)` on ``target_qubit`` is the signal the echo
detects. ``b`` sets the magic of the kicks (Clifford when it is a multiple of
:math:`\pi/2`) and also how much the dynamics spreads correlations. ``theta``
adds one magic gate. ``nlayers`` sets how far the forward evolution spreads
the perturbation before the echo refocuses it. The :math:`R_Z(\pi/4)` gates are
fixed :math:`T` gates, so the circuit holds some magic at every ``b``.

.. themed-figure:: ../_static/ansatze/floquet_ansatz.svg
   :align: center

   ``FloquetAnsatz(nqubits=4, nlayers=1, target_qubit=1)``: the whole echo.
   The fixed :math:`R_Z(\pi/4)` gates are magic at every ``b``.

KickedIsing
~~~~~~~~~~~

The kicked Ising chain, a periodically driven Ising model. It is a standard
model of quantum chaos, integrable at some angles and dual-unitary at the
self-dual point. Each step is one period of the drive,

.. math::
    U_\text{step} = \prod_{\text{bonds}} R_{ZZ}(\theta_J)
    \prod_q R_Z(\theta_z)\,R_X(\theta_h),

with the couplings applied on the even bonds, then on the odd bonds of an open
chain. Every coupling is written as ``CNOT · RZ · CNOT``, the form HSMPO splits
into Clifford gates and one rotation, so no transpilation is needed. Each angle
is a single gate. It is a magic gate unless it is a multiple of :math:`\pi/2`,
so ``theta_h``, ``theta_z`` and ``theta_j`` set both the magic and the
entangling power. At the default ``theta_j = -pi/2`` the couplings are Clifford
and only the kicks are magic.

.. themed-figure:: ../_static/ansatze/kicked_ising.svg
   :align: center

   ``KickedIsing(nqubits=4, n_steps=1)`` at generic angles: each angle is one gate.

ParametricKickedIsing
~~~~~~~~~~~~~~~~~~~~~

The same chain, written as a Clifford circuit plus small deviations:

.. math::
    \theta_h = \theta_h^C + \delta_h, \qquad
    \theta_J = \theta_J^C + \delta_J, \qquad
    \theta_z = \delta_z,

with :math:`\theta^C` multiples of :math:`\pi/2`. Each rotation is added as two
gates, :math:`R(\theta^C)` and then :math:`R(\delta)`. HSMPO puts the first in
the stabilizer part and keeps only the second as a magic gate. This is the
ansatz in which the two resources are tuned separately:

- **magic**: ``delta_h``, ``delta_j``, ``delta_z``. A zero deviation adds no
  gate. Small deviations give small magic angles.
- **entanglement**: the Clifford skeleton ``clifford_h``, ``clifford_j`` and
  ``n_steps``. The default :math:`\theta_h^C = \pi/2`,
  :math:`\theta_J^C = -\pi/2` is the self-dual point, where the entanglement
  grows at the maximal speed. At ``clifford_j = 0`` the qubits never interact.

The deviations change the entanglement only perturbatively.

.. themed-figure:: ../_static/ansatze/parametric_kicked_ising.svg
   :align: center

   ``ParametricKickedIsing(nqubits=4, n_steps=1)`` with all three deviations
   non-zero: each rotation is a yellow Clifford gate followed by a blue deviation.

RampedKickedIsing
~~~~~~~~~~~~~~~~~

A kicked Ising chain whose drive changes over time. The transverse deviation
goes from ``delta_h`` at the first step to ``delta_h_last`` at the last one,
geometrically when both have the same sign and linearly otherwise. The
longitudinal one goes from ``delta_z`` to ``delta_z_last``. The ramp decides
where in the circuit the magic sits. When it is concentrated in the first
steps, a head/tail cut leaves the large rotations to the head and only small
angles to the tail. The opposite ramp is a control.

Two optional sources of randomness, both reproducible through ``seed``:

- ``disorder`` (magic): every kick gets its own deviation
  :math:`\delta_h (1 + \text{disorder} \cdot u)`, :math:`u \in [-1, 1]`. The
  seeds are disorder realisations of the same magic profile.
- ``bond_density`` (entanglement): each coupling is kept with this
  probability. At :math:`\delta_J = 0` the couplings are Clifford, so dilution
  lowers the entanglement without changing the magic.

Without a ramp, at ``bond_density = 1`` and ``disorder = 0``, the circuit is
exactly ``ParametricKickedIsing``.

.. themed-figure:: ../_static/ansatze/ramped_kicked_ising.svg
   :align: center

   ``RampedKickedIsing(nqubits=4, n_steps=2, delta_h=0.4, delta_h_last=0.1,
   bond_density=0.8)``: the deviation shrinks from step 1 to step 2, and the
   coupling on bond (2, 3) of step 2 has been dropped.

StarKickedIsing
~~~~~~~~~~~~~~~

A kicked central-spin model: the kicks of ``RampedKickedIsing``, but every
step couples the hub (qubit 0) to every leaf instead of coupling neighbours in
a chain. It is the native circuit of devices with a central element, such as
the IQM Star, where the hub's state can stay in the resonator for a whole step.
Its knobs are those of ``RampedKickedIsing``. ``bond_density`` thins out the
hub-leaf bonds.

.. themed-figure:: ../_static/ansatze/star_kicked_ising.svg
   :align: center

   ``StarKickedIsing(nqubits=5, n_steps=1)``: every leaf is coupled to the hub.

TrotterIsing
~~~~~~~~~~~~

First-order Trotter evolution under the transverse-field Ising Hamiltonian

.. math::
    H = -J \sum_j Z_j Z_{j+1} - h \sum_j X_j

on an open chain. Each step applies :math:`R_{ZZ}(-2J\,dt)` on every bond, then
:math:`R_X(-2h\,dt)` on every qubit. It is a kicked Ising chain with angles set
by a Hamiltonian, transpiled to native gates. The magic comes from the angles,
so from ``dt``, ``J`` and ``h``. The entangling power per step is set by
``J * dt``, and the total by ``n_steps``.

.. themed-figure:: ../_static/ansatze/trotter_ising.svg
   :align: center

   ``TrotterIsing(nqubits=4, n_steps=1, transpile=False)``: one Trotter step,
   before the transpilation to native gates.

Circuit library
---------------

Textbook algorithms used as HSMPO/HQSMPO benchmarks. They are written with
high-level gates and transpiled to native gates on construction (pass
``transpile=False`` to keep the raw circuit).

- ``QFT``: the quantum Fourier transform. The controlled phases
  :math:`\pi/2^k` are fixed magic gates. The circuit has no knob besides
  ``nqubits``.
- ``QFTPhaseKernel``: a QFT followed by one ``RZ`` per qubit with angles
  ``coeffs``. HSMPO has to propagate these magic rotations back through the
  QFT. ``coeffs`` sets the trailing magic.
- ``Grover``: Grover search for ``marked_state``. The Toffoli gates of the
  oracle and the diffuser hold fixed magic. ``n_iterations`` sets the depth.
  The oracle is exact only up to three qubits.
- ``QPE``: phase estimation of :math:`U = \mathrm{diag}(1, e^{2\pi i\,\phi})`
  with ``n_counting`` counting qubits. ``phase`` sets the controlled-phase
  angles, hence the magic. ``n_counting`` sets the size of the inverse QFT.
- ``QAE``: amplitude estimation of :math:`a = \sin^2\theta_A`. ``theta_A``
  sets the magic, ``n_counting`` the counting register.

The figures show the raw circuits (``transpile=False``) on three qubits or
three counting qubits; HSMPO runs on their transpiled form.

.. themed-figure:: ../_static/ansatze/qft.svg
   :align: center

   ``QFT(nqubits=3)``.

.. themed-figure:: ../_static/ansatze/qft_phase_kernel.svg
   :align: center

   ``QFTPhaseKernel(nqubits=3)``, with the kernel angles :math:`c_j`.

.. themed-figure:: ../_static/ansatze/grover.svg
   :align: center

   ``Grover(nqubits=3, n_iterations=1)``: the oracle marks :math:`|111\rangle`.

.. themed-figure:: ../_static/ansatze/qpe.svg
   :align: center

   ``QPE(n_counting=3)``: three counting qubits and the eigenstate qubit.

.. themed-figure:: ../_static/ansatze/qae.svg
   :align: center

   ``QAE(n_counting=3)``: three counting qubits and the work qubit.

Wrappers
--------

- ``CircuitAnsatz`` wraps any Qibo circuit unchanged.
- ``TranspiledAnsatz`` wraps a Qibo circuit and transpiles it to
  ``native_gates``, optionally routed on a ``connectivity`` graph.

For both, the magic and the entanglement are those of the circuit given.

See also
--------

- :doc:`using_ansatze`: how to use an ansatz with HSMPO and how to write your own
- :doc:`../api/models`: the API reference of :mod:`mpstab.models.ansatze`
