"""Circuit templates to simulate: variational patterns and textbook algorithms."""

import random
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import combinations
from typing import ClassVar, List, Optional, Tuple

import numpy as np
from qibo import Circuit, gates
from qibo.models import QFT as qibo_qft

from mpstab.models.topology import Edge, GraphLike, Topology
from mpstab.models.utils import (
    DEFAULT_NATIVE_GATES,
    hardware_compatible_circuit,
    replace_non_clifford_gate,
)


@dataclass
class Ansatz(ABC):
    """
    Abstract ansatz to generate quantum states.

    Every ansatz names the fields that set its two resources, so a sweep knows
    which knob to turn:

    - ``magic_parameters``: fields that set how far the circuit is from a
      Clifford circuit, i.e. how many magic gates HSMPO keeps and how large
      their angles are;
    - ``entanglement_parameters``: fields that set which qubits interact and
      for how long, i.e. how fast the bond dimension of the state can grow.

    A field may appear in both: a generic rotation angle sets the magic and,
    through the couplings, also the entangling power.
    """

    nqubits: int

    magic_parameters: ClassVar[Tuple[str, ...]] = ()
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ()

    def __post_init__(self):
        self._circuit = Circuit(self.nqubits)

    @property
    def circuit(self) -> Circuit:
        return self._circuit

    @circuit.setter
    def circuit(self, value: Circuit):
        if not isinstance(value, Circuit):
            raise TypeError("Expected a Circuit instance")
        self._circuit = value
        self.nqubits = value.nqubits

    @property
    def nparams(self):
        return len(self.circuit.get_parameters())

    @property
    def topology(self) -> Topology:
        """
        The pairs of qubits the circuit couples, read off its gates.

        Every gate acting on two or more qubits couples each pair of them
        (measurements excepted); each pair is listed once, in the orientation
        and order of its first gate. This is what the circuit needs from a
        device, whatever pattern or ``graph`` built it: the circuit runs on a
        device without routing when
        ``ansatz.topology.is_subgraph_of(device)``. It can differ from the
        ``graph`` field, for example when ``bond_density`` drops some bonds.

        The qubits are named by the wire names of the circuit when these are
        integers: after routing on a device, the gates still act on wire
        indices and ``circuit.wire_names`` holds the device qubit of each wire,
        so the topology is the one on the device.
        """
        circuit = self.circuit
        names = list(circuit.wire_names)
        if not all(isinstance(name, int) for name in names):
            names = list(range(circuit.nqubits))
        pairs = [
            (names[a], names[b])
            for gate in circuit.queue
            if not isinstance(gate, gates.M)
            for a, b in combinations(gate.qubits, 2)
        ]
        return Topology.from_graph(pairs, nqubits=max(names, default=-1) + 1)

    def partitionate_circuit(
        self,
        replacement_probability: float = 0.0,
        replacement_method: str = "closest",
        rng: Optional[random.Random] = None,
    ):
        """
        Split the circuit into a Clifford part and the magic gates left over.

        Each non-Clifford gate is replaced by the nearest Clifford one with
        probability ``replacement_probability``, and kept as a magic gate
        otherwise. Every magic gate is recorded with the number of Clifford gates
        preceding it, which is where it has to be reinserted later.

        Args:
            replacement_probability: chance of replacing each non-Clifford gate.
                The default 0 keeps every non-Clifford gate as magic.
            replacement_method: ``"closest"`` or ``"random"`` Clifford angle.
            rng: source of the replacement draws, such as a seeded
                ``random.Random``. Defaults to the global ``random`` module.

        Returns:
            ``((magic_gates, clifford_circuit), full_circuit)``, where
            ``magic_gates`` is a list of ``(breakpoint_index, gate)`` and
            ``full_circuit`` is the sampled circuit including its magic gates.
        """
        return self._partition(
            self.circuit.queue, replacement_probability, replacement_method, rng
        )

    def _partition(self, queue, replacement_probability, replacement_method, rng=None):
        """:meth:`partitionate_circuit` over any gate sequence."""
        rng = random if rng is None else rng
        magic_gates = []
        clifford_circuit = Circuit(self.nqubits)
        full_circuit = Circuit(self.nqubits)

        for gate in queue:
            if not gate.clifford and not isinstance(gate, gates.M):
                if rng.random() > replacement_probability:
                    magic_gates.append((len(clifford_circuit.queue), gate))
                    full_circuit.add(gate)
                    continue
                gate = replace_non_clifford_gate(gate, replacement_method, rng=rng)
            clifford_circuit.add(gate)
            full_circuit.add(gate)

        return (magic_gates, clifford_circuit), full_circuit


def _add_rzz(circuit: Circuit, a: int, b: int, theta: float) -> None:
    """Add ``R_ZZ(theta)`` on ``a``, ``b`` as ``CNOT · RZ(theta) on b · CNOT``.

    This is the form HSMPO splits into Clifford gates and one rotation, so the
    coupling needs no transpilation.
    """
    circuit.add(gates.CNOT(a, b))
    circuit.add(gates.RZ(q=b, theta=theta))
    circuit.add(gates.CNOT(a, b))


def _topology(graph: Optional[GraphLike], nqubits: int) -> Optional[Topology]:
    """The :class:`.Topology` of a ``graph`` field, or ``None``."""
    return None if graph is None else Topology.coerce(graph, nqubits)


@dataclass
class HardwareEfficient(Ansatz):
    """
    Layers of random RY rotations, each followed by CZ gates on the couplings.

    Args:
        nqubits: number of qubits.
        nlayers: number of rotation-and-entangle layers.
        entangling: whether to add the CZ gates.
        graph: the couplings, any input of :meth:`.Topology.coerce` (a name
            such as ``"grid"``, a list of edges, a graph); a CZ on every edge,
            layer by layer. ``None`` (default) keeps the ring
            ``CZ(q, q+1 mod n)`` in qubit order.
    """

    nlayers: int = 1
    entangling: bool = True
    graph: Optional[GraphLike] = field(default=None, kw_only=True)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("nlayers",)
    entanglement_parameters: ClassVar[Tuple[str, ...]] = (
        "nlayers",
        "entangling",
        "graph",
    )

    def __post_init__(self):
        super().__post_init__()
        topology = _topology(self.graph, self.nqubits)
        if topology is None:
            # the ring; on 2 qubits it is the single bond (0, 1), since
            # CZ(0, 1) CZ(1, 0) would cancel out
            n = self.nqubits
            bonds = [(q, (q + 1) % n) for q in range(n if n > 2 else n - 1)]
        else:
            bonds = [bond for layer in topology.layers for bond in layer]
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            if self.entangling:
                for a, b in bonds:
                    self.circuit.add(gates.CZ(a, b))


@dataclass
class HardwareEfficientBrickwork(Ansatz):
    """
    Hardware Efficient ansatz with a full brickwork entanglement pattern.

    Each layer applies two sublayers of single-qubit RY+RZ rotations interleaved
    with CZ entanglement:

        sublayer 1: RY+RZ on all qubits → CZ on even bonds (0-1, 2-3, …)
        sublayer 2: RY+RZ on all qubits → CZ on odd bonds (1-2, 3-4, …) + CZ(0, n-1)

    A final layer of RY rotations closes the circuit.  The ring-closing CZ(0, n-1)
    in the odd sublayer creates long-range entanglement.

    With a ``graph``, each brickwork layer has one sublayer per layer of
    disjoint bonds of the topology (RY+RZ on all qubits, then CZ on those
    bonds), so a chain gives the two sublayers above without the ring closure.

    Args:
        nqubits: Number of qubits.
        nlayers: Number of full brickwork layers.
        graph: the couplings, any input of :meth:`.Topology.coerce`;
            ``None`` (default) keeps the pattern above.
    """

    nlayers: int = 1
    graph: Optional[GraphLike] = field(default=None, kw_only=True)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("nlayers",)
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("nlayers", "graph")

    def __post_init__(self):
        super().__post_init__()
        topology = _topology(self.graph, self.nqubits)
        if topology is not None:
            for _ in range(self.nlayers):
                for layer in topology.layers:
                    self._rotations(gates.RY, gates.RZ)
                    for a, b in layer:
                        self.circuit.add(gates.CZ(q0=a, q1=b))
            self._rotations(gates.RY)
            return
        for _ in range(self.nlayers):
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(self.nqubits):
                self.circuit.add(gates.RZ(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(0, self.nqubits - 1, 2):
                self.circuit.add(gates.CZ(q0=q, q1=q + 1))
            for q in range(self.nqubits):
                self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(self.nqubits):
                self.circuit.add(gates.RZ(q=q, theta=np.random.uniform(-np.pi, np.pi)))
            for q in range(1, self.nqubits - 2, 2):
                self.circuit.add(gates.CZ(q0=q, q1=q + 1))
            self.circuit.add(gates.CZ(q0=0, q1=self.nqubits - 1))
        for q in range(self.nqubits):
            self.circuit.add(gates.RY(q=q, theta=np.random.uniform(-np.pi, np.pi)))

    def _rotations(self, *kinds) -> None:
        """Random rotations of each kind on every qubit, one kind after the other."""
        for kind in kinds:
            for q in range(self.nqubits):
                self.circuit.add(kind(q=q, theta=np.random.uniform(-np.pi, np.pi)))


@dataclass
class _KickedIsingBase(Ansatz):
    r"""
    Step structure shared by the kicked Ising ansätze.

    Each of the ``n_steps`` steps kicks every qubit (:meth:`_kick`), then applies
    the :math:`R_{ZZ}` couplings (:meth:`_coupling`) on the edges of ``graph``,
    layer by layer (:attr:`.Topology.layers`). A subclass sets the kick and
    coupling angles; the bond dilution is set by :meth:`_keep_bond`.

    ``graph`` is any input of :meth:`.Topology.coerce`: a name (``"chain"``,
    ``"ring"``, ``"star"``, ``"grid"``, ``"grid:RxC"``, ``"complete"``), a list
    of edges, a ``networkx`` graph or a :class:`.Topology`, such as one built
    with :meth:`.Topology.from_device`. It is the interaction graph of the
    model; the circuit runs on a device without routing when every pair of
    ``graph`` is coupled on the chip (:meth:`.Topology.is_subgraph_of`).
    At the Clifford point, with the default
    Clifford angles, one step prepares a graph state of ``graph`` up to local
    Cliffords, so the topology sets how fast the stabilizer part entangles at
    fixed magic.
    """

    n_steps: int = 1
    graph: GraphLike = field(default="chain", kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        self._topology = Topology.coerce(self.graph, self.nqubits)
        self._validate()
        for step in range(self.n_steps):
            for q in range(self.nqubits):
                self._kick(step, q)
            for layer in self._topology.layers:
                for a, b in layer:
                    if self._keep_bond():
                        self._coupling(a, b)

    def _validate(self) -> None:
        """Check the fields before any gate is added."""

    @property
    def edges(self) -> List[Edge]:
        """The edges of ``graph``, in order; ``bond_density`` may skip some."""
        return list(self._topology.edges)

    @property
    def coupling_depth(self) -> int:
        """Number of layers of disjoint bonds of ``graph`` in one step."""
        return self._topology.depth

    def _keep_bond(self) -> bool:
        """Whether the next coupling is applied."""
        return True

    def _kick(self, step: int, qubit: int) -> None:
        raise NotImplementedError

    def _coupling(self, a: int, b: int) -> None:
        raise NotImplementedError


@dataclass
class KickedIsing(_KickedIsingBase):
    r"""
    Build a kicked Ising circuit, by default on an open chain.

    Each of the ``n_steps`` steps applies the kicks :math:`R_X(\theta_h)` and
    then :math:`R_Z(\theta_z)` on every qubit, followed by the couplings
    :math:`R_{ZZ}(\theta_J)` on the even bonds and then on the odd bonds. Every
    coupling is written as ``CNOT · RZ(theta_j) · CNOT``, the form HSMPO splits
    into Clifford and magic gates, so the circuit needs no transpilation. At the
    default ``theta_j = -pi/2`` the couplings are Clifford and only the kicks are
    magic, so the MPS bond dimension grows with the number of steps.

    Every angle that is not a multiple of :math:`\pi/2` is a magic gate as a
    whole; :class:`ParametricKickedIsing` separates the Clifford part of each
    angle from the magic one.

    Args:
        nqubits: number of qubits in the chain.
        n_steps: number of kick-and-couple steps.
        theta_h: angle of the transverse kicks :math:`R_X`.
        theta_z: angle of the longitudinal kicks :math:`R_Z`.
        theta_j: angle of the :math:`R_{ZZ}` couplings.
        graph: the couplings, any input of :meth:`.Topology.coerce`
            (default: the open chain).
    """

    theta_h: float = 0.8
    theta_z: float = 0.0
    theta_j: float = -0.5 * np.pi

    magic_parameters: ClassVar[Tuple[str, ...]] = ("theta_h", "theta_z", "theta_j")
    entanglement_parameters: ClassVar[Tuple[str, ...]] = (
        "n_steps",
        "theta_h",
        "theta_j",
        "graph",
    )

    def _kick(self, step: int, qubit: int) -> None:
        self.circuit.add(gates.RX(q=qubit, theta=self.theta_h))
        self.circuit.add(gates.RZ(q=qubit, theta=self.theta_z))

    def _coupling(self, a: int, b: int) -> None:
        _add_rzz(self.circuit, a, b, self.theta_j)


@dataclass
class ParametricKickedIsing(_KickedIsingBase):
    r"""
    Build a kicked Ising circuit parametrised by its distance from a Clifford point.

    It is the circuit of :class:`KickedIsing` with the angles

    .. math::
        \theta_h = \theta_h^C + \delta_h, \qquad
        \theta_J = \theta_J^C + \delta_J, \qquad
        \theta_z = \delta_z,

    where :math:`\theta_h^C` and :math:`\theta_J^C` are multiples of
    :math:`\pi/2`, so that at :math:`\delta = 0` the whole circuit is Clifford.
    The defaults :math:`\theta_h^C = \pi/2`, :math:`\theta_J^C = -\pi/2` are
    the self-dual point, where the entanglement grows at the maximal speed.

    Every rotation is written as its Clifford part followed by the deviation,
    :math:`R(\theta^C + \delta) = R(\delta)\,R(\theta^C)`, as two separate
    gates. HSMPO then puts :math:`R(\theta^C)` in the stabilizer part and keeps
    only the small rotation :math:`R(\delta)` as a magic gate, instead of
    treating the whole :math:`R(\theta^C + \delta)` as magic. The two groups of
    fields therefore tune the two resources separately:

    - the deviations ``delta_h``, ``delta_j``, ``delta_z`` set the magic: the
      number of magic gates (a deviation equal to zero adds no gate) and the
      size of their angles;
    - the Clifford skeleton ``clifford_h``, ``clifford_j`` and ``n_steps`` set
      the entanglement: which Clifford circuit the stabilizer part evolves,
      hence how fast the entanglement grows. The deviations change it only
      perturbatively.

    Each of the ``n_steps`` steps applies the kicks :math:`R_X` and then
    :math:`R_Z` on every qubit, followed by the couplings :math:`R_{ZZ}` on the
    even bonds and then on the odd bonds, each coupling written as
    ``CNOT · RZ · CNOT``.

    Args:
        nqubits: number of qubits in the chain.
        n_steps: number of kick-and-couple steps.
        delta_h: deviation of the transverse kicks :math:`R_X` from ``clifford_h``.
        delta_j: deviation of the :math:`R_{ZZ}` couplings from ``clifford_j``.
        delta_z: angle of the longitudinal kicks :math:`R_Z` (Clifford point: 0).
        clifford_h: Clifford angle of the transverse kicks, a multiple of pi/2.
        clifford_j: Clifford angle of the couplings, a multiple of pi/2.
        graph: the couplings, any input of :meth:`.Topology.coerce`
            (default: the open chain).
    """

    delta_h: float = 0.1
    delta_j: float = 0.0
    delta_z: float = 0.0
    clifford_h: float = 0.5 * np.pi
    clifford_j: float = -0.5 * np.pi

    magic_parameters: ClassVar[Tuple[str, ...]] = ("delta_h", "delta_j", "delta_z")
    entanglement_parameters: ClassVar[Tuple[str, ...]] = (
        "n_steps",
        "clifford_h",
        "clifford_j",
        "graph",
    )

    def _validate(self) -> None:
        for name in ("clifford_h", "clifford_j"):
            quarter_turns = getattr(self, name) / (0.5 * np.pi)
            if not np.isclose(quarter_turns, round(quarter_turns)):
                raise ValueError(
                    f"{name} must be a multiple of pi/2, got {getattr(self, name)}."
                )

    def _kick(self, step: int, qubit: int) -> None:
        self._rotation(gates.RX, qubit, self.clifford_h, self.delta_h)
        self._rotation(gates.RZ, qubit, 0.0, self.delta_z)

    def _coupling(self, a: int, b: int) -> None:
        """Add the ``R_ZZ`` coupling of qubits ``a``, ``b`` as ``CNOT · RZ · CNOT``."""
        self.circuit.add(gates.CNOT(a, b))
        self._rotation(gates.RZ, b, self.clifford_j, self.delta_j)
        self.circuit.add(gates.CNOT(a, b))

    def _rotation(self, gate, qubit: int, clifford_angle: float, delta: float):
        """Add ``gate(clifford_angle)`` then ``gate(delta)``, skipping zero angles."""
        if clifford_angle != 0:
            self.circuit.add(gate(q=qubit, theta=clifford_angle))
        if delta != 0:
            self.circuit.add(gate(q=qubit, theta=delta))

    @property
    def theta_h(self) -> float:
        """Total angle of the transverse kicks."""
        return self.clifford_h + self.delta_h

    @property
    def theta_j(self) -> float:
        """Total angle of the couplings."""
        return self.clifford_j + self.delta_j


def _ramp(first: float, last: float, n_steps: int) -> np.ndarray:
    """``n_steps`` values from ``first`` to ``last``: geometric when both have the
    same sign, linear otherwise."""
    if n_steps == 1 or first == last:
        return np.full(n_steps, float(first))
    if first * last > 0:
        return np.sign(first) * np.geomspace(abs(first), abs(last), n_steps)
    return np.linspace(first, last, n_steps)


@dataclass
class RampedKickedIsing(ParametricKickedIsing):
    r"""
    Build a :class:`ParametricKickedIsing` circuit whose magic changes from step to step.

    The deviation of the transverse kicks goes from ``delta_h`` at the first
    step to ``delta_h_last`` at the last one, geometrically when the two have
    the same sign and linearly otherwise; likewise the longitudinal kicks, from
    ``delta_z`` to ``delta_z_last``. With the magic concentrated in the first
    steps, a head/tail split leaves the large rotations to the head and only
    small angles to the tail; ``delta_h < delta_h_last`` is the opposite
    profile, as a control. Without ``delta_h_last`` and ``delta_z_last``, at
    ``bond_density = 1`` and ``disorder = 0``, the circuit is exactly
    :class:`ParametricKickedIsing`.

    Two optional sources of randomness, drawn from ``numpy.random.RandomState(seed)``
    (from numpy's global generator when ``seed`` is ``None``), in circuit order:

    - ``disorder > 0`` gives every kick its own deviation,
      :math:`\delta_h (1 + \mathrm{disorder} \cdot u)` with :math:`u` uniform
      in :math:`[-1, 1]`, so different seeds are disorder realisations of the
      same magic profile;
    - ``bond_density < 1`` keeps each coupling of each step with that
      probability and drops the others. At :math:`\delta_J = 0` the couplings
      are Clifford, so this lowers the entanglement without changing the magic.

    No number is drawn when ``disorder = 0`` and ``bond_density = 1``.

    The ramp and the disorder add magic knobs to those of
    :class:`ParametricKickedIsing`; ``bond_density`` adds an entanglement knob.

    Args:
        nqubits: number of qubits.
        n_steps: number of kick-and-couple steps.
        delta_h: deviation of the transverse kicks at the first step.
        delta_h_last: deviation of the transverse kicks at the last step
            (``None``: ``delta_h`` at every step).
        delta_j: deviation of the :math:`R_{ZZ}` couplings from ``clifford_j``.
        delta_z: angle of the longitudinal kicks at the first step.
        delta_z_last: angle of the longitudinal kicks at the last step
            (``None``: ``delta_z`` at every step).
        clifford_h: Clifford angle of the transverse kicks, a multiple of pi/2.
        clifford_j: Clifford angle of the couplings, a multiple of pi/2.
        bond_density: probability of keeping each coupling, in ``(0, 1]``.
        disorder: relative spread of the transverse deviations across the kicks.
        seed: seed of the random draws.
        graph: the couplings, any input of :meth:`.Topology.coerce`
            (default: the open chain); ``bond_density`` applies to its edges.
    """

    delta_h_last: Optional[float] = None
    delta_z_last: Optional[float] = None
    bond_density: float = 1.0
    disorder: float = 0.0
    seed: Optional[int] = None

    magic_parameters: ClassVar[Tuple[str, ...]] = (
        ParametricKickedIsing.magic_parameters
        + ("delta_h_last", "delta_z_last", "disorder")
    )
    entanglement_parameters: ClassVar[Tuple[str, ...]] = (
        ParametricKickedIsing.entanglement_parameters + ("bond_density",)
    )

    def __post_init__(self):
        self._rng = np.random if self.seed is None else np.random.RandomState(self.seed)
        last_h = self.delta_h if self.delta_h_last is None else self.delta_h_last
        last_z = self.delta_z if self.delta_z_last is None else self.delta_z_last
        self._deltas_h = _ramp(self.delta_h, last_h, self.n_steps)
        self._deltas_z = _ramp(self.delta_z, last_z, self.n_steps)
        super().__post_init__()

    def _validate(self) -> None:
        super()._validate()
        if not 0.0 < self.bond_density <= 1.0:
            raise ValueError(
                f"bond_density must lie in (0, 1], got {self.bond_density}."
            )

    def _kick(self, step: int, qubit: int) -> None:
        factor = (
            1.0 + self.disorder * self._rng.uniform(-1, 1) if self.disorder > 0 else 1.0
        )
        delta_h = float(self._deltas_h[step] * factor)
        self._rotation(gates.RX, qubit, self.clifford_h, delta_h)
        self._rotation(gates.RZ, qubit, 0.0, float(self._deltas_z[step]))

    def _keep_bond(self) -> bool:
        return self.bond_density >= 1.0 or self._rng.random() < self.bond_density


@dataclass
class StarKickedIsing(RampedKickedIsing):
    r"""
    Build a :class:`RampedKickedIsing` circuit on a star: a central-spin model.

    The kicks are those of :class:`RampedKickedIsing`; the couplings of each
    step are :math:`R_{ZZ}` between the hub (qubit 0) and every leaf, leaf 1
    first, each written as ``CNOT · RZ · CNOT``. The hub is an ordinary logical
    qubit; the label 0 is a convention. The circuit suits devices with a central
    element, such as an IQM Star: there the compiler can move the hub's state
    into the computational resonator once per step and apply all its CZ gates
    from it, instead of moving a qubit in and out for nearly every CZ as a
    chain needs (on a 10-qubit IQM Star, 6 steps: about 20 MOVE gates against
    about 100 for the chain).

    It is :class:`RampedKickedIsing` with ``graph="star"``.

    Args: as :class:`RampedKickedIsing`; ``bond_density`` applies to the
    hub-leaf couplings.
    """

    graph: GraphLike = field(default="star", kw_only=True)


@dataclass
class HammingWeightPreserving(Ansatz):
    r"""
    Hamming-weight-preserving ansatz built from decomposed RBS gates.

    Each RBS gate rotates within the :math:`\{|01\rangle, |10\rangle\}`
    subspace of a qubit pair, leaving :math:`|00\rangle` and :math:`|11\rangle`
    unchanged::

        RBS(θ)|01⟩ =  cos(θ)|01⟩ + sin(θ)|10⟩
        RBS(θ)|10⟩ = -sin(θ)|01⟩ + cos(θ)|10⟩

    RBS is decomposed into gates mpstab supports natively::

        CNOT(ctrl=q1, tgt=q0)
        RY(q1,  θ)              ← magic gate
        CNOT(ctrl=q0, tgt=q1)
        RY(q1, -θ)              ← magic gate
        CNOT(ctrl=q0, tgt=q1)
        CNOT(ctrl=q1, tgt=q0)

    The circuit always starts with n//2 X gates to prepare the half-filling
    state :math:`|1\dots10\dots0\rangle`, giving a fixed Hamming weight of n//2.

    Within each layer the connectivity pattern mirrors `hw_preserving` from
    Qibo examples: four `connect_qubits` passes with jump sizes 1 and 2 and
    different starting offsets to explore both nearest- and next-nearest-
    neighbour correlations with circular boundary conditions.

    With a ``graph``, each layer puts one RBS gate on every edge of the
    topology, layer of disjoint bonds after layer; RBS preserves the Hamming
    weight on any pair, so any topology keeps the half-filling sector.

    Args:
        nqubits: Number of qubits (must be even).
        nlayers: Number of RBS layers.
        graph: the couplings, any input of :meth:`.Topology.coerce`;
            ``None`` (default) keeps the pattern above.
    """

    nlayers: int = 1
    graph: Optional[GraphLike] = field(default=None, kw_only=True)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("nlayers",)
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("nlayers", "graph")

    def __post_init__(self):
        if self.nqubits % 2 != 0:
            raise ValueError(
                "HammingWeightPreserving requires an even number of qubits."
            )
        super().__post_init__()
        for q in range(self.nqubits // 2):
            self.circuit.add(gates.X(q=q))
        topology = _topology(self.graph, self.nqubits)
        if topology is not None:
            for _ in range(self.nlayers):
                for layer in topology.layers:
                    for a, b in layer:
                        theta = np.random.uniform(-np.pi, np.pi)
                        self.circuit += self._rbs_decomposition(a, b, theta)
            return
        for _ in range(self.nlayers):
            self._connect_qubits(jumpsize=1, start_from=0)
            self._connect_qubits(jumpsize=1, start_from=1)
            self._connect_qubits(jumpsize=2, start_from=0)
            self._connect_qubits(jumpsize=2, start_from=1)
            self._connect_qubits(jumpsize=2, start_from=3)

    def _connect_qubits(self, jumpsize: int, start_from: int) -> None:
        """
        Add RBS gates between qubit pairs (q, (q+jumpsize) % nqubits).

        A pair that wraps onto itself (``jumpsize`` a multiple of ``nqubits``,
        i.e. a jump of 2 on 2 qubits) is skipped, with no angle drawn.
        """
        for q in range(start_from, self.nqubits, jumpsize + 1):
            q0 = q
            q1 = (q + jumpsize) % self.nqubits
            if q1 == q0:
                continue
            theta = np.random.uniform(-np.pi, np.pi)
            self.circuit += self._rbs_decomposition(q0, q1, theta)

    def _rbs_decomposition(self, q0: int, q1: int, theta: float) -> Circuit:
        """
        Decompose RBS(theta) on qubits (q0, q1) into CNOT and RY gates.

        Equal to CNOT(q1→q0) · CRY(ctrl=q0, tgt=q1, 2θ) · CNOT(q1→q0),
        with CRY expanded as RY(θ)·CNOT·RY(-θ)·CNOT.  Verified analytically
        on all four computational basis states.
        """
        circ = Circuit(self.nqubits)
        circ.add(gates.CNOT(q1, q0))
        circ.add(gates.RY(q=q1, theta=theta))
        circ.add(gates.CNOT(q0, q1))
        circ.add(gates.RY(q=q1, theta=-theta))
        circ.add(gates.CNOT(q0, q1))
        circ.add(gates.CNOT(q1, q0))
        return circ


@dataclass
class FloquetAnsatz(Ansatz):
    """
    Floquet echo: U = (FL)^t · Rz(theta) on qubit q · (FL^t)†.

    Args:
        nlayers (int): number of Floquet layers.
        b (float): parameter controlling the magic in the circuit;
        theta (float): controls the signal;
        target_qubit (int): this is the qubit, in the circuit, on which we perform
            the final local measurement (and the one on which we apply H and RZ).
        decompose_rzz (bool): if ``True``, RZZ is decomposed into CNOTs and RZ.
        graph: the couplings of a Floquet layer, any input of
            :meth:`.Topology.coerce`; one sublayer per layer of disjoint bonds.
            The default, the open chain, gives the even and the odd sublayer.
    """

    nlayers: int = 2
    b: float = 0.4 * np.pi
    theta: float = 0.5 * np.pi
    target_qubit: int = 1
    decompose_rzz: bool = True
    graph: GraphLike = field(default="chain", kw_only=True)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("b", "theta")
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("nlayers", "b", "graph")

    def __post_init__(self):
        super().__post_init__()
        self._topology = Topology.coerce(self.graph, self.nqubits)

        self.half_sandwich = Circuit(self.nqubits)
        for _ in range(self.nlayers):
            self.half_sandwich += self._build_floquet_layer()

        self.circuit.add(gates.H(self.target_qubit))
        self.circuit += self.half_sandwich
        self.circuit.add(gates.RZ(q=self.target_qubit, theta=self.theta))
        self.circuit += self.half_sandwich.invert()

    def _build_floquet_layer(self) -> Circuit:
        """Construct one Floquet layer over all links, one sublayer per layer of bonds."""
        layer = Circuit(self.nqubits)
        for bonds in self._topology.layers:
            layer += self._build_sublayer(bonds)
        return layer

    def _build_sublayer(self, bonds):
        """One sublayer of a Floquet layer: kicks and RZZ on each of ``bonds``."""
        layer = Circuit(self.nqubits)

        for q1, q2 in bonds:
            layer.add(gates.RZ(q=q1, theta=0.25 * np.pi))
            layer.add(gates.RX(q=q1, theta=self.b))
            layer.add(gates.RZ(q=q2, theta=0.25 * np.pi))
            layer.add(gates.RX(q=q2, theta=self.b))
            if not self.decompose_rzz:
                layer.add(gates.RZZ(q0=q1, q1=q2, theta=0.5 * np.pi))
            else:
                _add_rzz(layer, q1, q2, 0.5 * np.pi)

        return layer

    def partitionate_circuit(
        self,
        replacement_probability: float = 0.0,
        replacement_method: str = "closest",
        rng: Optional[random.Random] = None,
    ):
        """
        Partitionate the full Floquet circuit U = H · half_sandwich · RZ · (half_sandwich)†,
        but only *sweep* the half_sandwich for magic gates.
        Returns ((magic_gates, clifford_only_circuit), full_circuit).
        """
        magic_gates = []
        clifford_only_circuit = Circuit(self.nqubits)
        full_circuit = Circuit(self.nqubits)

        # 1) initial H
        H = gates.H(self.target_qubit)
        full_circuit.add(deepcopy(H))
        clifford_only_circuit.add(deepcopy(H))

        # 2) first half: collect both full_circuit_1 and its magic break-points
        (magic_gates_1, clifford_block_1), full_circuit_1 = self._partition(
            deepcopy(self.half_sandwich).queue,
            replacement_probability,
            replacement_method,
            rng,
        )

        # shift each local_bp by +1 to account for the leading H at index 0
        magic_gates.extend([(bp + 1, gate) for bp, gate in magic_gates_1])
        clifford_only_circuit += clifford_block_1
        full_circuit += full_circuit_1

        # 3) the central RZ
        rz = gates.RZ(q=self.target_qubit, theta=self.theta)
        full_circuit.add(deepcopy(rz))
        if rz.clifford:
            clifford_only_circuit.add(deepcopy(rz))
        else:
            # its position is len(half_sandwich) + 1
            magic_gates.append((len(clifford_block_1.queue), deepcopy(rz)))

        # 4) inverted half + mirrored magic gates
        for bp, gate in magic_gates_1[::-1]:
            # mirror the *shifted* index
            magic_gates.append(
                (
                    len(clifford_block_1.queue) - bp + len(clifford_only_circuit.queue),
                    deepcopy(gate).dagger(),
                )
            )

        clifford_only_circuit += clifford_block_1.invert()
        full_circuit += full_circuit_1.invert()

        return (magic_gates, clifford_only_circuit), full_circuit


@dataclass(kw_only=True)
class CircuitAnsatz(Ansatz):
    """
    A simple wrapper that converts a Qibo Circuit into an Ansatz object.

    Args:
        input_circuit (Circuit): The Qibo circuit to wrap.
    """

    qibo_circuit: Circuit
    # We set nqubits as init=False because we derive it from input_circuit
    nqubits: int = field(init=False)

    def __post_init__(self):
        self.nqubits = self.qibo_circuit.nqubits
        super().__post_init__()
        self._circuit = deepcopy(self.qibo_circuit)


# ---------------------------------------------------------------------------
# Circuit-library ansatze: textbook algorithms (QFT, Grover, QPE, QAE, Trotter)
# to benchmark HSMPO and HQSMPO against.
#
# They are written with high-level gates (CU1, SWAP, TOFFOLI, RZZ, ...) and
# transpiled to the native gate set on construction, which turns every
# non-Clifford gate into an `rz` rotation -- the form HSMPO understands, so
# nothing needs hand-decomposing. Pass ``transpile=False`` to keep the raw
# high-level circuit for inspection; HSMPO itself needs the transpiled form.
# ---------------------------------------------------------------------------
def _qft_circuit(n: int) -> Circuit:
    """QFT sub-circuit using H, controlled-phase (CU1) and SWAP (no transpilation)."""
    return qibo_qft(nqubits=n)


def _multi_controlled_z(circuit: Circuit, controls: List[int], target: int) -> None:
    """
    Append a multi-controlled-Z built from CZ and TOFFOLI, so that it unrolls.

    Exact for up to two controls (CZ, and CCZ = H.CCX.H). Beyond that it chains
    pairwise Toffolis sharing the target, which is a benchmark-oriented
    approximation rather than a faithful multi-controlled-Z: do not rely on
    Grover oracle correctness above 3 qubits.
    """
    n_ctrl = len(controls)
    if n_ctrl == 0:
        circuit.add(gates.Z(target))
    elif n_ctrl == 1:
        circuit.add(gates.CZ(controls[0], target))
    elif n_ctrl == 2:
        circuit.add(gates.H(target))
        circuit.add(gates.TOFFOLI(controls[0], controls[1], target))
        circuit.add(gates.H(target))
    else:
        circuit.add(gates.H(target))
        for c in controls[:-1]:
            circuit.add(gates.TOFFOLI(c, controls[-1], target))
        circuit.add(gates.H(target))


@dataclass(kw_only=True)
class _CompiledAnsatz(Ansatz):
    """
    Base for circuit-library ansatze that build a high-level circuit and (by
    default) transpile it to the native gate set so HSMPO can consume it.

    Subclasses implement :meth:`_build_circuit`.

    Args:
        transpile: If ``True`` (default), unroll the built circuit into
            ``native_gates`` (Clifford GPI2 gates are frozen, as in
            :class:`TranspiledAnsatz`). If ``False``, keep the raw high-level
            circuit.
        native_gates: Target native gate set for transpilation.
        connectivity: Optional coupling map of the device, for
            transpilation; see :func:`.hardware_compatible_circuit`.
    """

    transpile: bool = True
    native_gates: Optional[List] = field(
        default_factory=lambda: list(DEFAULT_NATIVE_GATES)
    )
    connectivity: Optional[GraphLike] = None

    def __post_init__(self):
        super().__post_init__()
        circuit = self._build_circuit()
        if self.transpile:
            circuit = hardware_compatible_circuit(
                circuit, self.native_gates, self.connectivity
            )
            for g in circuit.parametrized_gates:
                if isinstance(g, gates.GPI2) and g.clifford:
                    g.trainable = False
        self._circuit = circuit

    def _build_circuit(self) -> Circuit:
        raise NotImplementedError


@dataclass(kw_only=True)
class TranspiledAnsatz(_CompiledAnsatz):
    """
    Any circuit, transpiled into the native gates of a given quantum device
    presenting a given connectivity.

    Args:
        original_circuit: The circuit to be transpiled.
        native_gates: list of native gates of the device. Default is
            ``[gates.GPI2, gates.RZ, gates.Z, gates.CZ]``.
        connectivity: the coupling map of the device, see
            :func:`.hardware_compatible_circuit`. ``None`` (default) ignores
            connectivity constraints.
    """

    original_circuit: Circuit
    # Always transpiled; nqubits is taken from the circuit.
    transpile: bool = field(default=True, init=False)
    nqubits: int = field(init=False)

    def __post_init__(self):
        self.nqubits = self.original_circuit.nqubits
        super().__post_init__()

    def _build_circuit(self) -> Circuit:
        return self.original_circuit


@dataclass(kw_only=True)
class QFT(_CompiledAnsatz):
    """Quantum Fourier Transform (H, controlled-phase and SWAP)."""

    def _build_circuit(self) -> Circuit:
        return _qft_circuit(self.nqubits)


@dataclass(kw_only=True)
class QFTPhaseKernel(_CompiledAnsatz):
    """QFT followed by a diagonal RZ phase kernel (no closing inverse QFT).

    The trailing RZ layer's magic rotations get Clifford-backpropagated through
    the QFT's H/CU1/SWAP structure -- a stronger HSMPO test than a single
    trailing CZ chain.

    Args:
        coeffs: Per-qubit RZ angles. Default: binary-weighted ``2*pi/2^(j+1)``.
    """

    coeffs: Optional[np.ndarray] = None

    magic_parameters: ClassVar[Tuple[str, ...]] = ("coeffs",)

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        coeffs = self.coeffs
        if coeffs is None:
            coeffs = np.array([2 * np.pi / 2 ** (j + 1) for j in range(n)])

        circuit = _qft_circuit(n)
        for j, theta in enumerate(coeffs):
            circuit.add(gates.RZ(j, theta=float(theta)))
        return circuit


@dataclass(kw_only=True)
class Grover(_CompiledAnsatz):
    r"""Grover search: H^n then n_iterations x (oracle + diffuser).

    The oracle phase-flips ``marked_state`` (default :math:`|1\dots1\rangle`) via a
    multi-controlled-Z; the diffuser is the standard inversion-about-the-mean.
    See :func:`_multi_controlled_z` for the oracle-correctness caveat at
    ``nqubits > 3``.

    Args:
        marked_state: Integer in [0, 2^nqubits). Default 2^nqubits - 1.
        n_iterations: Grover iterations. Default round(pi/4 * sqrt(2^nqubits)).
    """

    marked_state: Optional[int] = None
    n_iterations: Optional[int] = None

    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("n_iterations",)

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        marked_state = self.marked_state
        n_iterations = self.n_iterations
        if marked_state is None:
            marked_state = (1 << n) - 1
        if n_iterations is None:
            n_iterations = max(1, int(round(np.pi / 4 * np.sqrt(2**n))))

        circuit = Circuit(n)
        for q in range(n):
            circuit.add(gates.H(q))

        for _ in range(n_iterations):
            # Oracle: phase-flip on |marked_state>.
            for q in range(n):
                if not (marked_state >> (n - 1 - q)) & 1:
                    circuit.add(gates.X(q))
            _multi_controlled_z(circuit, controls=list(range(n - 1)), target=n - 1)
            for q in range(n):
                if not (marked_state >> (n - 1 - q)) & 1:
                    circuit.add(gates.X(q))

            # Diffuser: H^n (2|0><0| - I) H^n.
            for q in range(n):
                circuit.add(gates.H(q))
                circuit.add(gates.X(q))
            _multi_controlled_z(circuit, controls=list(range(n - 1)), target=n - 1)
            for q in range(n):
                circuit.add(gates.X(q))
                circuit.add(gates.H(q))

        return circuit


@dataclass(kw_only=True)
class QPE(_CompiledAnsatz):
    r"""Quantum Phase Estimation for U = diag(1, e^{2 pi i phase}).

    Layout: ``n_counting`` counting qubits + 1 eigenstate qubit (prepared in
    :math:`|1\rangle`). Controlled-U^{2^k} is a controlled phase (CU1); the counting register
    is closed with an inverse QFT. ``nqubits`` is derived as ``n_counting + 1``.

    Args:
        n_counting: Number of counting qubits.
        phase: True phase in [0, 1); accuracy ~ 1/2^n_counting.
    """

    n_counting: int = 3
    phase: float = 0.375
    nqubits: int = field(init=False)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("phase",)
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("n_counting",)

    def __post_init__(self):
        self.nqubits = self.n_counting + 1
        super().__post_init__()

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        eig = self.n_counting

        circuit = Circuit(n)
        circuit.add(gates.X(eig))
        for q in range(self.n_counting):
            circuit.add(gates.H(q))

        for k in range(self.n_counting):
            circuit.add(gates.CU1(k, eig, theta=2 * np.pi * self.phase * (2**k)))

        # Inverse QFT on the counting register (added gate by gate since it acts
        # on a sub-register of the full circuit).
        for gate in _qft_circuit(self.n_counting).invert().queue:
            circuit.add(gate)
        return circuit


@dataclass(kw_only=True)
class QAE(_CompiledAnsatz):
    """Canonical Quantum Amplitude Estimation.

    ``n_counting`` counting qubits + 1 work qubit. State preparation is
    A = R_Y(2*theta_A) on the work qubit (amplitude a = sin^2(theta_A)). The
    controlled Grover powers are modelled by controlled phases (CU1) with
    binary-scaled angles, closed with an inverse QFT on the counting register.
    ``nqubits`` is ``n_counting + 1``.

    Args:
        n_counting: Number of counting qubits.
        theta_A: Angle controlling the amplitude. Default 0.4 rad.
    """

    n_counting: int = 3
    theta_A: float = 0.4
    nqubits: int = field(init=False)

    magic_parameters: ClassVar[Tuple[str, ...]] = ("theta_A",)
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("n_counting",)

    def __post_init__(self):
        self.nqubits = self.n_counting + 1
        super().__post_init__()

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        work = self.n_counting

        circuit = Circuit(n)
        circuit.add(gates.RY(work, theta=2 * self.theta_A))
        for q in range(self.n_counting):
            circuit.add(gates.H(q))

        for k in range(self.n_counting):
            circuit.add(gates.CU1(k, work, theta=2 * (2**k) * self.theta_A))

        for gate in _qft_circuit(self.n_counting).invert().queue:
            circuit.add(gate)
        return circuit


@dataclass(kw_only=True)
class TrotterIsing(_CompiledAnsatz):
    """First-order Trotter evolution of the transverse-field Ising model.

        H = -J sum_{(i,j) in E} Z_i Z_j - h sum_j X_j

    on the edges ``E`` of ``graph``; by default the open chain, in bond order.
    Each step applies ZZ rotations (via RZZ) then transverse-field RX rotations,
    producing a long structured sequence of magic rotations between Cliffords.

    Args:
        n_steps: Number of Trotter steps.
        dt: Time step.
        J: Coupling strength.
        h: Transverse field strength.
        graph: the interaction graph, any input of :meth:`.Topology.coerce`;
            RZZ on its edges, layer of disjoint bonds after layer. ``None``
            (default) is the open chain, bonds in order ``(0, 1), (1, 2), ...``.
    """

    n_steps: int = 4
    dt: float = 0.2
    J: float = 1.0
    h: float = 0.5
    graph: Optional[GraphLike] = None

    magic_parameters: ClassVar[Tuple[str, ...]] = ("dt", "J", "h")
    entanglement_parameters: ClassVar[Tuple[str, ...]] = ("n_steps", "dt", "J", "graph")

    def _build_circuit(self) -> Circuit:
        n = self.nqubits
        circuit = Circuit(n)
        topology = _topology(self.graph, n)
        if topology is None:
            bonds = [(j, j + 1) for j in range(n - 1)]
        else:
            bonds = [bond for layer in topology.layers for bond in layer]
        for _ in range(self.n_steps):
            for a, b in bonds:
                circuit.add(gates.RZZ(a, b, theta=-2 * self.J * self.dt))
            for j in range(n):
                circuit.add(gates.RX(j, theta=-2 * self.h * self.dt))
        return circuit
