"""
Resynthesis of a dressed-rotation chain into a runnable circuit.

A head of dressed Pauli rotations is turned into a hardware-native gate list
plus a pure-Clifford residual applied last::

    target = U(residual) . U(head)
    <target|O|target> = <head 0| residual^dag . O . residual |head 0>

The residual is therefore reabsorbed into the observable with a Clifford
simulator (:meth:`~mpstab.engines.StimEngine.fold_pauli_through_tableau`), never
with dense matrices.

Three synthesisers produce that pair:

- :func:`build_head_and_residual` uses rustiq's Pauli-network synthesis, which
  cancels gates across rotations. ``rustiq`` is optional and not on PyPI, so it
  is imported lazily; install it with ``pip install "mpstab[rustiq]"``.
- :func:`build_extraction_head_and_residual` is a Clifford-extraction synthesis
  (the idea of QuCLEAR, arXiv:2408.13316): each rotation is reduced to a single
  qubit by a Clifford that is never undone but pushed through the rotations
  still to come. It needs no extra dependency and can be told the device
  connectivity, so that the CNOTs it chooses stay local.
- :func:`build_cnot_ladder_head_and_residual` needs no extra dependency and costs
  ``2*(w-1)`` CNOTs per weight-``w`` rotation, with an identity residual.

Both return gate lists using two entry shapes: a placed rotation
``(axis, [qubit], angle)`` with ``axis`` in :data:`ROTATION_GATES`, or a Clifford
gate ``(name, qubits)`` with ``name`` in :data:`CLIFFORD_GATES`.
"""

from __future__ import annotations

import stim
from qibo import Circuit, gates

#: Clifford gate names appearing in a synthesised gate list, and their qibo gates.
CLIFFORD_GATES = {
    "H": gates.H,
    "S": gates.S,
    "Sd": gates.SDG,
    "SqrtX": gates.SX,
    "SqrtXd": gates.SXDG,
    "X": gates.X,
    "Y": gates.Y,
    "Z": gates.Z,
    "CNOT": gates.CNOT,
    "CZ": gates.CZ,
}

#: Rotation axis names appearing in a synthesised gate list, and their qibo gates.
ROTATION_GATES = {"RX": gates.RX, "RY": gates.RY, "RZ": gates.RZ}

#: The same Clifford names as ``stim`` tableaus, for tracking Pauli images.
CLIFFORD_TABLEAUS = {
    name: stim.Tableau.from_named_gate(stim_name)
    for name, stim_name in {
        "H": "H",
        "S": "S",
        "Sd": "S_DAG",
        "SqrtX": "SQRT_X",
        "SqrtXd": "SQRT_X_DAG",
        "CNOT": "CX",
        "CZ": "CZ",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
    }.items()
}

_AXIS_ROTATION = {1: "RX", 2: "RY", 3: "RZ"}


def _import_rustiq():
    try:
        from rustiq import Metric, pauli_network_synthesis
    except ImportError as error:
        raise ImportError(
            "rustiq is required for Pauli-network resynthesis. Install it with: "
            'pip install "mpstab[rustiq]"'
        ) from error
    return Metric, pauli_network_synthesis


def build_head_and_residual(
    paulis: list[str],
    angles: list[float],
    metric_name: str = "count",
    preserve_order: bool = True,
):
    """
    Synthesise a head circuit and its pure-Clifford residual with rustiq.

    ``rustiq.pauli_network_synthesis`` called with ``fix_clifford=False`` returns
    a skeleton that is a prefix of the one it returns with ``fix_clifford=True``.
    The prefix becomes the head; what the longer version appends is the residual.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        metric_name: rustiq metric, ``"count"`` or ``"depth"``.
        preserve_order: keep the (non-commuting) rotation order.

    Returns:
        ``(head, residual_tableau, residual_gates)``. The residual gate list is
        non-deterministic across runs (rustiq's ``fix_clifford=True`` draws
        randomly) even though the Clifford operator it implements is unique, so
        compare tableaus rather than gate lists.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    metric = Metric(metric_name)
    skeleton = pauli_network_synthesis(
        paulis, metric, preserve_order, fix_clifford=False
    )
    fixed = pauli_network_synthesis(paulis, metric, preserve_order, fix_clifford=True)
    if fixed[: len(skeleton)] != skeleton:
        raise RuntimeError(
            "rustiq returned a fix_clifford=True skeleton that does not extend "
            "the fix_clifford=False one; cannot split off a Clifford residual."
        )
    residual = fixed[len(skeleton) :]

    head = _place_rotations(skeleton, paulis, angles, nqubits)

    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


def _place_rotations(skeleton, paulis, angles, nqubits) -> list:
    """
    Interleave one single-qubit rotation per Pauli into a Clifford ``skeleton``.

    A rotation can be emitted once its Pauli image has collapsed onto a single
    qubit and no earlier unresolved rotation anticommutes with it. Each image is
    carried as a ``stim.PauliString`` and updated only when a skeleton gate
    touches its support; the anticommutation graph is precomputed as bitmasks so
    the blocking test is a counter rather than a scan.
    """
    n_rotations = len(paulis)
    x_mask = [0] * n_rotations
    z_mask = [0] * n_rotations
    for i, pauli in enumerate(paulis):
        for qubit, label in enumerate(pauli):
            if label in ("X", "Y"):
                x_mask[i] |= 1 << qubit
            if label in ("Z", "Y"):
                z_mask[i] |= 1 << qubit

    blockers = [0] * n_rotations
    blocks = [[] for _ in range(n_rotations)]
    for i in range(n_rotations):
        for j in range(i):
            anticommutes = (
                (x_mask[i] & z_mask[j]) ^ (z_mask[i] & x_mask[j])
            ).bit_count() & 1
            if anticommutes:
                blockers[i] += 1
                blocks[j].append(i)

    images = [stim.PauliString(pauli) for pauli in paulis]
    resolved = [False] * n_rotations
    head: list = []

    def emit(i):
        image = images[i]
        support = [q for q in range(nqubits) if image[q] != 0]
        qubit = support[0] if support else 0
        axis = image[qubit] if support else 3
        sign = 1 if image.sign == 1 else -1
        head.append((_AXIS_ROTATION[axis], [qubit], float(angles[i]) * sign))
        resolved[i] = True
        for later in blocks[i]:
            blockers[later] -= 1

    def drain():
        progress = True
        while progress:
            progress = False
            for i in range(n_rotations):
                if not resolved[i] and blockers[i] == 0 and images[i].weight <= 1:
                    emit(i)
                    progress = True

    drain()
    for name, qubits in skeleton:
        tableau = CLIFFORD_TABLEAUS[name]
        for i in range(n_rotations):
            if resolved[i]:
                continue
            image = images[i]
            if any(image[q] != 0 for q in qubits):
                images[i] = image.after(tableau, targets=qubits)
        head.append((name, qubits))
        drain()

    if not all(resolved):
        unplaced = [i for i, done in enumerate(resolved) if not done]
        raise RuntimeError(f"failed to place rotations at indices {unplaced}")
    return head


def build_cnot_ladder_head_and_residual(paulis: list[str], angles: list[float]):
    """
    Decompose every dressed rotation into an exact CNOT-ladder circuit.

    Uses the textbook identity ``exp(-i theta P / 2) = U^dag L^dag RZ(theta) L
    U``, where ``U`` rotates every non-identity site of ``P`` into the ``Z``
    frame and ``L`` is a CNOT ladder chaining the parity of the support onto its
    last qubit. Every gate is placed exactly, so nothing is left over and the
    residual is the identity -- at the cost of ``2*(w-1)`` CNOTs per weight-``w``
    rotation and no cancellation across rotations.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.

    Returns:
        ``(head, residual_tableau, residual_gates)``, the same shape as
        :func:`build_head_and_residual`, with an identity tableau and an empty
        residual gate list.
    """
    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    head: list[tuple] = []
    for pauli, angle in zip(paulis, angles):
        support = [q for q, label in enumerate(pauli) if label != "I"]
        if not support:
            continue  # exp(-i theta/2 I) is a global phase, no gate needed.

        for qubit in support:
            if pauli[qubit] == "Y":
                head.append(("Sd", [qubit]))
            if pauli[qubit] in ("X", "Y"):
                head.append(("H", [qubit]))

        ladder = list(zip(support[:-1], support[1:]))
        head.extend(("CNOT", [q0, q1]) for q0, q1 in ladder)
        head.append(("RZ", [support[-1]], float(angle)))
        head.extend(("CNOT", [q0, q1]) for q0, q1 in reversed(ladder))

        for qubit in support:
            if pauli[qubit] in ("X", "Y"):
                head.append(("H", [qubit]))
            if pauli[qubit] == "Y":
                head.append(("S", [qubit]))

    return head, stim.Tableau(nqubits), []


#: The inverse of each Clifford gate an extraction skeleton is made of.
_EXTRACTION_INVERSE = {"H": "H", "SqrtX": "SqrtXd", "CNOT": "CNOT"}

#: What :func:`build_extraction_head_and_residual` tries, keeping the cheapest:
#: ``locality`` weighs how far apart the sites of the coming strings end up (it
#: only matters with a connectivity), ``reorder`` lets a rotation overtake the
#: ones it commutes with, ``lookahead`` is how many coming strings a CNOT is
#: judged on. No single setting wins on every circuit.
EXTRACTION_STRATEGIES = (
    {"locality": 2.0, "reorder": True, "lookahead": 8},
    {"locality": 2.0, "reorder": False, "lookahead": 8},
    {"locality": 2.0, "reorder": False, "lookahead": 4},
    {"locality": 0.0, "reorder": True, "lookahead": 8},
    {"locality": 0.0, "reorder": False, "lookahead": 8},
)

#: A CNOT between qubits at distance ``d`` costs about ``d - 1`` SWAPs more.
_SWAP_COST = 3.0


def connectivity_distances(connectivity, nqubits: int):
    """
    All-pairs distances of a device connectivity, as a ``nqubits x nqubits``
    list, or ``None`` for all-to-all.

    Args:
        connectivity: ``None`` (all-to-all), ``"line"`` (qubit ``q`` next to
            ``q + 1``), or the edges ``[(q0, q1), ...]`` of the coupling graph.
        nqubits: number of qubits.
    """
    if connectivity is None:
        return None
    if isinstance(connectivity, str):
        if connectivity != "line":
            raise ValueError(
                f"Unknown connectivity {connectivity!r}: use 'line' or a list of edges."
            )
        edges = [(q, q + 1) for q in range(nqubits - 1)]
    else:
        edges = [tuple(edge) for edge in connectivity]
    neighbours = [[] for _ in range(nqubits)]
    for q0, q1 in edges:
        neighbours[q0].append(q1)
        neighbours[q1].append(q0)
    distances = []
    for source in range(nqubits):
        row = [None] * nqubits
        row[source] = 0
        frontier = [source]
        while frontier:
            following = []
            for qubit in frontier:
                for other in neighbours[qubit]:
                    if row[other] is None:
                        row[other] = row[qubit] + 1
                        following.append(other)
            frontier = following
        if any(d is None for d in row):
            raise ValueError("The connectivity graph is not connected.")
        distances.append(row)
    return distances


def _extraction_skeleton(paulis, distances, locality, reorder, lookahead, discount=0.7):
    """
    The Clifford skeleton of one extraction strategy, and its estimated cost in
    two-qubit gates on the device (a CNOT at distance ``d`` counts as
    ``1 + 3 * (d - 1)``).

    Every string is carried as two bit masks (``x`` and ``z`` parts). For the
    rotation in turn: single-qubit Cliffords take its string to all-``Z``, then
    CNOTs between two sites of its support remove one site each until one is
    left. Each CNOT is the one that leaves the coming strings touching those
    sites lightest (and, with a connectivity, most local).
    """
    nqubits = len(paulis[0])
    xs = [sum(1 << q for q, c in enumerate(p) if c in "XY") for p in paulis]
    zs = [sum(1 << q for q, c in enumerate(p) if c in "ZY") for p in paulis]
    remaining = list(range(len(paulis)))
    skeleton: list = []
    estimated_cost = 0.0
    spread_cache: dict = {}

    def spread(mask):
        """Length of a tree joining the support of ``mask`` on the device."""
        if distances is None:
            return 0
        if mask not in spread_cache:
            sites = [q for q in range(nqubits) if (mask >> q) & 1]
            joined, total = sites[:1], 0
            pending = sites[1:]
            while pending:  # Prim: attach the closest site left
                best = min(pending, key=lambda q: min(distances[q][j] for j in joined))
                total += min(distances[best][j] for j in joined)
                joined.append(best)
                pending.remove(best)
            spread_cache[mask] = total
        return spread_cache[mask]

    def string_cost(mask):
        weight = mask.bit_count()
        if weight < 2 or not locality or distances is None:
            return weight
        return weight + locality * (spread(mask) - (weight - 1))

    def apply(name, qubits, strings):
        skeleton.append((name, list(qubits)))
        if name == "H":
            bit = 1 << qubits[0]
            for i in strings:
                if bool(xs[i] & bit) != bool(zs[i] & bit):
                    xs[i] ^= bit
                    zs[i] ^= bit
        elif name == "SqrtX":  # X -> X, Y -> Z, Z -> Y, up to signs
            bit = 1 << qubits[0]
            for i in strings:
                if zs[i] & bit:
                    xs[i] ^= bit
        else:
            control, target = 1 << qubits[0], 1 << qubits[1]
            for i in strings:
                if xs[i] & control:
                    xs[i] ^= target
                if zs[i] & target:
                    zs[i] ^= control

    def anticommute(i, j):
        return ((xs[i] & zs[j]).bit_count() + (zs[i] & xs[j]).bit_count()) & 1

    while remaining:
        pick = 0
        if reorder:  # a rotation may overtake the ones it commutes with
            front = [
                k
                for k, i in enumerate(remaining[:24])
                if not any(anticommute(i, j) for j in remaining[:k])
            ]
            # the cheapest of them first: lightest, and with a connectivity
            # most local, so that a commuting block is reduced from its easy end
            pick = min(
                front,
                key=lambda k: (string_cost(xs[remaining[k]] | zs[remaining[k]]), k),
            )
        current = remaining.pop(pick)
        strings = [current] + remaining
        support = [q for q in range(nqubits) if ((xs[current] | zs[current]) >> q) & 1]
        for qubit in support:
            has_x, has_z = (xs[current] >> qubit) & 1, (zs[current] >> qubit) & 1
            if has_x and has_z:
                apply("SqrtX", [qubit], strings)
            elif has_x:
                apply("H", [qubit], strings)

        while len(support) > 1:
            mask = sum(1 << q for q in support)
            touching = [j for j in remaining[:400] if (xs[j] | zs[j]) & mask]
            touching = touching[: 4 * lookahead]
            best = None
            for control in support:
                for target in support:
                    if control == target:
                        continue
                    c_bit, t_bit = 1 << control, 1 << target
                    cost, seen = 0.0, 0
                    for j in touching:
                        x, z = xs[j], zs[j]
                        if not (x | z) & (c_bit | t_bit):
                            continue
                        before = string_cost(x | z)
                        if x & c_bit:
                            x ^= t_bit
                        if z & t_bit:
                            z ^= c_bit
                        cost += discount**seen * (string_cost(x | z) - before)
                        seen += 1
                        if seen == lookahead:
                            break
                    if distances is not None:
                        cost += locality * (distances[control][target] - 1)
                    if best is None or cost < best[0]:
                        best = (cost, control, target)
            _, control, target = best
            apply("CNOT", [control, target], strings)
            support.remove(control)
            estimated_cost += 1.0
            if distances is not None:
                estimated_cost += _SWAP_COST * (distances[control][target] - 1)
    return skeleton, estimated_cost


def build_extraction_head_and_residual(
    paulis: list[str],
    angles: list[float],
    connectivity=None,
    strategies=EXTRACTION_STRATEGIES,
):
    """
    Synthesise a head circuit and its pure-Clifford residual by Clifford
    extraction.

    A rotation of a weight-``w`` string needs ``w - 1`` CNOTs to bring it onto
    one qubit, and the textbook circuit spends as many again to undo them.
    Here they are not undone: the Clifford is pushed through the rotations
    still to come, whose strings change accordingly, and what has accumulated
    at the end is the residual, folded into the observable. This is the idea
    of QuCLEAR (Liu et al., arXiv:2408.13316); the choice of each CNOT is our
    own greedy rule, see :func:`_extraction_skeleton`.

    Every strategy of ``strategies`` is run and the cheapest head kept, cost
    being the two-qubit gates the device would need: with a ``connectivity`` a
    CNOT between distant qubits is charged the SWAPs that route it.

    Args:
        paulis: dressed Pauli axis strings (qubit-0-leftmost), in circuit order.
        angles: matching rotation angles.
        connectivity: ``None`` (all-to-all), ``"line"``, or the edges
            ``[(q0, q1), ...]`` of the device coupling graph, in the qubit
            numbering of the strings.
        strategies: the settings to try, see :data:`EXTRACTION_STRATEGIES`.

    Returns:
        ``(head, residual_tableau, residual_gates)``, the same shape as
        :func:`build_head_and_residual`. Deterministic.
    """
    if not paulis:
        return [], stim.Tableau(0), []

    nqubits = len(paulis[0])
    distances = connectivity_distances(connectivity, nqubits)
    best = None
    tried = set()
    for strategy in strategies:
        if distances is None:  # locality has no meaning: skip the duplicates
            strategy = {**strategy, "locality": 0.0}
        key = tuple(sorted(strategy.items()))
        if key in tried:
            continue
        tried.add(key)
        skeleton, cost = _extraction_skeleton(paulis, distances, **strategy)
        if best is None or cost < best[0]:
            best = (cost, skeleton)
    skeleton = best[1]

    head = _place_rotations(skeleton, paulis, angles, nqubits)
    residual = [
        (_EXTRACTION_INVERSE[name], qubits) for name, qubits in reversed(skeleton)
    ]
    residual_tableau = stim.Tableau(nqubits)
    for name, qubits in residual:
        residual_tableau.append(CLIFFORD_TABLEAUS[name], qubits)
    return head, residual_tableau, residual


def head_counts_only(
    paulis: list[str], metric_name: str = "count", preserve_order: bool = True
):
    """
    Gate counts of the rustiq head without placing the rotations.

    The head is the rustiq Clifford skeleton with one single-qubit rotation
    interleaved per Pauli, so the total is ``len(skeleton) + len(paulis)`` and
    the two-qubit count is the skeleton's. One rustiq call, much cheaper than
    :func:`build_head_and_residual` and exact for counting.

    Returns:
        ``(total_gates, two_qubit_gates)``.
    """
    Metric, pauli_network_synthesis = _import_rustiq()

    if not paulis:
        return 0, 0
    skeleton = pauli_network_synthesis(
        paulis, Metric(metric_name), preserve_order, fix_clifford=False
    )
    two_qubit = sum(1 for gate in skeleton if len(gate[1]) == 2)
    return len(skeleton) + len(paulis), two_qubit


def head_to_qibo_circuit(head: list[tuple], nqubits: int) -> Circuit:
    """Convert a synthesised gate list into a runnable qibo ``Circuit``."""
    circuit = Circuit(nqubits)
    for entry in head:
        if len(entry) == 3:
            axis, qubits, angle = entry
            circuit.add(ROTATION_GATES[axis](qubits[0], theta=angle))
        else:
            name, qubits = entry
            circuit.add(CLIFFORD_GATES[name](*qubits))
    return circuit


def count_two_qubit_gates(head: list[tuple]) -> int:
    """Number of two-qubit gates in a synthesised gate list."""
    return sum(1 for entry in head if len(entry) == 2 and len(entry[1]) == 2)
