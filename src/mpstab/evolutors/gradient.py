"""Analytic gradients of the HSMPO surrogate, from the sign-flip rule of its faults.

In the Clifford frame the state is ``|psi> = R_R ... R_1 |init>``, with dressed
rotations ``R_j = exp(-i phi_j/2 G_j)``, and the observable is
``sum_t c_t s_t P_t`` once dressed. Since ``dR_j/dphi_j = -(i/2) G_j R_j``,
and ``R_k G_j = G_j R~_k`` with ``R~_k`` the rotation at angle ``-phi_k`` when
``G_k`` anticommutes with ``G_j`` and ``R_k`` itself otherwise::

    df/dphi_j = sum_t c_t s_t Im <psi| P_t G_j |psi~_j>,

where ``|psi~_j>`` is the chain with the angles of the later rotations
anticommuting with ``G_j`` negated: the sign pattern a Pauli fault ``G_j``
right after rotation ``j`` produces in :mod:`mpstab.models.noise`. It equals the
parameter-shift rule exactly, with one transition amplitude per term in place
of two shifted circuits. Read backwards it is the adjoint method: with
``|psi_j>`` the state after rotation ``j`` and ``T_j`` the rest of the chain,
``G_j T~_j = T_j G_j`` turns each term into ``Im <lambda_j| G_j |psi_j>``, where
the costate ``lambda_j = T_j^dag O |psi>`` follows from one backward pass.

A circuit parameter ``theta`` reaches the chain through the dressing sign,
``phi_j = sign_j theta``, so ``df/dtheta`` sums ``sign_j df/dphi_j`` over the
rotations of the parameter's gate: one per magic gate, and one per occurrence
of a gate object queued several times.

Not covered here: gradients under noise (an average over fault patterns of
the XOR of fault and parameter rows), and the parameters of the head that
:class:`~mpstab.evolutors.hqsmpo.HQSMPO` runs on hardware, which
still need shifted circuits.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mpstab.evolutors.optimization import adjoint_kernel
from mpstab.evolutors.utils import gate_angle
from mpstab.hamiltonians import Observable, pauli_terms
from mpstab.pauli import (
    anticommutation,
    multiply_codes,
    pauli_codes,
    pauli_labels,
    pauli_light_cone,
)

#: ``i**k`` for ``k = 0, 1, 2, 3``.
_POWERS_OF_I = np.array([1, 1j, -1, -1j])


@dataclass
class AnalyticGradient:
    """
    Result of :func:`analytic_gradient`.

    Attributes
    ----------
    values : numpy.ndarray
        The derivative of the expectation value with respect to each requested
        parameter, in the order requested.
    value : float
        The expectation value itself, on the same (light-cone-restricted)
        chain, so a training step needs no separate evaluation.
    n_rotations : int
        Rotations applied to an MPS: the base chain plus every shifted tail.
    fidelities : numpy.ndarray
        Truncation fidelity of the base state, then, with ``"adjoint"``, the
        weight the costate kept through its backward pass, or, with
        ``"flips"``, the fidelity of each shifted state in the order they were
        built. With a bond-dimension cap, the gradient error is set by how far
        these fall below 1.
    """

    values: np.ndarray
    value: float
    n_rotations: int
    fidelities: np.ndarray


def analytic_gradient(
    hsmpo,
    observable: Observable,
    parameters: Sequence[int] | None = None,
    light_cone: bool = True,
    method: str = "adjoint",
    costate_bond: int | None = None,
    groups: int = 1,
    n_workers: int | None = None,
) -> AnalyticGradient:
    """
    Gradient of ``hsmpo.expectation(observable)`` with respect to the circuit
    parameters, as returned by ``hsmpo.get_parameters()``.

    Two evaluations of the same identity are available:

    - ``"adjoint"``: since ``G_j T~_j = T_j G_j`` for the tail ``T_j`` after
      rotation ``j``, each derivative is ``Im <lambda_j| G_j |psi_j>`` with
      ``psi_j`` the checkpoint after rotation ``j`` and
      ``lambda_j = T_j^dag O |psi>``. One forward pass stores the
      checkpoints, and one backward pass of the costate ``lambda`` gives every
      derivative: about ``2 R`` rotations for ``R`` rotations in the chain,
      whatever the number of parameters. ``O |psi>`` is a sum of states, so
      the costate is compressed, by default to twice the state's bond cap.
      Beyond that the error is set by the state's own truncation: like any
      gradient built from overlaps, it is first order in the state error,
      where an energy (and so the parameter-shift rule) is second order.
    - ``"flips"``: each derivative is ``Im <psi| O G_j |psi~_j>``, with
      ``psi~_j`` the chain under the sign pattern of a fault ``G_j``. Each
      distinct pattern re-applies only the tail after its first flipped
      rotation, visited in lexicographic order so consecutive patterns share
      their prefix, and a rotation that commutes with every later one costs one
      expectation value on the base state. It needs no costate, so at equal
      bond dimension it is the more accurate, but it applies ``O(P R)``
      rotations for ``P`` parameters.

    Parameters
    ----------
    hsmpo : HSMPO
        The surrogate, with its parameters already set.
    observable : Observable
        A Pauli string or any other format
        :func:`~mpstab.hamiltonians.pauli_terms` accepts.
    parameters : sequence of int, optional
        Indices of the parameters to differentiate, into
        ``hsmpo.get_parameters()``. All of them by default.
    light_cone : bool, optional
        Drop the rotations outside the observable's light cone
        (:func:`~mpstab.pauli.pauli_light_cone`) first. This is exact: a parameter
        whose rotations all fall outside has zero derivative.
    method : str, optional
        ``"adjoint"`` (default) or ``"flips"``.
    costate_bond : int, optional
        Bond-dimension cap of the costate with ``"adjoint"``. Defaults to
        twice ``hsmpo.max_bond_dimension``, and to no cap when the state has
        none. Capping it at the state's own bond is about twice as fast, for
        roughly twice the gradient error; a larger cap gains nothing, since
        the state's truncation then dominates.
    groups : int, optional
        Split the observable's terms into this many blocks of neighbouring
        terms (by the centre of their support) and differentiate each block on
        its own light cone; the gradient is linear in the terms, so the sum is
        exact. A block's cone is narrower than the union of all of them and
        carries less entanglement: once the union saturates the bond cap,
        blocks are faster and, at the same cap, more accurate. While the bond
        stays below the cap the union is cheaper, since the blocks' cones
        overlap. One block (default) uses the union.
    n_workers : int, optional
        Processes the blocks run on, ``min(groups, os.cpu_count())`` by
        default. The pool is started on first use, which costs about a second
        per process, and kept for later calls with the same ``n_workers``, so a
        training loop pays it once. Each process holds the checkpoints of its
        blocks, so fewer workers bound the memory.

    Returns
    -------
    AnalyticGradient

    Raises
    ------
    ValueError
        If a requested parameter has no rotation in the chain: its gate sits
        at a Clifford angle (a multiple of pi/2, zero included) and was
        absorbed into the tableau, or the ansatz builds its rotations from
        copies of its gates. Also on an unknown ``method``.
    NotImplementedError
        If the tensor-network engine does not compute transition amplitudes,
        or, with ``"adjoint"``, Pauli sums applied to a state.

    Examples
    --------
    ::

        from qibo import Circuit, gates
        from mpstab import HSMPO

        circuit = Circuit(2)
        circuit.add(gates.RY(0, theta=0.4))
        circuit.add(gates.CNOT(0, 1))
        circuit.add(gates.RX(1, theta=0.3))
        print(HSMPO(circuit).gradient("ZZ"))
    """
    if method not in ("adjoint", "flips"):
        raise ValueError(f"Unknown method {method!r}, expected 'adjoint' or 'flips'.")
    nqubits = hsmpo.nqubits
    trainable = hsmpo.ansatz.circuit.trainable_gates
    parameters = list(range(len(trainable))) if parameters is None else list(parameters)

    # Each parameter's rotations in the chain, and their dressing signs.
    dressed = hsmpo._dress_generators(hsmpo.magic_gates, hsmpo.clifford_circuit)
    rotations_of_gate = defaultdict(list)
    for j, (_, gate) in enumerate(hsmpo.magic_gates):
        rotations_of_gate[id(gate)].append(j)
    rotations_of = {p: rotations_of_gate[id(gate)] for p, gate in enumerate(trainable)}
    _check_parameters(parameters, rotations_of, trainable)

    constant, terms = pauli_terms(observable, nqubits)
    generators = [generator for generator, _ in dressed]
    angles = np.array(
        [
            sign * gate_angle(gate)
            for (_, sign), (_, gate) in zip(dressed, hsmpo.magic_gates)
        ],
        dtype=float,
    )
    wanted = sorted({j for p in parameters for j in rotations_of[p]})
    if costate_bond is None and hsmpo.max_bond_dimension is not None:
        costate_bond = 2 * hsmpo.max_bond_dimension

    chains = []
    for group in _neighbour_groups(list(terms), groups):
        backpropagated = hsmpo.stab_engine.backpropagate_many(
            {pauli: terms[pauli] for pauli in group}, hsmpo.clifford_circuit
        )
        weights = np.array(
            [terms[pauli] * s for pauli, (_, s) in zip(group, backpropagated)]
        )
        observables = [pauli for pauli, _ in backpropagated]
        kept = np.arange(len(generators))
        if light_cone:
            kept = pauli_light_cone(generators, observables, nqubits)
        chains.append(
            (
                kept,
                [generators[j] for j in kept],
                angles[kept],
                np.flatnonzero(np.isin(kept, wanted)).tolist(),
                observables,
                weights,
            )
        )

    shared = (hsmpo.tn_engine, hsmpo.initial_mps(), hsmpo.max_bond_dimension)
    jobs = [(*shared, *chain[1:], method, costate_bond) for chain in chains]
    if n_workers is None:
        n_workers = min(groups, os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(jobs)))
    if n_workers > 1:
        try:
            pool = _pool(n_workers, hsmpo.tn_engine.worker_context())
            outcomes = list(pool.map(_on_chain, *zip(*jobs)))
        except BrokenProcessPool:
            _POOL.clear()  # start a fresh pool on the next call
            raise
    else:
        outcomes = [_on_chain(*job) for job in jobs]

    derivatives = np.zeros(len(generators))
    value, n_rotations, fidelities = constant, 0, []
    for chain, (
        group_value,
        group_derivatives,
        group_rotations,
        group_fidelities,
    ) in zip(chains, outcomes):
        derivatives[chain[0]] += group_derivatives
        value += group_value
        n_rotations += group_rotations
        fidelities.extend(group_fidelities)

    values = np.array(
        [
            sum(dressed[j][1] * derivatives[j] for j in rotations_of[p])
            for p in parameters
        ]
    )
    return AnalyticGradient(
        values=values,
        value=value,
        n_rotations=n_rotations,
        fidelities=np.array(fidelities),
    )


def _neighbour_groups(paulis, groups):
    """
    ``paulis`` split into ``groups`` blocks of consecutive terms, after sorting
    them by the centre of their support, so each block sits in one region.
    """
    groups = max(1, min(groups, len(paulis)))
    if groups == 1:
        return [paulis]
    order = np.argsort([_support_centre(pauli) for pauli in paulis], kind="stable")
    return [[paulis[i] for i in block] for block in np.array_split(order, groups)]


def _support_centre(pauli: str) -> float:
    """
    Centre of the shortest arc of the qubit ring that covers the support, so a
    term closing a periodic chain, such as ``Z_{n-1} Z_0``, sits at its end
    rather than in the middle.
    """
    n = len(pauli)
    sites = [q for q, label in enumerate(pauli) if label != "I"] or [0]
    gaps = np.diff(sites + [sites[0] + n])
    widest = int(np.argmax(gaps))
    start = sites[(widest + 1) % len(sites)]
    return (start + (n - gaps[widest]) / 2) % n


#: The process pool of :func:`analytic_gradient`, kept across calls.
_POOL: dict = {}


def _pool(n_workers: int, context=None) -> ProcessPoolExecutor:
    """
    A pool of ``n_workers`` processes started in ``context`` (the engine's
    :meth:`~mpstab.engines.TensorNetworkEngine.worker_context`), started once
    and reused while both stay the same.
    """
    key = (n_workers, None if context is None else context.get_start_method())
    if _POOL.get("key") != key:
        if _POOL:
            _POOL["executor"].shutdown()
        _POOL.update(
            key=key,
            executor=ProcessPoolExecutor(n_workers, mp_context=context),
        )
    return _POOL["executor"]


def _on_chain(
    engine,
    initial_mps,
    max_bond,
    generators,
    angles,
    sites,
    observables,
    weights,
    method,
    costate_bond,
):
    """
    Value and derivatives of ``sum_t weights_t <observables_t>`` on one
    rotation chain; the derivatives are filled at ``sites`` only.
    """
    derivatives = np.zeros(len(generators))
    if (
        method == "adjoint"
        and sites
        and observables
        and engine.has_site_view
        and engine.has_pauli_sum
        and len(generators[0]) > 1
    ):
        return adjoint_kernel(
            engine,
            initial_mps,
            generators,
            angles,
            sites,
            observables,
            weights,
            max_bond,
            costate_bond,
        )
    chain = _Chain(engine, initial_mps, max_bond, generators, angles)
    base = chain.state(np.zeros(len(generators), dtype=bool))
    value = 0.0
    if observables:
        value = float(
            np.real(
                np.dot(
                    weights,
                    np.asarray(engine.pauli_expvals(base, observables), complex),
                )
            )
        )
    fidelities = []
    if sites and observables:
        if method == "adjoint":
            fidelities = _adjoint(
                chain, sites, observables, weights, costate_bond, derivatives
            )
        else:
            _flips(chain, sites, observables, weights, derivatives)
    return value, derivatives, chain.n_rotations, chain.fidelities + fidelities


def _adjoint(chain, sites, observables, weights, costate_bond, derivatives):
    """
    Fill ``derivatives[sites]`` by a backward pass of the costate
    ``lambda = O |psi>``; returns the costate's truncation fidelity, as a
    one-element list.
    """
    engine, generators = chain.engine, chain.generators
    state = chain.state(np.zeros(len(generators), dtype=bool))
    checkpoints = list(chain.checkpoints)
    norm = chain.fidelities[-1]
    wanted = set(sites)
    costate = engine.apply_pauli_sum(state, observables, weights, costate_bond)
    initial = np.real(engine.squared_norm(costate))
    for j in range(len(generators) - 1, sites[0] - 1, -1):
        if j < len(generators) - 1:
            engine.apply_rotation(
                costate,
                engine.rotation_operator(generators[j + 1], -chain.angles[j + 1]),
                costate_bond,
            )
            chain.n_rotations += 1
        if j in wanted:
            amplitude = engine.pauli_transitions(
                costate, checkpoints[j + 1], [generators[j]], normalize=False
            )[0]
            derivatives[j] = np.imag(complex(amplitude)) / norm
    return [np.real(engine.squared_norm(costate)) / initial if initial > 0 else 1.0]


def _flips(chain, sites, observables, weights, derivatives):
    """Fill ``derivatives[sites]`` from the fault sign pattern of each rotation."""
    engine, generators = chain.engine, chain.generators
    nqubits = len(generators[0])
    flips = anticommutation([generators[k] for k in sites], generators, nqubits)
    flips &= np.arange(len(generators))[None, :] > np.array(sites)[:, None]
    base = chain.state(np.zeros(len(generators), dtype=bool))
    observable_codes = pauli_codes(observables)

    def transitions(state, rows):
        paulis, phases = [], []
        for k in rows:
            # O . G for every observable O, with its phase.
            codes, phase = multiply_codes(
                observable_codes, pauli_codes(generators[sites[k]])
            )
            paulis += pauli_labels(codes)
            phases += list(_POWERS_OF_I[phase])
        if state is base:
            # psi~ = psi: only the anticommuting terms, whose product is
            # anti-Hermitian, have an imaginary part.
            values = np.zeros(len(paulis), dtype=complex)
            imaginary = np.flatnonzero(np.imag(phases) != 0)
            if imaginary.size:
                values[imaginary] = engine.pauli_expvals(
                    state, [paulis[i] for i in imaginary]
                )
        else:
            values = np.asarray(
                engine.pauli_transitions(base, state, paulis), dtype=complex
            )
        values = np.imag(np.array(phases) * values).reshape(len(rows), -1)
        derivatives[[sites[k] for k in rows]] = values @ weights

    empty = np.flatnonzero(~flips.any(axis=1))
    if empty.size:
        transitions(base, empty)
    rows = np.flatnonzero(flips.any(axis=1))
    if rows.size:
        patterns, inverse = np.unique(flips[rows], axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for u, pattern in enumerate(patterns):
            transitions(chain.state(pattern), rows[inverse == u])


def _check_parameters(parameters, rotations_of, trainable):
    """Raise on a requested parameter that has no rotation to differentiate."""
    for p in parameters:
        if p not in rotations_of:
            raise ValueError(
                f"Parameter {p} is out of range: the circuit has "
                f"{len(trainable)} trainable gates."
            )
        if rotations_of[p]:
            continue
        gate = trainable[p]
        if gate.clifford:
            reason = (
                f"its gate {gate.name}({', '.join(f'{a:.6g}' for a in gate.parameters)}) "
                "is at a Clifford angle and was absorbed into the tableau. Move it "
                "off the Clifford angles, or differentiate it with the "
                "parameter-shift rule"
            )
        else:
            reason = (
                f"its gate {gate.name} is not among the HSMPO's magic gates; the "
                "ansatz builds its rotations from copies of its gates"
            )
        raise ValueError(f"Parameter {p} has no rotation in the chain: {reason}.")


class _Chain:
    """
    The rotation chain under sign patterns visited in lexicographic order:
    ``checkpoints[j]`` is the state after the first ``j`` rotations of the
    current pattern, so a pattern only re-applies the rotations from its first
    difference with the previous one. Each rotation's operator is built once
    per sign.
    """

    def __init__(self, engine, initial_mps, max_bond, generators, angles):
        self.engine = engine
        self.max_bond = max_bond
        self.generators, self.angles = generators, angles
        self.checkpoints = [initial_mps]
        self.operators = {}
        self.current = None
        self.n_rotations = 0
        self.fidelities = []

    def _rotation(self, j: int, flipped: bool):
        if (j, flipped) not in self.operators:
            angle = -self.angles[j] if flipped else self.angles[j]
            self.operators[j, flipped] = self.engine.rotation_operator(
                self.generators[j], angle
            )
        return self.operators[j, flipped]

    def state(self, pattern):
        """The final state of ``pattern``, which becomes the current one."""
        if self.current is not None and np.array_equal(pattern, self.current):
            return self.checkpoints[-1]
        start = (
            0
            if self.current is None
            else int(np.argmax(np.asarray(pattern) != self.current))
        )
        del self.checkpoints[start + 1 :]
        for j in range(start, len(self.generators)):
            state = self.engine.copy_state(self.checkpoints[-1])
            self.engine.apply_rotation(
                state, self._rotation(j, bool(pattern[j])), self.max_bond
            )
            self.checkpoints.append(state)
        self.current = np.array(pattern, dtype=bool)
        self.n_rotations += len(self.generators) - start
        state = self.checkpoints[-1]
        self.fidelities.append(float(np.real(self.engine.squared_norm(state))))
        return state
