"""Optimization of an HSMPO's MPS with DMRG, and the fused adjoint kernel of
its analytic gradient."""

from typing import Union

import numpy as np

from mpstab.engines.tensor_networks import _mps_kernels as kernels
from mpstab.hamiltonians import Observable, pauli_terms


def minimize_expectation_dmrg(
    hsmpo,
    observables: Observable,
    bond_dims: Union[int, list] = None,
    cutoff: float = 1e-9,
    tol: float = 1e-6,
    max_sweeps: int = 10,
    verbosity: int = 1,
):
    """
    Minimize an observable over MPS tensors with two-site DMRG.

    Every Hamiltonian term is first backpropagated through the HSMPO's Clifford
    circuit, which is what makes the MPO cheaper here than in plain DMRG: the
    Clifford part is absorbed into the observable rather than represented. The
    HSMPO's cached MPS is the starting guess, and is replaced by the optimized
    state on return. The sweeps themselves run on the tensor-network engine
    (:meth:`~mpstab.engines.TensorNetworkEngine.minimize_energy`).

    Args:
        hsmpo: the :class:`~mpstab.evolutors.hsmpo.HSMPO` to optimize.
        observables: the Hamiltonian, in any format
            :func:`~mpstab.hamiltonians.pauli_terms` accepts.
        bond_dims: max bond dimension per sweep, or one value for all. Defaults to
            a gradual ``[10, 20, 50, 100, 200]`` growth.
        cutoff: SVD truncation cutoff.
        tol: energy convergence tolerance.
        max_sweeps: maximum number of sweeps.
        verbosity: DMRG verbosity, 0 to 2.

    Returns:
        A dict with ``ground_state``, ``energy``, ``converged``, ``num_sweeps``
        and ``energy_history``. Energies include the Hamiltonian's constant
        offset and identity terms.

    Raises:
        ValueError: if the Hamiltonian has no non-identity terms.
    """
    constant, terms = pauli_terms(observables, hsmpo.nqubits)
    constant += terms.pop("I" * hsmpo.nqubits, 0.0)

    if bond_dims is None:
        bond_dims = [10, 20, 50, 100, 200]
    elif isinstance(bond_dims, int):
        bond_dims = [bond_dims]

    # Fold the Clifford part into the observable: U^dag P U for every term.
    evolved = hsmpo.stab_engine.backpropagate_many(terms, hsmpo.clifford_circuit)
    backpropagated = {
        pauli: sign * coefficient
        for (pauli, sign), coefficient in zip(evolved, terms.values())
    }

    if not backpropagated:
        raise ValueError("The Hamiltonian has no non-identity terms to optimize.")

    tn_engine = hsmpo.tn_engine
    result = tn_engine.minimize_energy(
        tn_engine.pauli_sum_mpo(backpropagated),
        hsmpo.original_circuit_mps,
        bond_dims=bond_dims,
        cutoff=cutoff,
        tol=tol,
        max_sweeps=max_sweeps,
        verbosity=verbosity,
    )
    energy_history = [e + constant for e in result["energy_history"]]
    hsmpo.original_circuit_mps = result["state"]

    return {
        "ground_state": result["state"],
        "energy": result["energy"] + constant,
        "converged": result["converged"],
        "num_sweeps": len(energy_history) if energy_history else max_sweeps,
        "energy_history": energy_history,
    }


def adjoint_kernel(
    engine,
    initial_state,
    generators,
    angles,
    sites,
    observables,
    weights,
    max_bond,
    costate_bond,
):
    """
    The expectation value and its derivative with respect to each rotation
    angle of a dressed-rotation chain, in one forward and one backward pass.

    With ``psi_j`` the state after rotation ``j``, ``T_j`` the rest of the
    chain and ``O = sum_t w_t P_t``, the derivative with respect to the angle
    of rotation ``j`` is ``Im <lambda_j| G_j |psi_j> / <psi|psi>``, with the
    costate ``lambda_j = T_j^dag O |psi>``. The forward pass keeps every
    ``psi_j``; the backward pass applies the inverse rotations to the
    costate and takes one transition amplitude per rotation in ``sites``.

    Everything runs on the windowed kernels of :mod:`._mps_kernels`, on plain
    lists of site arrays: a checkpoint is a shallow copy of the list, since the
    kernels replace arrays rather than write into them.

    Args:
        engine: a tensor-network engine with a site view, state import and
            Pauli sums (:attr:`~mpstab.engines.TensorNetworkEngine.has_site_view`,
            :attr:`~mpstab.engines.TensorNetworkEngine.has_pauli_sum`); the
            kernels run on its arrays (:attr:`~mpstab.engines.TensorNetworkEngine.linalg`).
        initial_state: the initial MPS, left untouched.
        generators, angles: the chain, ``exp(-i angle/2 generator)`` in order.
        sites: increasing indices of the rotations to differentiate.
        observables, weights: the dressed observable ``sum_t w_t P_t``.
        max_bond: bond cap of the state.
        costate_bond: bond cap of the costate.

    Returns:
        ``(value, derivatives, n_rotations, fidelities)``: the normalized
        expectation value, the derivative per rotation (zero outside
        ``sites``), the rotations applied, and the truncation fidelities of
        the state and of the costate.
    """
    cutoff = engine.truncation_cutoff
    linalg = engine.linalg
    arrays, center = engine.site_arrays(initial_state)
    checkpoints = [(list(arrays), center)]
    for generator, angle in zip(generators, angles):
        center = kernels.pauli_rotation(
            arrays, center, generator, angle, max_bond, cutoff, linalg
        )
        checkpoints.append((list(arrays), center))
    norm = kernels.squared_norm(arrays, center, linalg)
    value = (
        float(
            np.real(
                np.dot(
                    weights, _expectations(list(arrays), center, observables, linalg)
                )
            )
        )
        / norm
    )

    costate, costate_center = engine.site_arrays(
        engine.apply_pauli_sum(
            engine.state_from_sites(arrays, center), observables, weights, costate_bond
        )
    )
    initial = kernels.squared_norm(costate, costate_center, linalg)
    derivatives = np.zeros(len(generators))
    wanted = set(sites)
    n_rotations = len(generators)
    for j in range(len(generators) - 1, min(wanted) - 1, -1):
        if j < len(generators) - 1:
            costate_center = kernels.pauli_rotation(
                costate,
                costate_center,
                generators[j + 1],
                -angles[j + 1],
                costate_bond,
                cutoff,
                linalg,
            )
            n_rotations += 1
        if j in wanted:
            amplitude = kernels.pauli_transitions(
                costate, checkpoints[j + 1][0], [generators[j]], linalg
            )[0]
            derivatives[j] = amplitude.imag / norm
    kept = (
        kernels.squared_norm(costate, costate_center, linalg) / initial
        if initial
        else 1.0
    )
    return value, derivatives, n_rotations, [norm, kept]


def _expectations(arrays, center, paulis, linalg=kernels.NUMPY):
    """
    Unnormalized ``<psi|P|psi>`` for each string, visiting them in order of
    their first non-identity site so the center sweeps the chain once.
    """
    values = np.zeros(len(paulis), dtype=complex)
    windows = [kernels.bounds(pauli) for pauli in paulis]
    for k in sorted(
        range(len(paulis)),
        key=lambda k: center if windows[k] is None else windows[k][0],
    ):
        window = windows[k]
        if window is not None and not window[0] <= center <= window[1]:
            center = kernels.move_center(arrays, center, window[0], linalg)
        values[k] = kernels.pauli_expectation(arrays, center, paulis[k], linalg)
    return values
