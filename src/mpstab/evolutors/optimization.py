"""DMRG ground-state optimization of an HSMPO's MPS."""

import copy
from typing import Union

import numpy as np

from mpstab.hamiltonians import Observable, pauli_terms


def build_pauli_mpo(hsmpo, terms: dict):
    """
    Sum a ``{pauli_string: coefficient}`` map into one MPO.

    Identity terms are skipped: they contribute a constant energy offset, not an
    operator DMRG can optimize against.

    Raises:
        ValueError: if every term is the identity.
    """
    hamiltonian_mpo = None
    for pauli, coefficient in terms.items():
        if set(pauli) == {"I"}:
            continue
        term_mpo = coefficient * hsmpo.tn_engine.pauli_mpo(pauli)
        hamiltonian_mpo = (
            term_mpo if hamiltonian_mpo is None else hamiltonian_mpo + term_mpo
        )

    if hamiltonian_mpo is None:
        raise ValueError("The Hamiltonian has no non-identity terms to optimize.")
    return hamiltonian_mpo


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
    circuit, which is what makes the MPO cheaper here than in plain quimb DMRG:
    the Clifford part is absorbed into the observable rather than represented.
    The HSMPO's cached MPS is the starting guess, and is replaced by the optimized
    state on return.

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
        and ``energy_history``.
    """
    import quimb.tensor as qtn

    _, terms = pauli_terms(observables, hsmpo.nqubits)

    if bond_dims is None:
        bond_dims = [10, 20, 50, 100, 200]
    elif isinstance(bond_dims, int):
        bond_dims = [bond_dims]

    backpropagated = {}
    for pauli, coefficient in terms.items():
        if set(pauli) == {"I"}:
            backpropagated[pauli] = coefficient
            continue
        evolved, sign = hsmpo.stab_engine.backpropagate(
            observable=pauli, clifford_circuit=hsmpo.clifford_circuit
        )
        backpropagated[evolved] = coefficient * sign

    dmrg = qtn.DMRG2(
        build_pauli_mpo(hsmpo, backpropagated),
        bond_dims=bond_dims,
        cutoffs=cutoff,
        p0=copy.deepcopy(hsmpo.original_circuit_mps),
    )
    converged = dmrg.solve(tol=tol, max_sweeps=max_sweeps, verbosity=verbosity)

    energy_history = [float(np.real(e)) for e in getattr(dmrg, "energies", None) or []]
    hsmpo.original_circuit_mps = dmrg.state

    return {
        "ground_state": dmrg.state,
        "energy": float(np.real(dmrg.energy)),
        "converged": converged,
        "num_sweeps": len(energy_history) if energy_history else max_sweeps,
        "energy_history": energy_history,
    }
