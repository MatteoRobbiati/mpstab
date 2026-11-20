"""DMRG ground-state optimization of an HSMPO's MPS."""

from typing import Union

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
