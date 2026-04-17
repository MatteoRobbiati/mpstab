"""
Optimization utilities for HSMPO states.

Provides DMRG-based ground state optimization and Hamiltonian conversion functions.
Kept separate from HSMPO to maintain modularity.
"""

from typing import Union

import numpy as np
from qibo.hamiltonians import SymbolicHamiltonian


def hamiltonian_to_dict(
    hamiltonian: Union[str, list, dict, SymbolicHamiltonian],
    nqubits: int = None,
) -> dict:
    """
    Convert various Hamiltonian formats to dict of Pauli strings -> coefficients.

    Args:
        hamiltonian: Can be:
            - str: Single Pauli observable (e.g., "ZZZZ")
            - list of str: Multiple observables equal weight (e.g., ["ZZZZ", "XXXX"])
            - dict: Pauli string -> coefficient (e.g., {"ZZZZ": 1.0, "XXXX": 0.5})
            - SymbolicHamiltonian or Hamiltonian: Qibo Hamiltonian object
        nqubits: Number of qubits (inferred from hamiltonian if possible)

    Returns:
        dict: Normalized to {pauli_string: coefficient, ...}
    """
    if isinstance(hamiltonian, str):
        return {hamiltonian: 1.0}

    elif isinstance(hamiltonian, list):
        return {obs: 1.0 for obs in hamiltonian}

    elif isinstance(hamiltonian, dict):
        return hamiltonian

    elif isinstance(hamiltonian, SymbolicHamiltonian):
        # Extract terms from SymbolicHamiltonian
        coeffs, pauli_names, target_qubits = hamiltonian.simple_terms
        ham_dict = {}

        for coeff, p_name, targets in zip(coeffs, pauli_names, target_qubits):
            # Build full Pauli string with identities
            full_pauli_list = ["I"] * hamiltonian.nqubits
            for i, qubit_idx in enumerate(targets):
                full_pauli_list[qubit_idx] = p_name[i]
            full_pauli_string = "".join(full_pauli_list)

            # Add to dictionary
            ham_dict[full_pauli_string] = (
                coeff.real if hasattr(coeff, "real") else float(coeff)
            )

        # Add constant term if present
        if hamiltonian.constant != 0:
            ham_dict["I" * hamiltonian.nqubits] = hamiltonian.constant.real

        return ham_dict

    else:
        # Assume it's a SymbolicHamiltonian-like object with simple_terms
        # (matches the API already used in _expectation_from_symbolic_hamiltonian)
        coeffs, pauli_names, target_qubits = hamiltonian.simple_terms
        ham_dict = {}

        for coeff, p_name, targets in zip(coeffs, pauli_names, target_qubits):
            # Build full Pauli string with identities
            full_pauli_list = ["I"] * hamiltonian.nqubits
            for i, qubit_idx in enumerate(targets):
                full_pauli_list[qubit_idx] = p_name[i]
            full_pauli_string = "".join(full_pauli_list)

            # Add to dictionary
            ham_dict[full_pauli_string] = (
                coeff.real if hasattr(coeff, "real") else float(coeff)
            )

        # Add constant term if present
        if hasattr(hamiltonian, "constant") and hamiltonian.constant != 0:
            const_val = (
                hamiltonian.constant.real
                if hasattr(hamiltonian.constant, "real")
                else float(hamiltonian.constant)
            )
            ham_dict["I" * hamiltonian.nqubits] = const_val

        return ham_dict


def build_pauli_mpo(hsmpo, hamiltonian_dict: dict):
    """
    Build a Quimb MPO from Pauli string observable terms.

    Args:
        hsmpo: HSMPO instance (provides tn_engine)
        hamiltonian_dict: Dictionary mapping Pauli strings to coefficients
                         e.g., {"ZZ": 1.0, "XX": 0.5, "I": 0.2}

    Returns:
        mpo: Quimb MatrixProductOperator for the Hamiltonian
    """
    # Start with zero Hamiltonian
    H_mpo = None

    for pauli_string, coeff in hamiltonian_dict.items():
        # Skip identity terms for MPO (can be added as constant offset)
        if pauli_string == "I" * len(pauli_string) or pauli_string.upper() == "I":
            continue

        # Get MPO for this Pauli string (via TN engine)
        term_mpo = hsmpo.tn_engine.pauli_mpo(pauli_string)

        # Scale by coefficient
        if H_mpo is None:
            H_mpo = coeff * term_mpo
        else:
            H_mpo = H_mpo + coeff * term_mpo

    if H_mpo is None:
        raise ValueError("Hamiltonian has no non-identity terms")

    return H_mpo


def minimize_expectation_dmrg(
    hsmpo,
    observables: Union[str, list, dict, SymbolicHamiltonian],
    bond_dims: Union[int, list] = None,
    cutoff: float = 1e-9,
    tol: float = 1e-6,
    max_sweeps: int = 10,
    which: str = "SA",
    verbosity: int = 1,
):
    """
    Minimize expectation value(s) using DMRG (Density Matrix Renormalization Group).

    DMRG directly optimizes MPS tensors to minimize the given Hamiltonian,
    which is much more efficient than circuit-based VQE for ground state finding.

    Args:
        hsmpo: HSMPO instance to optimize
        observables: Hamiltonian to minimize. Can be:
            - str: Single observable (e.g., "ZZZZ")
            - list of str: Multiple observables sum equally (e.g., ["ZZZZ", "XXXX"])
            - dict: Observable -> coefficient (e.g., {"ZZZZ": 1.0, "XXXX": 0.5})
            - SymbolicHamiltonian: Qibo Hamiltonian object
        bond_dims: Max bond dimension(s). If int, constant. If list, grows per sweep.
                  Default: gradual growth [10, 20, 50, 100, 200]
        cutoff: SVD truncation cutoff (default: 1e-9)
        tol: Energy convergence tolerance (default: 1e-6)
        max_sweeps: Maximum number of full sweeps (default: 10)
        which: 'SA' for ground state (smallest eigenvalue), 'LA' for excited
        verbosity: Verbosity level 0-2

    Returns:
        dict with keys:
            - 'ground_state': Optimized MPS
            - 'energy': Minimum energy found
            - 'converged': Whether DMRG converged
            - 'num_sweeps': Number of sweeps performed
    """
    try:
        import quimb.tensor as qtn
    except ImportError:
        raise ImportError(
            "quimb not installed or tensor module not available. "
            "Install with: pip install quimb"
        )

    # Convert any format to dict
    obs_dict = hamiltonian_to_dict(observables, nqubits=hsmpo.nqubits)

    # Default bond dimension growth
    if bond_dims is None:
        bond_dims = [10, 20, 50, 100, 200]
    elif isinstance(bond_dims, int):
        bond_dims = [bond_dims]

    # Build MPO from observables (Hamiltonian)
    H_mpo = build_pauli_mpo(hsmpo, obs_dict)

    # Use precomputed MPS as initial guess
    import copy

    p0 = copy.deepcopy(hsmpo.original_circuit_mps)

    # Create DMRG solver
    dmrg = qtn.DMRG2(
        H_mpo,
        bond_dims=bond_dims,
        cutoffs=cutoff,
        p0=p0,
    )

    # Run DMRG optimization
    converged = dmrg.solve(
        tol=tol,
        max_sweeps=max_sweeps,
        verbosity=verbosity,
    )

    # Extract optimized state and energy
    optimized_mps = dmrg.state
    min_energy = dmrg.energy

    # Update the original_circuit_mps with the optimized state
    hsmpo.original_circuit_mps = optimized_mps

    return {
        "ground_state": optimized_mps,
        "energy": float(
            np.real(min_energy)
        ),  # Extract real part (complex may have tiny imag)
        "converged": converged,
        "num_sweeps": len(dmrg.energies) if hasattr(dmrg, "energies") else max_sweeps,
    }
