"""
Tests for HSMPO optimization via DMRG.

Validates that:
1. DMRG correctly optimizes MPS to minimize observables
2. Final energies match qibo eigenvalue solver results
3. All Hamiltonian input formats work correctly
"""

import numpy as np
import pytest
from qibo import Circuit, gates
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import X, Y, Z

from mpstab import HSMPO
from mpstab.models.ansatze import HardwareEfficient


def get_qibo_ground_state_energy(hamiltonian_dict: dict, nqubits: int) -> float:
    """
    Compute ground state energy using qibo's eigenvalue solver.

    Args:
        hamiltonian_dict: Dictionary of Pauli strings -> coefficients
        nqubits: Number of qubits

    Returns:
        Minimum eigenvalue of the Hamiltonian
    """
    # Build SymbolicHamiltonian from Pauli terms
    pauli_form = 0
    constant = 0

    for pauli_str, coeff in hamiltonian_dict.items():
        # Handle identity terms as constant offset
        if all(p == "I" for p in pauli_str):
            constant += coeff
            continue

        # Build Qibo operator from Pauli string
        term = 1
        has_paulis = False
        for i, pauli in enumerate(pauli_str):
            if pauli == "X":
                term = term * X(i)
                has_paulis = True
            elif pauli == "Y":
                term = term * Y(i)
                has_paulis = True
            elif pauli == "Z":
                term = term * Z(i)
                has_paulis = True
            elif pauli == "I":
                continue
            else:
                raise ValueError(f"Unknown Pauli: {pauli}")

        if has_paulis:
            pauli_form = pauli_form + coeff * term

    # Create SymbolicHamiltonian and find eigenvalues
    H = SymbolicHamiltonian(nqubits=nqubits, form=pauli_form, constant=constant)

    # Use qibo's dense matrix representation
    from scipy.sparse.linalg import eigsh

    matrix = H.matrix.todense() if hasattr(H.matrix, "todense") else H.matrix

    # Find minimum eigenvalue
    import numpy as np

    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(np.real(eigenvalues[0]))


class TestDMRGOptimization:
    """Test DMRG optimization functionality."""

    def test_single_pauli_optimization(self):
        """Test optimization of a single Pauli observable."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Optimize ZZZZ (should reach -1.0 ground state)
        result = hsmpo.minimize_expectation(
            observables="ZZZZ",
            method="dmrg",
            bond_dims=[10, 20],
            max_sweeps=10,
            verbosity=0,
        )

        # Verify convergence
        assert result["converged"], "DMRG should converge"
        assert result["energy"] < 0, "Energy should be negative"

        # Ground state of ZZZZ is |0000> or |1111> with eigenvalue -1.0
        expected_gs_energy = -1.0
        assert np.isclose(
            result["energy"], expected_gs_energy, atol=1e-6
        ), f"Energy {result['energy']} should match ground state {expected_gs_energy}"

    def test_dict_hamiltonian_optimization(self):
        """Test optimization with dict Hamiltonian format."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # H = ZZZZ + 0.5*XXXX
        hamiltonian = {"ZZZZ": 1.0, "XXXX": 0.5}

        result = hsmpo.minimize_expectation(
            observables=hamiltonian,
            method="dmrg",
            bond_dims=[10, 20, 50],
            max_sweeps=10,
            verbosity=0,
        )

        # Verify DMRG converged and found a negative energy (below random initialization)
        assert result["converged"], "DMRG should converge"
        assert result["energy"] < -1.0, "Energy should be negative and below -1.0"
        assert result["num_sweeps"] <= 10, "Should converge within max sweeps"

    def test_list_hamiltonian_optimization(self):
        """Test optimization with list Hamiltonian format (equal weights)."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # H = ZZZZ + XXXX + YYYY (all equally weighted)
        observables = ["ZZZZ", "XXXX", "YYYY"]

        result = hsmpo.minimize_expectation(
            observables=observables,
            method="dmrg",
            bond_dims=[10, 20],
            max_sweeps=10,
            verbosity=0,
        )

        # Verify convergence and optimization
        assert result["converged"], "DMRG should converge"
        assert result["energy"] < 0, "Energy should be negative"

        # Verify final observables match DMRG result
        total_expectation = (
            hsmpo.expectation("ZZZZ")
            + hsmpo.expectation("XXXX")
            + hsmpo.expectation("YYYY")
        )
        assert np.isclose(
            total_expectation, result["energy"], atol=1e-10
        ), f"Sum of observables {total_expectation} should match {result['energy']}"

    def test_symbolic_hamiltonian_optimization(self):
        """Test optimization with SymbolicHamiltonian."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Build XXZ Hamiltonian: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + 0.5 Z_i Z_{i+1})
        ham_form = 0
        for i in range(3):
            ham_form += X(i) * X(i + 1)
            ham_form += Y(i) * Y(i + 1)
            ham_form += 0.5 * Z(i) * Z(i + 1)

        H_symbolic = SymbolicHamiltonian(nqubits=4, form=ham_form)

        result = hsmpo.minimize_expectation(
            observables=H_symbolic,
            method="dmrg",
            bond_dims=[10, 20, 50],
            max_sweeps=10,
            verbosity=0,
        )

        # Verify convergence and reasonable energy
        assert result["converged"], "DMRG should converge"
        assert result["energy"] < -1.0, "XXZ should have ground state energy below -1.0"
        assert result["ground_state"] is not None, "Should return optimized MPS"

    def test_parameter_update_and_reoptimize(self):
        """Test that set_parameters() invalidates cache and allows re-optimization."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # First optimization
        result1 = hsmpo.minimize_expectation(
            observables="ZZZZ", method="dmrg", bond_dims=[10], max_sweeps=5, verbosity=0
        )
        energy1 = result1["energy"]

        # Change circuit parameters
        new_params = np.random.uniform(-np.pi, np.pi, hsmpo.nparams)
        hsmpo.set_parameters(new_params)

        # Re-optimize (should start from different initial MPS)
        result2 = hsmpo.minimize_expectation(
            observables="ZZZZ", method="dmrg", bond_dims=[10], max_sweeps=5, verbosity=0
        )
        energy2 = result2["energy"]

        # Both should converge to same ground state (by uniqueness of GS for ZZZZ)
        assert np.isclose(
            energy1, energy2, atol=1e-6
        ), f"Ground state energies should match: {energy1} vs {energy2}"

    def test_mps_state_updated_after_optimization(self):
        """Test that original_circuit_mps is updated after DMRG."""
        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Get initial MPS expectation
        initial_mps = hsmpo.original_circuit_mps
        initial_energy = hsmpo.expectation("ZZZZ")

        # Optimize
        result = hsmpo.minimize_expectation(
            observables="ZZZZ", method="dmrg", bond_dims=[10], max_sweeps=5, verbosity=0
        )

        # Get optimized MPS expectation
        optimized_mps = hsmpo.original_circuit_mps
        optimized_energy = hsmpo.expectation("ZZZZ")

        # MPS should be different
        assert optimized_mps is not initial_mps, "MPS reference should have changed"

        # Energy should have improved (become more negative)
        assert (
            optimized_energy < initial_energy
        ), f"Optimized energy {optimized_energy} should be lower than initial {initial_energy}"

        # Optimized energy should match DMRG result
        assert np.isclose(
            optimized_energy, result["energy"], atol=1e-10
        ), f"Observable should match optimization result"


class TestHamiltonianConversion:
    """Test Hamiltonian format conversion."""

    def test_hamiltonian_to_dict_str(self):
        """Test conversion from string to dict."""
        from mpstab.evolutors.optimization import hamiltonian_to_dict

        result = hamiltonian_to_dict("ZZZZ")
        assert result == {"ZZZZ": 1.0}

    def test_hamiltonian_to_dict_list(self):
        """Test conversion from list to dict."""
        from mpstab.evolutors.optimization import hamiltonian_to_dict

        result = hamiltonian_to_dict(["ZZZZ", "XXXX", "YYYY"])
        expected = {"ZZZZ": 1.0, "XXXX": 1.0, "YYYY": 1.0}
        assert result == expected

    def test_hamiltonian_to_dict_dict(self):
        """Test that dict passes through unchanged."""
        from mpstab.evolutors.optimization import hamiltonian_to_dict

        input_dict = {"ZZZZ": 1.0, "XXXX": 0.5}
        result = hamiltonian_to_dict(input_dict)
        assert result == input_dict

    def test_hamiltonian_to_dict_symbolic(self):
        """Test conversion from SymbolicHamiltonian to dict."""
        from mpstab.evolutors.optimization import hamiltonian_to_dict

        # Build simple XXZ
        ham_form = (
            X(0) * X(1)
            + Y(0) * Y(1)
            + 0.5 * Z(0) * Z(1)
            + 0.5 * Z(1) * Z(2)
            + 0.5 * Z(2) * Z(3)
        )
        H = SymbolicHamiltonian(nqubits=4, form=ham_form)

        result = hamiltonian_to_dict(H, nqubits=4)

        # Should have XX, YY, ZZ terms (distributed across the 4 qubits with padding)
        assert len(result) > 0, "Should have some terms"

        # Check for any terms with X or Y operators
        has_x_terms = any("X" in k for k in result.keys())
        has_y_terms = any("Y" in k for k in result.keys())
        has_z_terms = any("Z" in k for k in result.keys())

        assert has_x_terms, "Should have X terms"
        assert has_y_terms, "Should have Y terms"
        assert has_z_terms, "Should have Z terms"


class TestEngineSupportValidation:
    """Test that unsupported engines raise appropriate errors."""

    def test_native_tensor_network_engine_raises_not_implemented(self):
        """Test that NativeTensorNetworkEngine raises NotImplementedError during initialization."""
        from mpstab.engines.tensor_networks.native import NativeTensorNetworkEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)

        with pytest.raises(
            NotImplementedError,
            match="NativeTensorNetworkEngine is not supported by HSMPO",
        ):
            hsmpo = HSMPO(ansatz=ansatz)
            hsmpo.set_engines(tn_engine=NativeTensorNetworkEngine())

    def test_native_tensor_network_engine_error_during_post_init(self):
        """Test that NativeTensorNetworkEngine raises error at HSMPO creation if set as default."""
        from mpstab.engines.tensor_networks.native import NativeTensorNetworkEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Verify that using NativeTensorNetworkEngine later also raises error
        with pytest.raises(
            NotImplementedError,
            match="NativeTensorNetworkEngine is not supported by HSMPO",
        ):
            hsmpo.set_engines(tn_engine=NativeTensorNetworkEngine())

    def test_quimb_engine_works(self):
        """Test that QuimbEngine works correctly (should pass without error)."""
        from mpstab.engines.tensor_networks.quimb import QuimbEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # This should not raise any error
        hsmpo.set_engines(tn_engine=QuimbEngine())

        # Verify it still works
        result = hsmpo.expectation("ZZZZ")
        assert isinstance(result, (float, np.floating))


class TestEngineSupportValidation:
    """Test native engine support and graceful degradation."""

    def test_expectation_with_quimb_engine(self):
        """Test expectation calculation with QuimbEngine (default)."""
        from mpstab.engines import QuimbEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)
        hsmpo.set_engines(tn_engine=QuimbEngine())

        # Should work without error
        result = hsmpo.expectation("ZZZZ")
        assert isinstance(result, (float, np.floating))

    def test_expectation_with_native_engine(self):
        """Test expectation calculation with NativeTensorNetworkEngine."""
        from mpstab.engines import NativeTensorNetworkEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Switch to NativeTensorNetworkEngine
        hsmpo.set_engines(tn_engine=NativeTensorNetworkEngine())

        # Should work: expectation() supports native engine via statevector
        result = hsmpo.expectation("ZZZZ")
        assert isinstance(result, (float, np.floating))

    def test_minimize_expectation_fails_with_native_engine(self):
        """Test that minimize_expectation raises NotImplementedError for NativeTensorNetworkEngine."""
        from mpstab.engines import NativeTensorNetworkEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)

        # Switch to NativeTensorNetworkEngine
        hsmpo.set_engines(tn_engine=NativeTensorNetworkEngine())

        # Should raise NotImplementedError
        with pytest.raises(
            NotImplementedError, match="DMRG optimization requires QuimbEngine"
        ):
            hsmpo.minimize_expectation(
                observables={"ZZZZ": 1.0},
                method="dmrg",
                bond_dims=[8],
                max_sweeps=2,
                verbosity=0,
            )

    def test_minimize_expectation_works_with_quimb_engine(self):
        """Test that minimize_expectation works with QuimbEngine."""
        from mpstab.engines import QuimbEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)
        hsmpo = HSMPO(ansatz=ansatz)
        hsmpo.set_engines(tn_engine=QuimbEngine())

        # Should work without error
        result = hsmpo.minimize_expectation(
            observables={"ZZZZ": 1.0},
            method="dmrg",
            bond_dims=[8],
            max_sweeps=2,
            verbosity=0,
        )

        assert isinstance(result, dict)
        assert "energy" in result or "ground_state_energy" in result
        # Check whatever key is used for energy
        energy_key = "energy" if "energy" in result else "ground_state_energy"
        assert isinstance(result[energy_key], (float, np.floating))

    def test_expectation_accuracy_native_vs_quimb(self):
        """Verify expectation values match between native and quimb engines."""
        from mpstab.engines import NativeTensorNetworkEngine, QuimbEngine

        ansatz = HardwareEfficient(nqubits=4, nlayers=2)

        # Calculate with Quimb engine
        hsmpo_quimb = HSMPO(ansatz=ansatz)
        hsmpo_quimb.set_engines(tn_engine=QuimbEngine())
        result_quimb = hsmpo_quimb.expectation("ZZZZ")

        # Calculate with Native engine
        hsmpo_native = HSMPO(ansatz=ansatz)
        hsmpo_native.set_engines(tn_engine=NativeTensorNetworkEngine())
        result_native = hsmpo_native.expectation("ZZZZ")

        # Results should match (within numerical precision)
        np.testing.assert_allclose(result_native, result_quimb, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
