Fidelity and Approximation
==========================

Understanding fidelity bounds and approximation errors in MPSTAB.

Concept: Fidelity Lower Bound
-----------------------------

MPSTAB tracks approximation errors through the **fidelity lower bound**, which indicates how much quantum information is preserved when truncating the MPO. The fidelity is computed by calculating the error introduced during gate approximation in the tensor network representation, as described in `Phys. Rev. X 10, 041038 <https://journals.aps.org/prx/pdf/10.1103/PhysRevX.10.041038>`_.

What is Fidelity?
~~~~~~~~~~~~~~~~~

Fidelity is a measure of similarity between two quantum states:

- **F = 1**: States are identical
- **F = 0**: States are orthogonal
- **0 < F < 1**: Partial overlap

Why Truncate?
~~~~~~~~~~~~~

When the MPO bond dimension grows too large:

- **Memory usage** becomes prohibitive
- **Computation time** increases dramatically
- **Storage** of tensors becomes infeasible

By truncating singular values below a threshold, we reduce bond dimension but lose some fidelity.

Accessing Fidelity Information
-------------------------------

Basic Usage
~~~~~~~~~~~

::

    from mpstab import HSMPO
    from qibo import Circuit, gates

    # Create circuit
    circuit = Circuit(5)
    for q in range(5):
        circuit.add(gates.H(q))
        circuit.add(gates.RY(q, theta=0.5))

    # With truncation
    simulator = HSMPO(ansatz=circuit, max_bond_dimension=8)
    result = simulator.expectation("ZZZZZ")

    # Check fidelity lower bound
    fidelity = simulator.fidelity_lower_bound
    print(f"Fidelity: {fidelity:.6f}")
    print(f"Information loss: {(1 - fidelity)*100:.2f}%")

Fidelity vs Bond Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from mpstab import HSMPO

    # Study approximation quality
    results = []
    for max_bd in [2, 4, 8, 16, 32, 64, None]:
        simulator = HSMPO(ansatz=circuit, max_bond_dimension=max_bd)
        expval = simulator.expectation("ZZZZZ")
        fidelity = simulator.fidelity_lower_bound

        bd_str = f"χ={max_bd}" if max_bd else "Exact"
        print(f"{bd_str:<10} | Expval: {expval:8.4f} | Fidelity: {fidelity:.6f}")

Interpreting Fidelity Values
-----------------------------

=========  =======================  ==========================
Fidelity   Interpretation           Typical Usage
=========  =======================  ==========================
1.0        No information loss      Exact simulation
0.99-1.0   Negligible error         Production calculations
0.95-0.99  Small error              Most applications
0.90-0.95  Moderate error           Preliminary studies
0.80-0.90  Significant error        Quick prototyping
< 0.80     Large error              Use larger χ
=========  =======================  ==========================

Benchmarking: Hybrid vs Pure Tensor Network
---------------------------------------------

MPSTAB provides built-in properties to benchmark the hybrid HSMPO representation against a pure tensor network approach. This helps you understand the advantage of the hybrid method.

Two key properties are available:

**1. Pure Tensor Network Fidelity** (``truncation_fidelity_pure_tn``)
    * Fidelity if the circuit were represented as a pure Matrix Product State (MPS)
    * Shows the baseline accuracy of tensor network truncation alone
    * Useful for comparison

**2. Hybrid HSMPO Fidelity** (``truncation_fidelity()``)
    * Actual fidelity of the hybrid stabilizer-MPO representation
    * Accounts for both Clifford gate treatment and tensor network truncation
    * Expected to be equal or better than pure tensor network

Example usage::

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient

    ansatz = HardwareEfficient(nqubits=16, nlayers=8)
    simulator = HSMPO(ansatz=circuit, max_bond_dimension=8)

    # Get pure tensor network fidelity (baseline)
    pure_tn_fidelity = simulator.truncation_fidelity_pure_tn
    print(f"Pure TN fidelity: {pure_tn_fidelity:.6f}")

    # Get hybrid HSMPO fidelity
    hybrid_fidelity = simulator.truncation_fidelity()
    print(f"Hybrid fidelity: {hybrid_fidelity:.6f}")

    # The hybrid method should be at least as good as pure TN
    advantage = hybrid_fidelity - pure_tn_fidelity
    print(f"Hybrid advantage: {advantage:.6f}")

This comparison shows how efficiently MPSTAB leverages Clifford gates to improve simulation fidelity compared to treating all gates uniformly.

Practical Guidelines
--------------------

For Accurate Simulations
~~~~~~~~~~~~~~~~~~~~~~~~

::

    # Aim for fidelity > 0.99
    simulator = HSMPO(ansatz=circuit, max_bond_dimension=32)
    if simulator.fidelity_lower_bound > 0.99:
        result = simulator.expectation("ZZZZZ")
        print("Result is reliable")
    else:
        # Increase bond dimension
        simulator = HSMPO(ansatz=circuit, max_bond_dimension=64)

Adaptive Bond Dimension
~~~~~~~~~~~~~~~~~~~~~~~

Automatically find the minimum bond dimension needed::

    def find_min_bond_dimension(circuit, observable, min_fidelity=0.99):
        """Find minimum bond dimension for desired fidelity."""

        for max_bd in [2, 4, 8, 16, 32, 64, 128, 256]:
            simulator = HSMPO(ansatz=circuit, max_bond_dimension=max_bd)
            result = simulator.expectation(observable)
            fidelity = simulator.fidelity_lower_bound

            print(f"χ={max_bd}: fidelity={fidelity:.4f}")

            if fidelity >= min_fidelity:
                return max_bd, result, fidelity

        # If we get here, even χ=256 isn't enough
        return None, None, None

Sources of Fidelity Loss
------------------------

1. **Bond Dimension Truncation**
   - Most common source
   - Controlled by ``max_bond_dimension`` parameter
   - Larger dimension = better fidelity

2. **Non-Clifford Gate Effects**
   - Circuits with more non-Clifford gates lose fidelity faster
   - MPO must track more quantum information
   - Consider reducing circuit depth

3. **Entanglement Growth**
   - Highly entangled states require larger bond dimensions
   - Area-law entanglement: scales favorably
   - Volume-law entanglement: problematic for large systems

Further Reading
---------------

- :doc:`../api/evolutors` - HSMPO API documentation
- Research papers on tensor network truncation
