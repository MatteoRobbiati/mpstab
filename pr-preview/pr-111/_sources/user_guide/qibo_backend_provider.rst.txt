MPSTAB as a Qibo Backend Provider
==================================

``mpstab`` can be used as a custom backend provider for `Qibo <https://qibo.science/>`_, an open-source (and full-stack!) quantum computing framework.
This allows you to use MPSTAB's efficient hybrid stabilizer-MPO simulation as a backend for any Qibo circuits and Hamiltonians.

.. note::
    Currently, ``mpstab`` can be used to compute expectation values, and not for pure circuit execution. In the end, the main scope of this library
    is to provide a tool for expectation value calculation. We plan to implement sampling strategies in the future, and new features will be available when this will be done.

What is a Backend Provider?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Qibo, a backend is responsible for executing quantum circuits and computing observables (Hamiltonians). By registering ``mpstab`` as a backend, you can seamlessly integrate hybrid stabilizer-MPO simulation into Qibo workflows without modifying your Qibo code. You simply specify ``"mpstab"`` as the backend when constructing circuits or Hamiltonians.


Computing Hamiltonian Expectation Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A powerful use case is computing expectation values of Hamiltonians using MPSTAB. Here's an example with a Transfer Field Ising Model::

    from qibo import Circuit, gates, construct_backend
    from qibo.hamiltonians import TFIM

    # Create the MPSTAB backend
    mpstab_backend = construct_backend("mpstab")

    # Create a test circuit
    circuit = Circuit(8)
    circuit.add(gates.H(q) for q in range(8))
    for q in range(8):
        circuit.add(gates.RY(q, theta=0.3))
    circuit.add(gates.CNOT(0, 1))

    # Create a Hamiltonian with MPSTAB backend
    hamiltonian = TFIM(nqubits=8, h=0.5, backend=mpstab_backend, dense=False)

    # Compute expectation value
    expectation = hamiltonian.expectation(circuit)
    print(f"<TFIM> = {expectation}")

Benefits of Using MPSTAB as a Backend
--------------------------------------

- **Efficient Clifford Simulation**: Pure Clifford operations are handled with :math:`\mathcal{O}(n^2)` complexity using the Gottesman-Knill theorem. Note that Qibo currently supports a dedicated Clifford backend. If you plan to execute pure Clifford circuits with Qibo, we recommend using the dedicated backend.
- **Non-Clifford Support**: Unlike pure stabilizer backends, MPSTAB can handle arbitrary quantum gates through adaptive bond dimension truncation.
- **Seamless Integration**: Use MPSTAB wherever you would use any other Qibo backend—no code changes needed.

Comparing Results with Other Backends
--------------------------------------

You can easily compare MPSTAB results with other Qibo backends like the numpy (statevector) backend::

    from qibo import construct_backend, Circuit, gates
    from qibo.hamiltonians import TFIM

    # Create backends
    mpstab_backend = construct_backend("mpstab")
    numpy_backend = construct_backend("numpy")

    # Create circuit
    circuit = Circuit(5)
    circuit.add(gates.H(q) for q in range(5))
    circuit.add(gates.RY(q, theta=0.2) for q in range(5))

    # Compute with both backends
    ham = TFIM(nqubits=5, h=0.5, backend=mpstab_backend, dense=False)
    mpstab_result = ham.expectation(circuit)

    ham_numpy = TFIM(nqubits=5, h=0.5, backend=numpy_backend, dense=False)
    numpy_result = ham_numpy.expectation(circuit)

    print(f"MPSTAB result: {mpstab_result}")
    print(f"Numpy result: {numpy_result}")
    print(f"Agreement: {abs(mpstab_result - numpy_result) < 1e-6}")
