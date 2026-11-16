Noise
=====

Stochastic Pauli noise, including correlated multi-qubit terms, and coherent
Pauli-rotation errors on the HSMPO surrogate.

A Pauli fault pulled into the Clifford frame leaves the tableau unchanged: it
only negates the angle of every later dressed rotation it anticommutes with,
and the sign of the dressed observable if it anticommutes with that too. A noisy
expectation value is then an average of noiseless HSMPO expectation values over
sign patterns of the rotation chain. :func:`~mpstab.noise.noisy_expectation`
sums the patterns with zero and one fault exactly and samples those with two or
more. Coherent errors are dressed like magic gates and join the chain.

General Kraus channels, such as amplitude damping, are not supported.

**Usage Example**::

    import numpy as np
    from qibo import gates
    from qibo.noise import DepolarizingError, NoiseModel
    from mpstab import HSMPO
    from mpstab.models.ansatze import KickedIsing
    from mpstab.noise import (
        CoherentRotation,
        PauliChannel,
        compile_noise,
        from_qibo_noise_model,
        noisy_expectation,
    )

    ansatz = KickedIsing(nqubits=4, n_steps=1, theta_h=0.7, theta_z=0.3)
    surrogate = HSMPO(ansatz)

    # Depolarizing noise after every CNOT, from a qibo noise model...
    model = NoiseModel()
    model.add(DepolarizingError(0.01), gates.CNOT)
    events = from_qibo_noise_model(model, ansatz.circuit)

    # ...plus correlated ZZ dephasing and ZZ crosstalk after the first one.
    first_cnot = next(i for i, g in enumerate(ansatz.circuit.queue) if g.name == "cx")
    events.append(PauliChannel(first_cnot, (0, 1), {"ZZ": 0.005}))
    events.append(CoherentRotation(first_cnot, "ZZII", 0.02))

    compiled = compile_noise(surrogate, events)
    result = noisy_expectation(compiled, "ZIIZ", n_samples=500, seed=0)
    print(f"noiseless {surrogate.expectation('ZIIZ'):+.4f}")
    print(f"noisy     {result.mean:+.4f} +- {result.stderr:.4f}")
    print(f"{result.n_patterns} sign patterns, Pr(k>=2) = {result.stratum_weights[2]:.4f}")

.. automodule:: mpstab.noise.events
   :members:
   :show-inheritance:

.. automodule:: mpstab.noise.compile
   :members:

.. automodule:: mpstab.noise.estimator
   :members:

See ``examples/noisy_simulation.py`` for a comparison against exact
density-matrix simulation.
