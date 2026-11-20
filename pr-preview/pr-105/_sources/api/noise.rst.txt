Noise
=====

Stochastic Pauli noise, including correlated multi-qubit terms, and coherent
Pauli-rotation errors on the HSMPO surrogate.

Describe the noise with a :class:`~mpstab.noise.NoiseModel`, adding errors
rule by rule as in qibo, then call :meth:`~mpstab.HSMPO.noisy_expectation`.
:meth:`~mpstab.noise.NoiseModel.apply` returns the same noisy circuit as a
qibo density-matrix circuit, for exact reference values on small systems.

A Pauli fault pulled into the Clifford frame leaves the tableau unchanged: it
only negates the angle of every later dressed rotation it anticommutes with,
and the sign of the dressed observable if it anticommutes with that too. A noisy
expectation value is then an average of noiseless HSMPO expectation values over
sign patterns of the rotation chain: the patterns with zero and one fault are
summed exactly, those with two or more are sampled. Coherent errors are dressed
like magic gates and join the chain.

Readout noise is a symmetric bit flip of each measured bit: it multiplies every
non-identity factor of a Pauli term on that qubit by ``1 - 2 p``.

General Kraus channels, such as amplitude damping, are not supported.

**Usage Example**::

    from qibo import gates
    from mpstab import HSMPO
    from mpstab.models.ansatze import KickedIsing
    from mpstab.noise import (
        CoherentError,
        DepolarizingError,
        NoiseModel,
        PauliError,
        ReadoutError,
    )

    ansatz = KickedIsing(nqubits=4, n_steps=1, theta_h=0.7, theta_z=0.3)
    surrogate = HSMPO(ansatz)

    noise = NoiseModel()
    noise.add(PauliError({"X": 1e-3, "Y": 1e-3, "Z": 1e-3}))  # every gate, every qubit
    noise.add(DepolarizingError(0.01), gates.CNOT)           # two-qubit depolarizing
    noise.add(PauliError({"ZZ": 0.005}), gates.CNOT)         # correlated dephasing
    noise.add(CoherentError("ZZ", 0.02), gates.CNOT)         # crosstalk
    noise.add(ReadoutError(0.02))                            # readout bit flips

    result = surrogate.noisy_expectation("ZIIZ", noise, n_samples=500, seed=0)
    print(f"noiseless {surrogate.expectation('ZIIZ'):+.4f}")
    print(f"noisy     {result.mean:+.4f} +- {result.stderr:.4f}")

    # Noise-strength sweeps, e.g. for zero-noise extrapolation:
    stronger = noise.scaled(2.0)

.. automodule:: mpstab.noise.model
   :members: NoiseModel, PauliError, DepolarizingError, CoherentError, ReadoutError

.. autoclass:: mpstab.noise.NoisyExpectation

Low-level interface
-------------------

:meth:`~mpstab.HSMPO.noisy_expectation` resolves the model into noise events
placed by gate index, compiles them into the Clifford frame and runs the
fault-pattern estimator. The three steps are available on their own.

.. automodule:: mpstab.noise.events
   :members:
   :show-inheritance:

.. automodule:: mpstab.noise.compile
   :members:

.. automodule:: mpstab.noise.mc_noise_estimator
   :members: noisy_expectation

See ``examples/scripts/noise/noisy_simulation.py`` for a comparison against exact
density-matrix simulation.
