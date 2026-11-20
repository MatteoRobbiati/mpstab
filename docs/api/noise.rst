Noise
=====

Stochastic Pauli noise, including correlated multi-qubit terms, coherent
Pauli-rotation errors, and general channels such as amplitude damping and
thermal relaxation on the HSMPO surrogate; after gates, on the spectators of a
gate (crosstalk), on idle qubits, at preparation and at readout.

Describe the noise with a :class:`~mpstab.noise.NoiseModel`, adding errors
rule by rule as in qibo, then call :meth:`~mpstab.HSMPO.noisy_expectation`:

- :meth:`~mpstab.noise.NoiseModel.add` puts an error after every matching gate,
  or on the initial state and the measured bits for
  :class:`~mpstab.noise.StatePreparationError` and
  :class:`~mpstab.noise.ReadoutError`;
- :meth:`~mpstab.noise.NoiseModel.add_crosstalk` puts it on the qubits coupled
  to a gate but not acted on by it;
- :meth:`~mpstab.noise.NoiseModel.add_idle` puts it on the qubits left idle in
  each moment, or on coupled pairs of them, as an always-on ``ZZ`` does.

:meth:`~mpstab.noise.NoiseModel.apply` returns the same noisy circuit as a
qibo density-matrix circuit, for exact reference values on small systems.

A Pauli fault pulled into the Clifford frame leaves the tableau unchanged: it
only negates the angle of every later dressed rotation it anticommutes with,
and the sign of the dressed observable if it anticommutes with that too. A noisy
expectation value is then an average of noiseless HSMPO expectation values over
sign patterns of the rotation chain: the patterns with zero and one fault are
summed exactly, those with two or more are sampled. Coherent errors are dressed
like magic gates and join the chain.

A general channel, written in the Pauli basis as
``sum_{A, B} chi[A, B] A rho B``, splits into its diagonal, a Pauli channel
handled as above, and its off-diagonal coherences, which make it non-unital.
A coherence acts on the ket with ``A`` and on the bra with ``B``: it flips
the two chains' signs differently and contributes the transition amplitude
between them, weighted by the complex ``chi[A, B]``. The one-location terms
stay exact; with several locations the sampled weights become a
quasi-probability, whose total exceeds 1 by about the summed ``|chi|`` of the
coherences.

Readout noise flips a measured ``0`` with probability ``p0`` and a ``1`` with
probability ``p1``: every non-identity factor of a Pauli term on that qubit
becomes ``(1 - p0 - p1) P + (p1 - p0) I``.

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
        StatePreparationError,
        ThermalRelaxationError,
    )

    ansatz = KickedIsing(nqubits=4, n_steps=1, theta_h=0.7, theta_z=0.3)
    surrogate = HSMPO(ansatz)

    noise = NoiseModel()
    noise.add(PauliError({"X": 1e-3, "Y": 1e-3, "Z": 1e-3}))  # every gate, every qubit
    noise.add(DepolarizingError(0.01), gates.CNOT)           # two-qubit depolarizing
    noise.add(PauliError({"ZZ": 0.005}), gates.CNOT)         # correlated dephasing
    noise.add(CoherentError("ZZ", 0.02), gates.CNOT)         # residual coupling
    noise.add(ThermalRelaxationError(t1=100.0, t2=80.0, time=0.05), gates.CNOT)

    chain = [(0, 1), (1, 2), (2, 3)]
    noise.add_crosstalk(CoherentError("ZZ", 0.01), chain, gates.CNOT)
    noise.add_idle(ThermalRelaxationError(t1=100.0, t2=80.0, time=0.02))
    noise.add_idle(CoherentError("ZZ", 0.005).twirled(), coupling=chain)

    noise.add(StatePreparationError(0.005))                  # |0> prepared as |1>
    noise.add(ReadoutError(0.01, 0.03))                      # 1 -> 0 likelier

    result = surrogate.noisy_expectation("ZIIZ", noise, n_samples=500, seed=0)
    print(f"noiseless {surrogate.expectation('ZIIZ'):+.4f}")
    print(f"noisy     {result.mean:+.4f} +- {result.stderr:.4f}")

    # Noise-strength sweeps, e.g. for zero-noise extrapolation:
    stronger = noise.scaled(2.0)

.. automodule:: mpstab.noise.model
   :members: NoiseModel, PauliError, DepolarizingError, CoherentError, KrausError,
      AmplitudeDampingError, ThermalRelaxationError, StatePreparationError,
      ReadoutError

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
