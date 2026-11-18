Error Mitigation
================

Clifford data regression (CDR) learns a map from noisy expectation values to
noiseless ones on training circuits close to the target, then applies it to the
target's noisy value. The training circuits are low-magic variants of the
target, in which each non-Clifford gate is replaced by a Clifford one with some
probability. HSMPO computes their noiseless values cheaply, even with the
leftover magic that keeps them close to the target.

- :func:`~mpstab.models.error_mitigation.cdr` fits ``exact = a * noisy + b``;
- :func:`~mpstab.models.error_mitigation.vncdr` (variable-noise CDR) fits the
  exact value as a linear combination of the noisy values at several noise
  levels.

Both return a :class:`~mpstab.models.error_mitigation.MitigationNoiseMap`, which holds
the fitted parameters and the training data, and is called on the target's
noisy value(s) to mitigate them.

The noisy values come from ``noise``: a :class:`~mpstab.models.noise.NoiseModel`,
simulated with :meth:`~mpstab.HSMPO.noisy_expectation`, or any callable
returning the noisy value of a circuit, such as one running it on hardware.
For vnCDR a noise model is amplified either by scaling every error's strength
(``amplification="scale"``) or by global unitary folding
(``amplification="fold"``, :func:`~mpstab.models.error_mitigation.fold_circuit`).

Example
-------

::

    import numpy as np
    from qibo import gates

    from mpstab import HSMPO
    from mpstab.models.ansatze import HardwareEfficient
    from mpstab.models.error_mitigation import cdr, noisy_value, vncdr
    from mpstab.models.noise import DepolarizingError, NoiseModel, PauliError

    ansatz = HardwareEfficient(nqubits=4, nlayers=2)
    ansatz.circuit.set_parameters(
        np.random.default_rng(1).uniform(-np.pi, np.pi, ansatz.nparams)
    )
    noise = (
        NoiseModel()
        .add(DepolarizingError(0.01))
        .add(PauliError({"ZZ": 0.01}), gates.CZ)
    )

    target = HSMPO(ansatz)
    noisy = noisy_value(target, "ZZII", noise, seed=0)

    noise_map = cdr(ansatz, "ZZII", noise, n_training_circuits=30,
                    replacement_probability=0.7, seed=0)
    print(target.expectation("ZZII"), noisy, noise_map(noisy))

    levels = (1, 3, 5)
    noise_map = vncdr(ansatz, "ZZII", noise, noise_levels=levels,
                      n_training_circuits=30, replacement_probability=0.7, seed=0)
    print(noise_map([noisy_value(target, "ZZII", noise, l, seed=0) for l in levels]))

On hardware, pass the executor instead of the noise model, and mitigate the
value measured on the target circuit (``run_on_qpu`` stands for your own
execution code):

.. code-block:: python

    noise_map = cdr(ansatz, "ZZII", lambda circuit: run_on_qpu(circuit, "ZZII"))
    mitigated = noise_map(run_on_qpu(ansatz.circuit, "ZZII"))

API
---

.. automodule:: mpstab.models.error_mitigation.cdr
   :members: MitigationNoiseMap, cdr, vncdr, noisy_value

.. automodule:: mpstab.models.error_mitigation.training
   :members: training_circuits, fold_circuit
