"""
State metrics: the stabilizer Renyi entropy, a measure of how much magic a state
carries, and the Meyer-Wallach global entanglement.

Only the HSMPO estimators live here. For a dense statevector, Qibo already
provides both metrics, :func:`qibo.quantum_info.stabilizer_renyi_entropy` and
:func:`qibo.quantum_info.meyer_wallach_entanglement`, and the estimators follow
the same conventions so the two can be compared directly.
"""

from typing import TYPE_CHECKING

import numpy as np

from mpstab.pauli import PAULI_LABELS

if TYPE_CHECKING:
    from mpstab.evolutors.hsmpo import HSMPO


def stabilizer_renyi_entropy_mps(
    hsmpo: "HSMPO",
    alpha: int = 2,
    n_samples: int = 1000,
    seed: int | None = None,
) -> float:
    r"""
    Estimate the stabilizer Rényi entropy of the state an HSMPO represents.

    Samples ``n_samples`` Pauli strings :math:`P_k` uniformly from the
    :math:`4^n` of them and replaces the exact Pauli sum with its Monte Carlo
    estimate,

    .. math::
        M_\alpha \approx \frac{1}{1 - \alpha}
        \log_2\left(\frac{4^n}{K} \sum_{k=1}^{K}
        \frac{|\langle P_k\rangle|^{2\alpha}}{2^n}\right),

    with :math:`K` the number of samples. The estimate of the sum is unbiased;
    its logarithm is not, and the bias shrinks as :math:`K` grows. The result is
    the total :math:`M_\alpha` in bits, the convention of
    :func:`qibo.quantum_info.stabilizer_renyi_entropy` with ``base=2``, which
    gives the exact value for a dense statevector; divide by ``hsmpo.nqubits``
    for the entropy per qubit. Each :math:`\langle P_k\rangle` is backpropagated
    through the cached Clifford circuit and contracted against the cached MPS, so
    the MPS is never rebuilt.

    Args:
        hsmpo: the hybrid stabilizer MPO, with its MPS cache already evolved.
        alpha: Rényi order :math:`\alpha`, an integer of at least 2.
        n_samples: number of Pauli strings to sample. More samples lower the
            variance, but with uniform sampling most strings have a vanishing
            expectation value, so the relative variance grows exponentially
            with the number of qubits.
        seed: seed of the random generator, for reproducible samples.

    Returns:
        The estimated stabilizer Rényi entropy :math:`M_\alpha`, in bits.

    Raises:
        ValueError: if ``alpha`` is below 2, or if the estimated Pauli sum is
            not positive.
    """
    if alpha < 2:
        raise ValueError(f"alpha must be an integer ≥ 2, got {alpha}.")

    rng = np.random.default_rng(seed)
    n = hsmpo.nqubits
    pauli_chars = np.array(list(PAULI_LABELS))

    # Each sample is an independent draw of n Pauli operators
    # Shape: (n_samples, n)
    sampled_indices = rng.integers(0, 4, size=(n_samples, n))

    expval_sum = 0.0
    for sample in sampled_indices:
        pauli_string = "".join(pauli_chars[sample])

        backprop_obs, sign = hsmpo.stab_engine.backpropagate(
            observable=pauli_string,
            clifford_circuit=hsmpo.clifford_circuit,
        )
        mpo = hsmpo.tn_engine.pauli_mpo(backprop_obs)
        expval = (
            hsmpo.tn_engine.expval(
                state_circuit=hsmpo.original_circuit_mps,
                operator=mpo,
            )
            * sign
        )

        expval_sum += abs(expval) ** (2 * alpha)

    # Unbiased estimator: rescale by 4^n / K to recover the full sum,
    # then normalise by 2^n as in the definition
    estimated_full_sum = (4**n / n_samples) * expval_sum / 2**n

    if estimated_full_sum <= 0:
        raise ValueError(
            f"Non-positive estimated Pauli sum ({estimated_full_sum}); "
            "try increasing n_samples or check for numerical issues."
        )

    return float(np.log2(estimated_full_sum) / (1.0 - alpha))


def meyer_wallach_mps(hsmpo: "HSMPO") -> float:
    r"""
    Compute the Meyer-Wallach global entanglement of the state an HSMPO represents.

    Evaluates the average single-qubit linear entropy

    .. math::
        Q(\psi) = 2 \left(1 - \frac{1}{n} \sum_{k=0}^{n-1}
        \mathrm{Tr}\,\rho_k^2\right),

    with :math:`\rho_k` the reduced state of qubit :math:`k`. The value is 0 for
    product states and 1 when every qubit is maximally entangled with the rest.
    For a dense statevector, use :func:`qibo.quantum_info.meyer_wallach_entanglement`
    instead.

    The HSMPO state is :math:`C|\psi_{\mathrm{MPS}}\rangle`, with :math:`C` the
    cached Clifford circuit, so the single-qubit reduced states cannot be read off
    the MPS sites. The function rebuilds each purity from the qubit's Bloch vector,

    .. math::
        \mathrm{Tr}\,\rho_k^2 = \frac{1}{2} \left(1 + \langle X_k\rangle^2
        + \langle Y_k\rangle^2 + \langle Z_k\rangle^2\right),

    which turns :math:`Q` into

    .. math::
        Q = 1 - \frac{1}{n} \sum_{k=0}^{n-1} \sum_{P \in \{X, Y, Z\}}
        \langle P_k\rangle^2 .

    It backpropagates every :math:`\langle P_k\rangle` through :math:`C` and
    contracts it against the cached MPS. That costs :math:`3n` contractions, and
    the result is exact up to the MPS truncation.

    Args:
        hsmpo: the hybrid stabilizer MPO, with its MPS cache already evolved.

    Returns:
        The Meyer-Wallach :math:`Q` of the circuit's output state.
    """
    n = hsmpo.nqubits
    local_paulis = [
        "I" * k + pauli + "I" * (n - k - 1) for k in range(n) for pauli in "XYZ"
    ]
    bloch_sq_sum = sum(hsmpo.expectation(p) ** 2 for p in local_paulis)
    return float(1 - bloch_sq_sum / n)
