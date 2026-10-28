"""
State metrics: the stabilizer Renyi entropy, a measure of how much magic a state
carries, and the Meyer-Wallach global entanglement.
"""

from itertools import product
from typing import TYPE_CHECKING

import numpy as np
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian

from mpstab.pauli import PAULI_LABELS

if TYPE_CHECKING:
    from mpstab.evolutors.hsmpo import HSMPO


def generate_pauli_strings(n: int) -> list[str]:
    """Every ``n``-qubit Pauli string. There are ``4**n`` of them."""
    return ["".join(p) for p in product(PAULI_LABELS, repeat=n)]


def stabilizer_renyi_entropy(state: np.ndarray, alpha: int) -> float:
    """
    Compute the exact Stabilizer Rényi Entropy of order alpha for a dense state vector.
    Implementation follows Eq. (1) of https://arxiv.org/pdf/2207.13076.

    Cost is O(4^n) — only suitable for n ≲ 12.
    """
    nqubits = int(np.log2(len(state)))
    pauli_strings = generate_pauli_strings(nqubits)

    expval_sum = 0.0
    for pauli_string in pauli_strings:
        for i, pauli_op in enumerate(pauli_string):
            if i == 0:
                symbolic_obs = getattr(symbols, pauli_op)(i)
            else:
                symbolic_obs *= getattr(symbols, pauli_op)(i)

        obs = SymbolicHamiltonian(form=symbolic_obs)
        expval_sum += (obs.expectation_from_state(state) ** (2 * alpha)) / (2**nqubits)

    raw = float((1.0 / (1 - alpha)) * np.log(expval_sum))
    return raw / (nqubits * np.log(2))


def stabilizer_renyi_entropy_mps(
    hsmpo: "HSMPO",
    alpha: int = 2,
    n_samples: int = 1000,
    seed: int | None = None,
) -> float:
    """
    Stochastic estimate of the Stabilizer Rényi Entropy of order alpha,
    using importance-free uniform Pauli sampling over the MPS representation.

    Follows the estimator of Lami & Collura (PRL 2023):

        S_α ≈ (1/(1-α)) * log( (4^n / K) * Σ_{k=1}^{K} |⟨P_k⟩|^{2α} / 2^n )

    where P_k are sampled uniformly at random from the 4^n Pauli strings.
    This is an unbiased Monte Carlo estimate of the exact sum.

    Exploits the precomputed MPS cache in HSMPO (original_circuit_mps,
    clifford_circuit), so no MPS re-initialization is performed.

    Parameters
    ----------
    hsmpo : HSMPO
        The hybrid stabilizer MPO object with a precomputed MPS cache.
    alpha : int
        Rényi order, must be ≥ 2.
    n_samples : int
        Number of random Pauli strings to sample. Higher gives lower variance.
        As a rough guide, n_samples ~ 1000–5000 is sufficient for n ≲ 25.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        Stochastic estimate of SRE_α.
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

    raw = float(np.log(estimated_full_sum) / (1.0 - alpha))
    return raw / (n * np.log(2))


def meyer_wallach(state: np.ndarray) -> float:
    r"""
    Exact Meyer-Wallach global entanglement of a dense state vector,

    .. math::
        Q(\psi) = 2 \left(1 - \frac{1}{n} \sum_{k=0}^{n-1} \mathrm{Tr}\,\rho_k^2\right),

    with :math:`\rho_k` the reduced state of qubit :math:`k`. It is 0 for product
    states and 1 when every qubit is maximally entangled with the rest; a
    Haar-random state averages :math:`(2^n - 2) / (2^n + 1)`.
    See https://arxiv.org/abs/quant-ph/0108104.

    Cost is O(n 2^n), so only suitable for small n.
    """
    nqubits = int(np.log2(len(state)))
    tensor = np.asarray(state).reshape([2] * nqubits)

    purity_sum = 0.0
    for k in range(nqubits):
        m = np.moveaxis(tensor, k, 0).reshape(2, -1)
        rho = m @ m.conj().T
        purity_sum += np.real(np.trace(rho @ rho))

    return float(2 * (1 - purity_sum / nqubits))


def meyer_wallach_mps(hsmpo: "HSMPO") -> float:
    r"""
    Meyer-Wallach global entanglement of the state an HSMPO represents.

    The state is :math:`C|\psi_{\mathrm{MPS}}\rangle`, with :math:`C` the cached
    Clifford circuit, so the single-qubit reduced states cannot be read off the
    MPS sites. Each purity is rebuilt from the qubit's Bloch vector instead,

    .. math::
        \mathrm{Tr}\,\rho_k^2 = \frac{1}{2}
        \left(1 + \langle X_k\rangle^2 + \langle Y_k\rangle^2 + \langle Z_k\rangle^2\right),

    which turns :math:`Q` into

    .. math::
        Q = 1 - \frac{1}{n} \sum_{k=0}^{n-1} \sum_{P \in \{X, Y, Z\}} \langle P_k\rangle^2,

    where every :math:`\langle P_k\rangle` is backpropagated through :math:`C`
    and contracted against the cached MPS. That is :math:`3n` contractions, exact
    up to the MPS truncation.

    Parameters
    ----------
    hsmpo : HSMPO
        The hybrid stabilizer MPO object with a precomputed MPS cache.

    Returns
    -------
    float
        Meyer-Wallach Q of the circuit's output state.
    """
    n = hsmpo.nqubits
    local_paulis = [
        "I" * k + pauli + "I" * (n - k - 1) for k in range(n) for pauli in "XYZ"
    ]
    bloch_sq_sum = sum(hsmpo.expectation(p) ** 2 for p in local_paulis)
    return float(1 - bloch_sq_sum / n)
