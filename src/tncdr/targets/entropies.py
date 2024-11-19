from qibo.quantum_info import renyi_entropy

def scan_renyi_entropy(state, n_moments):
    """Scan Renyi entropy of a quantum state up to n_moments."""
    renyi_entropies = []
    for n in range(n_moments):
        renyi_entropies.append(renyi_entropy(state=state, alpha=n))
    return renyi_entropies