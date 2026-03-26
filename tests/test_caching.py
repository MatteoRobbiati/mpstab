import time
import pytest

import numpy as np
from qibo import Circuit, gates, set_backend
from utils import expectation_with_qibo, set_rng_seed

from mpstab.engines import StimEngine, QuimbEngine
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import CircuitAnsatz, FloquetAnsatz

set_backend("numpy")
set_rng_seed()

def layered_magical_circuit(n_qubits, n_layers, b=np.pi/4):
    circ = Circuit(n_qubits)
    
    for q in range(n_qubits):
        circ.add(gates.H(q))
        
    for layer_idx in range(n_layers):
        for q1 in range(n_qubits - 1):
            q2 = q1 + 1
            
            circ.add(gates.RZ(q1, theta=0.25 * np.pi))
            circ.add(gates.RX(q1, theta=b))
            circ.add(gates.RZ(q2, theta=0.25 * np.pi))
            circ.add(gates.RX(q2, theta=b))
            
            circ.add(gates.CNOT(q1, q2))
            
    return circ

@pytest.mark.parametrize("n_qubits", [10, 16])
@pytest.mark.parametrize("n_layers", [3, 5])
def test_caching_contractions(n_qubits, n_layers):

    observable = "ZX" * int(n_qubits/2)

    magical_circ = layered_magical_circuit(n_qubits, n_layers)
    ansatz = CircuitAnsatz(qibo_circuit=magical_circ)

    hs = HSMPO(ansatz)
    hs.set_engines(stab_engine=StimEngine(), tn_engine=QuimbEngine(cache=True))
    start = time.time()
    hs.expectation(observable)
    first_time = time.time() - start
    print(f"Hybrid time: {first_time:.4f}s")
    
    start = time.time()
    hs.expectation(observable)
    second_time = time.time() - start
    print(f"Hybrid time: {second_time:.4f}s")

    assert first_time>second_time or np.allclose(first_time, second_time, atol=1e-2)
