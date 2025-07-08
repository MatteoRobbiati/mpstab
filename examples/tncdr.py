import json

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder

max_bd = 256
n = 30
path = f"20q_5l"
hybrid_path = path + f"/results/hybrid_{max_bd}MAXBD"
hybrid_path = path + f"/results/statevector"

indexes = [i + 1 for i in range(199)]

with open(f"{path}/results/tncdr_results.json") as file:
    result = json.load(file, cls=RuntimeDecoder)

hardware_values = []
for res in result:
    hardware_values.append(res.data.evs)

approx_values = []
for idx in indexes:
    approx_values.append(np.load(f"{hybrid_path}/expectation_circuit{idx}.npy"))


plt.figure(figsize=(5, 5 * 6 / 8))
plt.scatter(np.array(hardware_values)[indexes], approx_values)
plt.ylabel("Exact")
plt.xlabel("Noisy")
plt.savefig("first_test.pdf")
