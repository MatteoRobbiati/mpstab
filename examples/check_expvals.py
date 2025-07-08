import json

import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import RuntimeDecoder

path = f"20q_5l"
bd_list = [16, 48, 128, 256, 512]
index = 74

with open(f"{path}/results/tncdr_results.json") as file:
    result = json.load(file, cls=RuntimeDecoder)

hardware_values = []
for res in result:
    hardware_values.append(res.data.evs)

approx_values = []
for bd in bd_list:
    hybrid_path = path + f"/results/hybrid_{bd}MAXBD"
    approx_values.append(np.load(f"{hybrid_path}/expectation_circuit{index}.npy"))


exact_value = np.load(f"{path}/results/statevector/expectation_circuit{index}.npy")

print(f"Hardware value: {hardware_values[index+1]}")
for i in range(len(bd_list)):
    print(f"HSMPO with BD={bd_list[i]}: {approx_values[i]}")
print(f"Statevector simulation: {exact_value}")
