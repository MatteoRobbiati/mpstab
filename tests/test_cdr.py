import numpy as np
import matplotlib.pyplot as plt

from qibo.models import error_mitigation
from qibo.noise import NoiseModel, PauliError
from qibo import hamiltonians, set_backend

from tncdr.targets.ansatze import HardwareEfficient

nqubits = 4
nlayers = 3
nshots = 10000

set_backend("numpy")

# construct ansatz
# set density_matrix = True because we are going to do noisy simulation
ansatz = HardwareEfficient(nqubits=nqubits, nlayers=nlayers, density_matrix=True)

print(f"Ansatz:")
ansatz.circuit.draw()

# construct global observable
obs = hamiltonians.Z(nqubits=nqubits)

# generating 3 * nqubits pauli parameters to construct PauliNoiseChannel
np.random.seed(42)
noise_parameters = np.abs(np.random.normal(0, 0.01, 3 * nqubits))

print(f"Noise model parameters: {noise_parameters}\n")

# construct noise model
noise = NoiseModel()
for q in range(nqubits):
    noise.add(
        PauliError(
            [
                ("X", noise_parameters[q * 3]),
                ("Y", noise_parameters[q * 3 + 1]),
                ("Z", noise_parameters[q * 3 + 2]),
            ]
        ),
        qubits=q,
    )


# collect an ansatz unitary with all random gates
random_circ = ansatz.random_unitary(random_seed=31)

# compute exact expectation value
exact_exp = obs.expectation(random_circ().state())

# compute noiseless (but shot-noisy) expectation value
frequencies = random_circ(nshots=nshots).frequencies()
exact_shots_exp = obs.expectation_from_samples(frequencies)

# use clifford data regression on random circuit from ansatz
mit_exp, noisy_exp, fit_params, training_data = error_mitigation.CDR(
    circuit=random_circ,
    observable=obs,
    noise_model=noise,
    nshots=nshots,
    n_training_samples=40,
    full_output=True,
)

print("\n CDR applied to random unitary\n")
print(f"Exact expectation value: {exact_exp}")
print(f"Noiseless expectation value (with shots): {exact_shots_exp}")
print(f"Noisy expectation value: {noisy_exp}")
print(f"Mitigated expectation value: {mit_exp}")
print(f"Absolute difference exact-mitigated: {np.abs(exact_exp - mit_exp)}")


delta = 0.2
x = np.linspace(
    min(training_data["noisy"]) - delta, max(training_data["noisy"]) + delta, 100
)
y = fit_params[0] * x + fit_params[1]

# plot the CDR results
plt.figure(figsize=(6, 6 * 6 / 8))
plt.title("Noisy vs. noiseless results")
plt.ylabel(r"$\langle \mathcal{O} \rangle_{\rm noisy}$")
plt.xlabel(r"$\langle \mathcal{O} \rangle$")
plt.plot(x, y, color="black", ls="--", label=r"Noise map $\ell$")
plt.scatter(
    training_data["noisy"],
    training_data["noise-free"],
    color="purple",
    alpha=0.6,
    s=50,
    label="Training circuits",
)
plt.vlines(
    0,
    min(training_data["noise-free"]) - delta,
    max(training_data["noise-free"]) + delta,
    color="black",
    ls="-",
    lw=1,
)
plt.hlines(
    0,
    min(training_data["noisy"]) - delta,
    max(training_data["noisy"]) + delta,
    color="black",
    ls="-",
    lw=1,
)
plt.grid(True)
plt.legend()
plt.savefig("./plots/cdr_test.png")


# collect an ansatz unitary with all random gates
random_qc_circ = ansatz.random_quasi_clifford_unitary(random_seed=31)

# compute exact expectation value
exact_exp = obs.expectation(random_circ().state())

# compute noiseless (but shot-noisy) expectation value
frequencies = random_circ(nshots=nshots).frequencies()
exact_shots_exp = obs.expectation_from_samples(frequencies)

# use clifford data regression on random circuit from ansatz
mit_exp, noisy_exp, fit_params, training_data = error_mitigation.CDR(
    circuit=random_circ,
    observable=obs,
    noise_model=noise,
    nshots=nshots,
    n_training_samples=40,
    full_output=True,
)

print("\n CDR applied to quasi clifford unitary\n")
print(f"Exact expectation value: {exact_exp}")
print(f"Noiseless expectation value (with shots): {exact_shots_exp}")
print(f"Noisy expectation value: {noisy_exp}")
print(f"Mitigated expectation value: {mit_exp}")
print(f"Absolute difference exact-mitigated: {np.abs(exact_exp - mit_exp)}")
