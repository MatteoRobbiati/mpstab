import matplotlib.pyplot as plt
import numpy as np
from tncdr.stabilizer_mps.tensor_network import TensorNetwork

basis_to_amplitudes = {
    "0": (1.0, 0.0),
    "1": (0.0, 1.0),
    "+": (np.sqrt(0.5), np.sqrt(0.5)),
    "-": (np.sqrt(0.5), -np.sqrt(0.5)),
}


def star_tn(N: int):

    tn = TensorNetwork()
    c_shape = (2,) * (N - 1)
    tn.add_tensor("C", tensor=np.random.rand(*c_shape))

    for n in range(N - 1):
        tn.add_measurement(f"M{n}_1", alpha=np.random.rand(), beta=np.random.rand())
        tn.add_measurement(f"M{n}_2", alpha=np.random.rand(), beta=np.random.rand())
        tn.add_tensor(f"T{n}", tensor=np.random.rand(2, 2, 2))
        tn.add_edge("C", f"T{n}", f"X{n}", directions=(n, 0))
        tn.add_edge(f"T{n}", f"M{n}_1", f"p{n}_1", directions=(1, 0))
        tn.add_edge(f"T{n}", f"M{n}_2", f"p{n}_2", directions=(2, 0))

    return tn


def contract_measurements(star, n):
    star.contract(f"T{n}", f"M{n}_1", f"p{n}_1", "tmp")
    star.contract("tmp", f"M{n}_2", f"p{n}_2", f"T{n}")


def contract_star(star, N: int):

    contract_measurements(star, 0)
    star.contract("C", "T0", "X0", "C0")
    for n in range(1, N - 2):
        contract_measurements(star, n)
        star.contract(f"C{n-1}", f"T{n}", f"X{n}", f"C{n}")

    contract_measurements(star, N - 2)
    star.contract(f"C{N-3}", f"T{N-2}", f"X{N-2}", f"F")


def ghz_tn(N, basis_element=str):

    tn = TensorNetwork()

    A0 = np.array([[1, 0], [0, 0]])
    A1 = np.array([[0, 0], [0, 1]])
    tensor = ((0.5) ** (0.5 / N)) * np.array([A0, A1])

    for n in range(N):
        alpha, beta = basis_to_amplitudes[basis_element[n]]
        tn.add_measurement(f"M{n}", alpha, beta)
        tn.add_tensor(id=f"T{n}", tensor=tensor)
        tn.add_edge(f"M{n}", f"T{n}", f"p{n}", (0, 0))

    for n in range(N):
        n1, n2 = n % N, (n + 1) % N
        tn.add_edge(f"T{n1}", f"T{n2}", f"X{n1}->{n2}", (1, 2))
    return tn


def contract_ghz(tn, N):

    tn.contract(f"M0", f"T0", f"p0", f"C0")
    for n in range(N - 1):
        tn.contract(f"C{n}", f"T{n+1}", f"X{n}->{n+1}", f"tmp")
        tn.contract(f"M{n+1}", f"tmp", f"p{n+1}", f"C{n+1}")

    tn.contract(f"C{N-1}", f"C{N-1}", f"X{N-1}->0", f"F")


def main():

    N = 10

    # Define the basis elements to measure
    possible_symbols = list(basis_to_amplitudes.keys())
    all_zero = "0" * N
    all_plus = "+" * N
    random_mix = "".join(
        [possible_symbols[np.random.randint(len(possible_symbols))] for _ in range(N)]
    )

    basis_elements = [all_zero, all_plus, random_mix]

    for be in basis_elements:

        # Construct a GHZ state, and measure along the selected basis element
        ghz = ghz_tn(N, basis_element=be)

        # Contract the networks
        contract_ghz(ghz, N)
        print(
            f'Network contraction completed. <{be}|GHZ>: {ghz.tensornet.nodes["F"]["tensor"].item()}'
        )

    # Construct a random, star-shaped tensornetwork
    star = star_tn(N)

    # Contract the network
    contract_star(star, N)
    print(
        f'Contracted, random tensor network: {star.tensornet.nodes["F"]["tensor"].item()}'
    )


if __name__ == "__main__":
    main()
