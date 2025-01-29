import numpy as np

from graph import TensorNetwork


def main():

    tn = TensorNetwork()
    tn.add_tensor(id="A", tensor=np.random.rand(4, 3, 3))
    tn.add_tensor(id="B", tensor=np.random.rand(3, 4, 6))
    tn.add_tensor(id="C", tensor=np.random.rand(4, 3, 3))
    tn.add_tensor(id="D", tensor=np.random.rand(3, 4, 6))

    tn.add_edge(
        node_in="A",
        node_out="B",
        edge_id="beta1",
        directions=(0, 1),
        comment="metadata",
    )

    tn.add_edge(
        node_in="B",
        node_out="C",
        edge_id="beta2",
        directions=(0, 1),
        comment="metadata",
    )

    tn.add_edge(
        node_in="C",
        node_out="D",
        edge_id="beta3",
        directions=(2, 0),
        comment="metadata",
    )

    tn.contract("B", "C", "beta2", "Z")


if __name__ == "__main__":
    main()
