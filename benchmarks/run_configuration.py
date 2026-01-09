import argparse
from mpstab.analysis_scripts.utils import run_experiment

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MPSTAB expectation value sampling"
    )
    parser.add_argument(
        "--max-bond-dim",
        type=int,
        default=None,
        help="Maximum bond dimension for MPSTAB",
    )
    parser.add_argument(
        "--replacement-probability",
        type=float,
        required=True,
        help="Replacement probability for partitioning",
    )
    parser.add_argument(
        "--nqubits",
        type=int,
        required=True,
        help="Number of qubits",
    )
    parser.add_argument(
        "--nlayers",
        type=int,
        required=True,
        help="Number of ansatz layers",
    )
    parser.add_argument(
        "--nruns",
        type=int,
        default=10,
        help="Number of independent runs",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Random number generator seed"
    )
    parser.add_argument(
        "--set_initial_state",
        type=bool,
        default=True,
        help="Whether an initial state is required. If True, it is prepared with a set of local RY rotations, one per qubit."
    )

    args = parser.parse_args()

    run_experiment(
        max_bond_dim=args.max_bond_dim,
        replacement_probability=args.replacement_probability,
        nqubits=args.nqubits,
        nlayers=args.nlayers,
        nruns=args.nruns,
        rng_seed=args.rng_seed,
        set_initial_state=args.set_initial_state,
    )


if __name__ == "__main__":
    main()