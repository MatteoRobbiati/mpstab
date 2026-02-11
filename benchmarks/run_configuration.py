import argparse

from mpstab.analysis_scripts.utils import run_experiment


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Expectation value sampling benchmark")

    parser.add_argument(
        "--backend",
        type=str,
        choices=["mpstab", "numpy", "qibojit", "quimb", "stim"],
        required=True,
        help="Execution backend",
    )

    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Optional backend platform (e.g. cuda)",
    )

    parser.add_argument(
        "--max-bond-dim",
        type=int,
        default=None,
        help="Maximum bond dimension (used by MPSTAB / qibotn)",
    )

    parser.add_argument(
        "--replacement-probability",
        type=float,
        required=True,
        help="Replacement probability for circuit partitioning",
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
        "--rng-seed",
        type=int,
        default=42,
        help="Random number generator seed",
    )

    parser.add_argument(
        "--set-initial-state",
        type=str2bool,
        default=True,
        help=(
            "Whether to prepare an initial state. "
            "If True, prepares local RY rotations."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        backend=args.backend,
        platform=args.platform,
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
