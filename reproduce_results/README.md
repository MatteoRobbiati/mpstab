# Reproducing Benchmark Results

This directory contains the necessary configuration and scripts to reproduce the benchmarks presented in the paper. The simulations are performed using the [mssim](https://github.com/mattia-robbiano/mssim) package.

## Benchmark Parameters

The simulations sweep across several hyperparameters. Note that while most parameters sweep automatically, the **Magic fraction ($M$)** must be set manually in the settings file for each run.

| Parameter | Values |
| :--- | :--- |
| Number of qubits | 20, 35, 50, 65, 80 |
| Circuit depth | 1, 2, 3, 4, 5 |
| Bond dimension $\chi_{\max}$ | 2, 4, 8, 16, 32 |
| Magic fraction $M$ | 0, 0.2, 0.4, 0.6, 0.8, 1.0 |

*Hyperparameter values used in the benchmark simulations.*

## Setup & Execution

### 1. File Preparation
Move the provided files into the `mssim` repository structure:
- Copy `settings_bench_mpstab.json` to the root of `mssim/`.
- Copy `run_bench_mpstab.sh` to `mssim/scripts/`.

### 2. Configuration
The `settings_bench_mpstab.json` file contains the simulation specifics. Manual adjustment is required for the magic fraction ($M$) to replicate all data points in the table; other parameters will sweep automatically.

### 3. Running the Benchmark
Execute the bash script from the root of the `mssim` directory:

```bash
./scripts/run_bench_mpstab.sh settings_bench_mpstab.json
```

## Results
Once the simulation completes, the results will be saved to:
`results/result.jsonl`
