
<img width="350" height="150" alt="image" src="https://github.com/user-attachments/assets/059b9588-4ae6-4951-9ab8-3f30c989e592" />

# MPSTAB: Hybrid Stabilizers-MPO Quantum Circuit Simulator

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/Coverage-See%20htmlcov-blue.svg)](htmlcov/index.html)

## Overview

**MPSTAB** is a cutting-edge quantum circuit simulator implementing a hybrid **stabilizers-MPO** (Matrix Product Operator) formalism in pure Python. This package provides an efficient framework for simulating quantum circuits by leveraging the power of both stabilizer states and tensor network methods, enabling accurate simulation of larger quantum systems with reduced computational overhead.

The simulator offers in-house stabilizers and tensor network engines, alongside with state-of-art Python packages such as Stim and Quimb.
The quantum circuits and observables interface is inherited form the open-source Python project Qibo.


## Key Features

- 🎯 **Hybrid Formalism**: Combines stabilizer states with tensor networks for efficient simulation
- 🚀 **Flexible engines**: Flexible backend support: from in-house Clifford simulator and Tensor Network engines to Stim and Quimb.
- **Qibo backend provider**: `mpstab` can be used as a backend provides for Qibo, a state-of-art full-stack quantum computing framework.
- 📊 **Fidelity Tracking**: Built-in fidelity lower bounds and truncation monitoring
- 🔧 **Pre-computed quantum circuit Ansätze**: Pre-built quantum circuits for variational algorithms
- 🧮 **Arbitrary Observables**: Support for Pauli string expectation values
- ⚡ **Differentiability**: Integrated with Quimb, `mpstab` inherits differentiability over the circuit executions

## Installation

Clone the repository and install using pip:

```bash
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab
pip install .
```

Or install with optional dependencies for development and documentation:

```bash
pip install -e ".[dev,tests,docs,benchmark]"
```

### Requirements

- Python ≥ 3.11 and < 3.14
- NumPy ≥ 2.0.0
- PyTorch (any version)
- JAX (any version)
- Qibo ≥ 0.2.19
- Stim ≥ 1.15.0
- Quimb ≥ 1.12.1
- Cotengra (tensor contraction optimization)

## Quick Start

### Basic Circuit Simulation

```python
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient

# Create a quantum circuit ansatz
ansatz = HardwareEfficient(nqubits=5, nlayers=3)

# Initialize the hybrid stabilizer MPO simulator
simulator = HSMPO(ansatz=ansatz, max_bond_dimension=64)

# Compute expectation value of a Pauli observable
observable = "ZIZ"
expectation_value = simulator.expectation(observable)
print(f"<{observable}> = {expectation_value}")

# Access fidelity lower bound (indicates truncation error)
fidelity = simulator.fidelity_lower_bound
print(f"Fidelity Lower Bound: {fidelity}")
```

### Scaling Analysis

```python
# Benchmark fidelity decay with system size
from mpstab.models.ansatze import HardwareEfficient

max_bond_dim = 8
for n_qubits in range(5, 21):
    ansatz = HardwareEfficient(nqubits=n_qubits, nlayers=5)
    hs = HSMPO(ansatz=ansatz, max_bond_dimension=max_bond_dim)

    # Trigger contraction and extract fidelity
    _ = hs.expectation("Z" * n_qubits)
    print(f"Qubits: {n_qubits}, Fidelity: {hs.fidelity_lower_bound:.6f}")
```

## Project Structure

```
mpstab/
├── src/mpstab/
│   ├── __init__.py
│   ├── utils.py
│   ├── engines/               # Computation backends
│   │   ├── stabilizers/       # Stabilizer-based simulators
│   │   └── tensor_networks/   # Tensor network engines
│   ├── evolutors/             # Time evolution and circuits
│   │   └── hsmpo.py           # Hybrid Stabilizer MPO main class
│   ├── models/                # Quantum circuit models
│   │   └── ansatze/           # Pre-built circuit patterns
│   └── qibo_backend/          # Qibo integration
├── tests/                     # Comprehensive test suite
│   ├── test_compute_fidelity.py
│   ├── test_hsmpo_properties.py
│   ├── test_backends_interface.py
│   ├── test_backends_support.py
│   ├── test_surrogate_expectation.py
│   ├── test_pure_clifford_circuits.py
│   └── test_caching.py
├── docs/                      # Sphinx documentation
├── examples/                  # Example notebooks and scripts
└── pyproject.toml             # Poetry project configuration
```

## Testing

Run the complete test suite:

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=mpstab --cov-report=html
```

Specific test categories:

```bash
# Test backend interfaces
pytest tests/test_backends_interface.py

# Test fidelity computation
pytest tests/test_compute_fidelity.py

# Test hardware-efficient circuits
pytest tests/test_hsmpo_properties.py
```

## Development Workflow

### Code Quality

Code formatting and linting are managed through `pre-commit`. To set up:

```bash
pip install pre-commit
cd mpstab
pre-commit install
```

From now on, `pre-commit` automatically checks and standardizes code on every commit. Additional checks:

```bash
# Lint with pylint
poe lint

# Check for warnings
poe lint-warnings
```

### Running Tasks

Use Poetry's task runner:

```bash
# Run tests
poe test

# Run benchmarks
poe bench

# Build documentation
poe docs
```

## Contribution Guidelines

Contributions are welcome! Please follow these guidelines:

1. **Code Style**: Use `pre-commit` to maintain consistent formatting. Run `pre-commit install` and let it automatically standardize your code.

2. **Testing**:
   - Write tests for all new features
   - Ensure all tests pass locally: `pytest`
   - Include tests in your pull request

3. **Documentation**:
   - Document all public functions and classes
   - Update documentation when changing APIs
   - Add docstrings following NumPy style

4. **Pull Request Process**:
   - Fork the repository
   - Create a feature branch
   - Commit your changes with clear messages
   - Open a pull request with a descriptive title
   - Wait for review from a core team member

## Citation

If you use **MPSTAB** in your research, please cite:

```bibtex
@software{mpstab2024,
  author = {Robbiati, Matteo and Crognaletti, Giulio and Robbiano, Mattia and Grossi, Michele},
  title = {MPSTAB: Hybrid Stabilizers-MPO Quantum Circuit Simulator},
  url = {https://github.com/MatteoRobbiati/mpstab},
  year = {2024}
}
```

## Architecture Overview

<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/925b99ba-bd72-4eb1-a895-9e3b518e4f6d" />

## Roadmap

- [ ] Performance comparison vs. quimb for pure tensor networks
- [ ] Scaling hybrid benchmarks to larger qubit counts
- [ ] Support for sum of Pauli observables
- [ ] Additional quantum gate sets
- [ ] GPU acceleration improvements
- [ ] Interactive visualization tools

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Matteo Robbiati** - Core development
- **Giulio Crognaletti** - Contributions
- **Mattia Robbiano** - Contributions
- **Michele Grossi** - Contributions

## Support & Feedback

For issues, feature requests, or contributions, please open an issue on [GitHub Issues](https://github.com/MatteoRobbiati/mpstab/issues).

Questions and discussions are welcome in the [GitHub Discussions](https://github.com/MatteoRobbiati/mpstab/discussions) forum.
