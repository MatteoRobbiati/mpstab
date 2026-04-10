
<img width="350" height="150" alt="image" src="https://github.com/user-attachments/assets/059b9588-4ae6-4951-9ab8-3f30c989e592" />

# MPSTAB: Hybrid Stabilizers-MPO Quantum Circuit Simulator

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)
[![Code Coverage](https://img.shields.io/badge/Coverage-See%20htmlcov-blue.svg)](htmlcov/index.html)

## Overview

**MPSTAB** is a cutting-edge quantum circuit simulator implementing a hybrid **stabilizers-MPO** (Matrix Product Operator) formalism in pure Python. This package provides an efficient framework for simulating quantum circuits by leveraging the power of both stabilizer states and tensor network methods, enabling accurate simulation of larger quantum systems with reduced computational overhead.

The simulator offers in-house stabilizers and tensor network engines, alongside with state-of-art Python packages such as [Stim](https://github.com/quantumlib/Stim) and [Quimb](https://quimb.readthedocs.io/).
The quantum circuits and observables interface is inherited from the open-source Python project [Qibo](https://qibo.science/).


## Key Features

- 🎯 **Hybrid Formalism**: Combines stabilizer states with tensor networks for efficient simulation
- 🚀 **Flexible engines**: Flexible backend support: from in-house Clifford simulator and Tensor Network engines to [Stim](https://github.com/quantumlib/Stim) and [Quimb](https://quimb.readthedocs.io/).
- ⚙️ **[Qibo](https://qibo.science/) backend provider**: `mpstab` can be used as a backend provider for Qibo, a state-of-art full-stack quantum computing framework.
- 📊 **Fidelity Tracking**: Built-in fidelity lower bounds and truncation monitoring
- 🔧 **Pre-computed quantum circuit Ansätze**: Pre-built quantum circuits for variational algorithms
- 🧮 **Arbitrary Observables**: Support for Pauli string expectation values
- ⚡ **Differentiability**: Integrated with Quimb, `mpstab` inherits differentiability over the circuit executions

## Installation

### Using pip

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

### Using Poetry (Recommended)

We recommend using [Poetry](https://python-poetry.org/) for dependency management and development:

```bash
# Clone the repository
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab

# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Install with optional dependencies for development
poetry install --with dev --with tests --with docs --with benchmark

# Activate the virtual environment
poetry shell
```

Poetry automatically handles virtual environment management and locks all dependencies for reproducibility.


## Quick Start

### Basic Circuit Simulation

```python
from mpstab.evolutors.hsmpo import HSMPO
from mpstab.models.ansatze import HardwareEfficient

# Create a quantum circuit ansatz
ansatz = HardwareEfficient(nqubits=5, nlayers=3)

# Initialize the hybrid stabilizer MPO simulator
surrogate = HSMPO(ansatz=ansatz, max_bond_dimension=64)

# Compute expectation value of a Pauli observable
observable = "ZIZ"
expectation_value = surrogate.expectation(observable)
print(f"<{observable}> = {expectation_value}")

# Access fidelity lower bound (indicates truncation error)
fidelity = surrogate.fidelity_lower_bound
print(f"Fidelity Lower Bound: {fidelity}")
```


## Testing

Run the complete test suite using [pytest](https://docs.pytest.org/):

```bash
pytest
```

Run with coverage report:

```bash
pytest --cov=mpstab --cov-report=html
```

Or using Poetry's task runner:

```bash
poetry run pytest
```


## Development Workflow

### Code Quality

Code formatting and linting are managed through [pre-commit](https://pre-commit.com/). To set up:

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

If using Poetry, you can run linting within the Poetry environment:

```bash
poetry run pylint src/**/*.py -E
```

### Running Tasks

Use [Poetry](https://python-poetry.org/)'s task runner with [Poethepoet](https://poethepoet.nauce.org/):

```bash
# Run tests
poetry run poe test

# Run benchmarks
poetry run poe bench

# Build documentation
poetry run poe docs
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



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Authors

- **Matteo Robbiati** - Core development
- **Giulio Crognaletti** - Core development
- **Mattia Robbiano** - Core development

## Support & Feedback

For issues, feature requests, or contributions, please open an issue on [GitHub Issues](https://github.com/MatteoRobbiati/mpstab/issues).

Questions and discussions are welcome in the [GitHub Discussions](https://github.com/MatteoRobbiati/mpstab/discussions) forum.
