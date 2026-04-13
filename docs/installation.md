# Installation

## Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **pip** or **Poetry** (recommended for development)

## Quick Install

### Using pip

The easiest way to get started is to install directly from GitHub:

```bash
pip install git+https://github.com/MatteoRobbiati/mpstab.git
```

## Development Installation

### Using Poetry (Recommended)

For development, testing, and documentation building, we recommend [Poetry](https://python-poetry.org/):

#### 1. Clone the repository

```bash
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab
```

#### 2. Install Poetry

If you don't have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### 3. Install dependencies

```bash
# Install base dependencies
poetry install

# Install with all optional dependencies
poetry install --with dev --with test --with docs --with benchmark

# Activate the virtual environment
poetry shell
```

### Using pip (editable install)

Alternatively, you can use pip for an editable install with optional dependencies:

```bash
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab
pip install -e ".[dev,test,docs,benchmark]"
```

## Dependency Groups

When using Poetry, you can selectively install optional dependency groups:

- **`dev`**: Development tools (IPython, debuggers, task runner)
- **`test`**: Testing framework and code quality tools (pytest, coverage, pylint)
- **`docs`**: Documentation generation tools (Sphinx, theme, plugins)
- **`benchmark`**: Performance benchmarking utilities

## Verification

To verify your installation works correctly:

```bash
# Run a simple test
python -c "from mpstab import HSMPO; print('MPSTAB installed successfully!')"

# Run the test suite (requires test dependency group)
poetry run pytest
```

## What's Installed

The base installation includes:
- Core mpstab library
- NumPy 2.0+
- Qibo (quantum circuit framework)
- Stim (stabilizer simulator)
- Quimb (tensor network library)
- PyTorch (computation backend)
- JAX (automatic differentiation)
- Cotengra (tensor contraction optimization)
- Matplotlib (visualization)
