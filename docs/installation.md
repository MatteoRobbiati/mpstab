# Installation

## Prerequisites
- **Python**: 3.11, 3.12, or 3.13
- **uv** (recommended) or **pip**

## Quick Install

### Using pip

Install directly from GitHub:

```bash
pip install git+https://github.com/MatteoRobbiati/mpstab.git
```

## Development Installation

### Using uv (Recommended)

We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible environment management.

#### 1. Clone the repository

```bash
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab
```

#### 2. Install uv

If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Create the environment and install base dependencies

```bash
uv sync
```

#### 4. Activate the virtual environment

```bash
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

## Optional Backends

PyTorch and JAX are **not installed by default**. Install them on demand:

```bash
# PyTorch backend
uv sync --extra pytorch

# JAX backend
uv sync --extra jax

# Both backends
uv sync --extra pytorch --extra jax
```

## Dependency Groups

Optional development groups are installed with `--group`:

```bash
# Testing tools
uv sync --group test

# Documentation tools
uv sync --group docs

# Development utilities
uv sync --group dev

# Benchmarking tools
uv sync --group benchmark

# Install everything (all extras and all groups)
uv sync --all-extras --all-groups
```

| Group | Contents |
|-------|----------|
| `dev` | IPython, debugger, task runner (poethepoet) |
| `test` | pytest, pytest-cov, pylint |
| `docs` | Sphinx, furo theme, nbsphinx, katex |
| `benchmark` | pytest-benchmark |

### Using pip (editable install)

```bash
git clone https://github.com/MatteoRobbiati/mpstab.git
cd mpstab

# Base install
pip install -e .

# With optional backends
pip install -e ".[pytorch]"
pip install -e ".[jax]"
pip install -e ".[pytorch,jax]"
```

## Verification

```bash
python -c "from mpstab import HSMPO; print('MPSTAB installed successfully!')"

# Run the test suite (requires the test group)
uv run pytest
```

## What's Installed

The base installation includes:
- Core mpstab library
- NumPy 2.0+
- Qibo (quantum circuit framework)
- Stim (stabilizer simulator)
- Quimb (tensor network library)
- Cotengra (tensor contraction optimization)
- Matplotlib (visualization)

PyTorch and JAX are available as optional extras (`pytorch`, `jax`) and must be installed explicitly.
