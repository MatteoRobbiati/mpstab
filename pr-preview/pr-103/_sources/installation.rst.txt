Installation
=============

Prerequisites
-------------

- **Python**: 3.11, 3.12, or 3.13
- **uv** (recommended) or **pip**

Quick Install
-------------

Using pip
~~~~~~~~~

Install directly from GitHub::

    pip install git+https://github.com/MatteoRobbiati/mpstab.git

Development Installation
-------------------------

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

We recommend `uv <https://docs.astral.sh/uv/>`_ for fast, reproducible environment management.

**1. Clone the repository**

::

    git clone https://github.com/MatteoRobbiati/mpstab.git
    cd mpstab

**2. Install uv**

If you don't have uv installed::

    curl -LsSf https://astral.sh/uv/install.sh | sh

**3. Create the environment and install base dependencies**

::

    uv sync

**4. Activate the virtual environment**

::

    source .venv/bin/activate      # macOS / Linux
    .venv\Scripts\activate         # Windows

Optional Backends
-----------------

PyTorch and JAX are **not installed by default**. Install them on demand:

::

    # PyTorch backend
    uv sync --extra pytorch

    # JAX backend
    uv sync --extra jax

    # Both backends
    uv sync --extra pytorch --extra jax

The foldable low-level-rustiq resynthesis path (see
:doc:`guides/rustiq_resynthesis`) needs the optional ``rustiq`` package, built
from source (it ships no PyPI wheels):

::

    uv sync --extra rustiq

Dependency Groups
-----------------

Optional development groups are installed with ``--group``:

::

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

Available groups:

- **``dev``**: Development tools (IPython, debugger, task runner)
- **``test``**: Testing framework (pytest, coverage, pylint)
- **``docs``**: Documentation generation (Sphinx, theme, plugins)
- **``benchmark``**: Performance benchmarking utilities

Using pip (editable install)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    git clone https://github.com/MatteoRobbiati/mpstab.git
    cd mpstab

    # Base install
    pip install -e .

    # With optional backends
    pip install -e ".[pytorch]"
    pip install -e ".[jax]"
    pip install -e ".[pytorch,jax]"

Verification
------------

To verify your installation works correctly::

    python -c "from mpstab import HSMPO; print('MPSTAB installed successfully!')"

    # Run the test suite (requires the test group)
    uv run pytest
