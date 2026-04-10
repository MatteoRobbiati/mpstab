import os
import sys

# Path to your source code
sys.path.insert(0, os.path.abspath("../src"))

project = "mpstab"
copyright = "2026, Matteo Robbiati, Giulio Crognaletti, Mattia Robbiano, Michele Grossi"
author = "Matteo Robbiati, Giulio Crognaletti, Mattia Robbiano"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "nbsphinx",  # For Jupyter Notebook support
    "sphinx_copybutton",  # Adds "copy" button to code blocks
    "sphinxcontrib.katex",  # For fast math rendering
]

# Furo Theme Settings
html_theme = "furo"
html_title = "Documentation"
html_static_path = ["_static"]
html_logo = "mpstab_logo.png"

# Exclude build directory and notebook checkpoints
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# nbsphinx configuration: don't fail if a notebook isn't executed
nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # Change to 'always' if you want RTD to run the code
