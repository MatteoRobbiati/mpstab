import os
import sys

# Path to your source code
sys.path.insert(0, os.path.abspath("../src"))

project = "mpstab"
copyright = "2026, Matteo Robbiati, Giulio Crognaletti, Mattia Robbiano, Roberto Di Lauro, Michele Grossi"
author = "Matteo Robbiati, Giulio Crognaletti, Mattia Robbiano, Roberto Di Lauro"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google-style docstrings
    "sphinx.ext.viewcode",
    "nbsphinx",  # For Jupyter Notebook support
    "sphinx_copybutton",  # Adds "copy" button to code blocks
    "sphinxcontrib.katex",  # For fast math rendering
]

# Docstrings are written in Google style only.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render "Attributes:" sections as :ivar: fields instead of separate
# attribute directives, so dataclass fields (also picked up by autodoc's
# :undoc-members:) are not documented twice.
napoleon_use_ivar = True

# Furo Theme Settings
html_theme = "furo"
html_title = "Documentation"
html_static_path = ["_static"]  # custom.css (see themed-figure below)
# Furo switches the logo with the page theme (docs/scripts/draw_logo.py).
html_theme_options = {
    "light_logo": "logo/mpstab_logo_light.svg",
    "dark_logo": "logo/mpstab_logo_dark.svg",
}

# Exclude build directory and notebook checkpoints
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# nbsphinx configuration: don't fail if a notebook isn't executed
nbsphinx_allow_errors = True
nbsphinx_execute = "never"  # Change to 'always' if you want RTD to run the code


# ---------------------------------------------------------------------------
# ``themed-figure``: a figure whose image has a dark-theme twin
# ``<name>_dark.<ext>`` (written by ``docs/scripts/draw_ansatze.py``). Both
# images go in the same figure, one shown per Furo theme, so the caption is
# written once.
# ---------------------------------------------------------------------------
from docutils import nodes  # noqa: E402
from docutils.parsers.rst.directives.images import Figure  # noqa: E402


class ThemedFigure(Figure):
    def run(self):
        result = super().run()
        for image in result[0].findall(nodes.image):
            dark = image.deepcopy()
            stem, ext = os.path.splitext(image["uri"])
            dark["uri"] = f"{stem}_dark{ext}"
            image["classes"].append("only-light")
            dark["classes"].append("only-dark")
            image.parent.insert(image.parent.index(image) + 1, dark)
            break
        return result


def setup(app):
    app.add_directive("themed-figure", ThemedFigure)
    app.add_css_file("custom.css")
