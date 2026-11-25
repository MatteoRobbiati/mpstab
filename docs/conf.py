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
    # Brand colors, from the logo's blue: a deeper shade for contrast on the
    # light theme, the logo's own on the dark one.
    "light_css_variables": {
        "color-brand-primary": "#2f63c4",
        "color-brand-content": "#2f63c4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#a7c7ff",
        "color-brand-content": "#a7c7ff",
    },
    # "View source" links on every page, and a GitHub icon in the footer.
    "source_repository": "https://github.com/MatteoRobbiati/mpstab/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/MatteoRobbiati/mpstab",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
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
