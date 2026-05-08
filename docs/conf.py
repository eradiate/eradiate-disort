# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version as get_version

project = "eradiate-disort"
copyright = "2025, Vincent Leroy"
author = "Vincent Leroy"
version = get_version("eradiate-disort")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Core extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Third-party
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_iconify",
    "autodocsumm",
]

# -- MyST-NB options ---------------------------------------------------------
# Never execute notebooks during the Sphinx build.  Outputs must be pre-stored
# in the .ipynb files (run `pixi run nb-execute` to generate them locally or
# in CI before the docs build).
nb_execution_mode = "off"

templates_path = ["_templates"]
source_suffix = [".rst", ".md", ".ipynb"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "autosummary": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_title = "eradiate-disort"

# Use Shibuya theme
# https://shibuya.lepture.com/
html_theme = "shibuya"
html_theme_options = {
    "accent_color": "blue",
    "navigation_with_keys": True,
    "github_url": "https://github.com/eradiate/eradiate-disort",
}
