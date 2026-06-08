"""Sphinx configuration for eradiate-disort documentation."""

import datetime
from importlib.metadata import version as get_version

# Project information
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "eradiate-disort"
copyright = f"2025-{datetime.datetime.now().year}, Rayference"
author = "The Eradiate Team"
version = get_version("eradiate-disort")
release = version

# General configuration
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    # Core extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    # Third-party
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_iconify",
    "autodocsumm",
    "sphinxcontrib.bibtex",
]

# Templates and static files
templates_path = ["_templates"]
source_suffix = [".rst", ".md", ".ipynb"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "nanodisort": ("https://nanodisort.readthedocs.io/stable/", None),
    "eradiate": ("https://eradiate.readthedocs.io/en/stable/", None),
}

# BibTeX configuration
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# Todo options
todo_include_todos = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "autosummary": True,
    "show-inheritance": True,
}

# MyST-NB options
# Never execute notebooks during the Sphinx build.  Outputs must be pre-stored
# in the .ipynb files (run `pixi run nb-execute` to generate them locally or
# in CI before the docs build).
nb_execution_mode = "off"

# HTML output options
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Eradiate DISORT documentation"
html_short_title = "Eradiate DISORT documentation"
html_favicon = "_images/icon_eradiate.png"
html_show_sourcelink = False

# Use Shibuya theme
# https://shibuya.lepture.com/
html_theme = "shibuya"
html_theme_options = {
    "accent_color": "blue",
    "navigation_with_keys": True,
    "github_url": "https://github.com/eradiate/eradiate-disort",
    "light_logo": "_static/eradiate-disort-logo-typo_simple-black.svg",
    "dark_logo": "_static/eradiate-disort-logo-typo_simple-white.svg",
    "nav_links_align": "center",
    "nav_links": [
        {"title": "User guide", "url": "user_guide/index"},
        {"title": "Examples", "url": "examples/index"},
        {"title": "API", "url": "api/eradiate_disort"},
        {"title": "Contributing", "url": "dev/index"},
        {
            "title": "Main docs",
            "url": "https://eradiate.readthedocs.org/",
            "external": True,
        },
    ],
}

# Pygments options
pygments_style = "default"
