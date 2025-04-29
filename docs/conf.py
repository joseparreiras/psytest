# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../psytest"))

import psytest

project = psytest.__name__
author = psytest.__author__
release = psytest.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google/NumPy style docstrings
    "sphinx.ext.mathjax",  # For LaTeX math
    "sphinx.ext.viewcode",  # Adds links to source code
    "sphinx_paramlinks",
    "sphinxcontrib.bibtex",  # For bibliography management
    "sphinx.ext.intersphinx",  # For linking to other projects
    "nbsphinx",
]

bibtex_bibfiles = ["refs.bib"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_favicon = "_static/bubbles.gif"
html_logo = "_static/logo.svg"

html_theme_options = {
    "repository_url": psytest.__url__,  # optional
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",  # relative to root
    "show_navbar_depth": 2,
    "logo_only": True,
    "home_page_in_toc": True,
    "display_version": False,
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,  # Change to True if you want _hidden attrs
    "show-inheritance": True,
}

autodoc_member_order = "bysource"

napoleon_numpy_docstring = True  # Enable NumPy style docstrings

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

nbsphinx_allow_errors = True  # Optional: allows notebooks with errors to build
