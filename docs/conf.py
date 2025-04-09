# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath("..")) 

project = 'psytest'
copyright = '2025, Jose Antunes-Neto'
author = 'Jose Antunes-Neto'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",   # For Google/NumPy style docstrings
    "sphinx.ext.mathjax",    # For LaTeX math
    "sphinx.ext.viewcode",   # Adds links to source code
    "sphinxcontrib.bibtex",  # For bibliography management
]

bibtex_bibfiles = ["refs.bib"]
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,  # Change to True if you want _hidden attrs
    "show-inheritance": True,
}
html_static_path = ['_static']
napoleon_google_docstring = True # Enable Google style docstrings