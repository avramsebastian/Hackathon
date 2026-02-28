# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_dark_mode

# Add the project root to sys.path so Sphinx can import all modules
sys.path.insert(0, os.path.abspath("../.."))  # assuming conf.py is in docs/source

project = 'Hexa Hackaton'
copyright = '2026, Hexa Team'
author = 'Hexa Team'
release = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",    # generate docs from docstrings
    "sphinx.ext.napoleon",   # parse Google/NumPy style docstrings
    "sphinx.ext.viewcode",   # add links to source code
    "sphinx_rtd_dark_mode"
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd"  # just the theme name
default_dark_mode = True
html_static_path = ['_static']

# conf.py
# -- Mock imports to avoid ModuleNotFoundError during docs build --
# Mock imports that are optional or heavy
autodoc_mock_imports = ["pygame"]
