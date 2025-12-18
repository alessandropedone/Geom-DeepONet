import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Geom-DeepONet'
copyright = '2025, Alessandro Pedone, Marta Pignatelli'
author = 'Alessandro Pedone, Marta Pignatelli'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {'members': True, 'private-members': True}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ['_static']

# Base URL for GitHub Pages (used for canonical links / SEO)
html_baseurl = "https://alessandropedone.github.io/Geom-DeepONet/"

# Ensure Furo generates relative URLs correctly for GitHub Pages
html_theme_options = {
    "light_logo": "_static/logo-light.svg",
    "dark_logo": "_static/logo-dark.svg",
    "sidebar_hide_name": False,
}

# Optional: GitHub integration in the top-right corner
html_context = {
    "display_github": True,
    "github_user": "alessandropedone",
    "github_repo": "Geom-DeepONet",
    "github_version": "main/docs/",  # points to docs folder in main branch
}

# Extra CSS/JS if needed
html_css_files = [
    "custom.css",
]