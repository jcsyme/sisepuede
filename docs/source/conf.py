# Configuration file for the Sphinx documentation builder.
import os, os.path
import sys


# -- Project information

project = "SISEPUEDE"
copyright = "2024"
author = "James Syme, Edmundo Molina Perez, Nidhi Kalra"
release = "1.0"
version = "1.1.0"

# solution to allow variable window width: https://stackoverflow.com/questions/23211695/modifying-content-width-of-the-sphinx-theme-read-the-docs

def setup(app):
    app.add_css_file("theme_adjustments.css")


# add to path
for path in [".", "../.."]:
    path_abs = os.path.abspath(path)
    print(f"conf path in readthedocs:\t{path_abs}")
    if path_abs not in sys.path:
        sys.path.insert(0, path_abs)



# -- General configuration

extensions = [
    "sphinx.directives.patches",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# -- Options for EPUB output

epub_show_urls = "footnote"
