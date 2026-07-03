import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "MatMMExtract"
author = "Shubham Ghosh, Abhishek Tewari and Mohammad Ibrahim"
project = "MatMMExtract"

release = "0.1.2"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

add_module_names = False
toc_object_entries_show_parents = "hide"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "MatMMExtract Documentation"

html_theme_options = {
    "top_of_page_button": "edit",
}

html_show_sourcelink = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

def skip_member(app, what, name, obj, skip, options):
    if name in {"main", "_parse_args"}:
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)

html_logo = "../logo.svg"

html_favicon = "_static/favicon.png"
