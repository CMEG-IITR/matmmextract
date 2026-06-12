import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "MultiMat"
author = "NAN"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

add_module_names = False
toc_object_entries_show_parents = "hide"

html_theme = "sphinx_rtd_theme"
html_title = "MultiMat"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "titles_only": False,
}

html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
    ]
}

def skip_member(app, what, name, obj, skip, options):
    if name in {"main", "_parse_args"}:
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)
