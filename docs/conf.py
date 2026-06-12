import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "Paper Pipeline"
author = "NAN"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True

add_module_names = False
toc_object_entries_show_parents = "hide"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"

html_title = "Paper Pipeline"
html_show_sourcelink = False

pygments_style = "sphinx"

def skip_member(app, what, name, obj, skip, options):
    if name in {"main", "_parse_args"}:
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "titles_only": False,
}


html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "titles_only": False,
}

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 10,
    "titles_only": False,
    "includehidden": False,
}

html_sidebars = {
    "**": [
        "searchbox.html",
        "globaltoc.html",
    ]
}
