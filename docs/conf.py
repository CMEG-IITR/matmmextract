
project = "paper-pipeline"



extensions = [

    "sphinx.ext.autodoc",

    "sphinx.ext.napoleon",

    "sphinx.ext.viewcode",

]



html_theme = "pydata_sphinx_theme"


def skip_member(app, what, name, obj, skip, options):
    if name == "main":
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)

