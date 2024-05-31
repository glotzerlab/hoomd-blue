# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Sphinx configuration."""

import sys
import os
import sphinx
import datetime

from sphinx.domains.python import PythonDomain

# allows typing objects like variant_like to be documented correctly.
# See: https://github.com/sphinx-doc/sphinx/issues/9560
PythonDomain.object_types['class'].roles = ('class', 'exc', 'data', 'obj')
PythonDomain.object_types['data'].roles = ('data', 'class', 'obj')

sphinx_ver = tuple(map(int, sphinx.__version__.split('.')))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

os.environ['SPHINX'] = '1'

extensions = [
    'nbsphinx', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'sphinx.ext.napoleon', 'sphinx.ext.intersphinx', 'sphinx.ext.mathjax',
    'sphinx.ext.todo', 'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_copybutton'
]

if os.getenv("READTHEDOCS"):
    extensions.append("sphinxcontrib.googleanalytics")
    googleanalytics_id = "G-ZR0DNZD21E"

napoleon_include_special_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'gsd': ('https://gsd.readthedocs.io/en/stable/', None)
}
autodoc_docstring_signature = True
autodoc_typehints_format = 'short'

autodoc_mock_imports = [
    'hoomd._hoomd',
    'hoomd.version_config',
    'hoomd.md._md',
    'hoomd.metal._metal',
    'hoomd.mpcd._mpcd',
    'hoomd.minimize._minimize',
    'hoomd.hpmc._jit',
    'hoomd.hpmc._hpmc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'figures', '**/create-figures.ipynb']

source_suffix = '.rst'

master_doc = 'index'

project = 'HOOMD-blue'
year = datetime.date.today().year
copyright = f'2009-{year} The Regents of the University of Michigan'
author = 'The Regents of the University of Michigan'

version = '4.7.0'
release = '4.7.0'

language = 'en'

default_role = 'any'

pygments_style = "friendly"
pygments_dark_style = "native"

todo_include_todos = False

html_theme = 'furo'
html_static_path = ['_static']
html_logo = 'hoomdblue-logo-vertical.svg'
html_theme_options = {
    'sidebar_hide_name': True,
    'top_of_page_buttons': [],
    "navigation_with_keys": True,
    "dark_css_variables": {
        "color-brand-primary": "#5187b2",
        "color-brand-content": "#5187b2",
    },
    "light_css_variables": {
        "color-brand-primary": "#406a8c",
        "color-brand-content": "#406a8c",
    },
}
html_favicon = 'hoomdblue-logo-favicon.svg'

IGNORE_MODULES = ['hoomd._hoomd']
IGNORE_CLASSES = []

copybutton_prompt_text = "$ "
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"


def autodoc_process_bases(app, name, obj, options, bases):
    """Ignore base classes from the '_hoomd' module."""
    # bases must be modified in place.
    remove_indices = []
    for i, base in enumerate(bases):
        if (base.__module__ in IGNORE_MODULES or base.__name__.startswith("_")
                or base.__name__ in IGNORE_CLASSES):
            remove_indices.append(i)
    for i in reversed(remove_indices):
        del bases[i]


def setup(app):
    """Configure the Sphinx app."""
    app.connect('autodoc-process-bases', autodoc_process_bases)
