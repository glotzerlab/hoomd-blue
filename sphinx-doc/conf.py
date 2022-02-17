# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Sphinx configuration."""

import sys
import os
import sphinx
import datetime

sphinx_ver = tuple(map(int, sphinx.__version__.split('.')))

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('..'))

os.environ['SPHINX'] = '1'

extensions = [
    'nbsphinx', 'sphinx.ext.autodoc', 'sphinx.ext.autosummary',
    'sphinx.ext.napoleon', 'sphinx.ext.intersphinx', 'sphinx.ext.mathjax',
    'sphinx.ext.todo', 'IPython.sphinxext.ipython_console_highlighting'
]

napoleon_include_special_with_doc = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'gsd': ('https://gsd.readthedocs.io/en/stable/', None)
}
autodoc_docstring_signature = True

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
exclude_patterns = ['_build', 'figures']

source_suffix = '.rst'

master_doc = 'index'

project = 'HOOMD-blue'
year = datetime.date.today().year
copyright = f'2009-{ year } The Regents of the University of Michigan'
author = 'The Regents of the University of Michigan'

version = '3.0.0-beta.13'
release = '3.0.0-beta.13'

language = None

default_role = 'any'

pygments_style = 'sphinx'

todo_include_todos = False

html_theme = 'sphinx_rtd_theme'
html_css_files = ['css/hoomd-theme.css']
html_static_path = ['_static']
