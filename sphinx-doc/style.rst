Code style
==========

All code in HOOMD-blue must follow a consistent style to ensure readability.
We provide configuration files for linters (specified below) so that developers
can automatically validate and format files.

At this time, the codebase is in a transition period to strict style
guidelines. New code should follow these guidelines. However, **DO NOT**
re-style code unrelated to your changes. Please ignore any linter errors in
areas of a file that you do not modify.

Python
------

Python code in HOOMD should follow `PEP8
<https://www.python.org/dev/peps/pep-0008>`_ with the following choices:

* 80 character line widths.
* Hang closing brackets.
* Break before binary operators.

Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_ with
`pep8-naming <https://pypi.org/project/pep8-naming/>`_

Autoformatter: `autopep8 <https://pypi.org/project/autopep8/>`_

See ``setup.cfg`` for the **flake8** configuration (also used by **autopep8**).

C++/CUDA
--------

See ``SourceConventions.md``.

We plan to provide a style configuration file once **clang-format** 10 is more
widely available.

Other file types
----------------

Use your best judgment and follow existing patterns when styling CMake,
restructured text, markdown, and other files. The following general guidelines
apply:

* 100 character line with.
* Indent only with spaces.
* 4 space indent.
