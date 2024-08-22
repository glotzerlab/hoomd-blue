.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Code style
==========

All code in HOOMD-blue follows a consistent style to ensure readability. We
provide configuration files for linters (specified below) so that developers can
automatically validate and format files.

These tools are configured for use with `pre-commit`_ in
``.pre-commit-config.yaml``. You can install pre-commit hooks to validate your
code. Checks will run on pull requests. Run checks manually with::

    pre-commit run --all-files

.. _pre-commit: https://pre-commit.com/

Python
------

Python code in HOOMD-blue should follow `PEP8`_ with the formatting performed by
`yapf`_ (configuration in ``setup.cfg``). Code should pass all **flake8** tests
and formatted by **yapf**.

.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _yapf: https://github.com/google/yapf

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_

  * With these plugins:

    * `pep8-naming <https://github.com/PyCQA/pep8-naming>`_
    * `flake8-docstrings <https://pypi.org/project/flake8-docstrings/>`_
    * `flake8-rst-docstrings <https://github.com/peterjc/flake8-rst-docstrings>`_

  * Configure flake8 in your editor to see violations on save.

* Autoformatter: `yapf <https://github.com/google/yapf>`_

  * Run: ``pre-commit run --all-files`` to apply style changes to the whole
    repository.

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``doc/``. Docstrings should follow `Google style`_
formatting for use in `Napoleon`_, with explicit Sphinx directives where necessary to obtain the
proper formatting in the final HTML output.

.. _Google Style: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google
.. _Napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

Simulation operations should unambiguously document what calculations they perform using formal
mathematical notation and use a consistent set of symbols and across the whole codebase.
HOOMD-blue documentation should follow standard physics and statistical mechanics notation with
consistent use of symbols detailed in `notation`.

When referencing classes, methods, and properties in documentation, use ``name`` to refer to names
in the local scope (class method or property, or classes in the same module). For classes outside
the module, use the fully qualified name (e.g. ``numpy.ndarray`` or
``hoomd.md.compute.ThermodynamicQuantities``).

Provide code examples for all classes, methods, properties, etc... as appropriate. To the best
extent possible, structure the example so that a majority of users can copy and paste the example
into their script and set parameters to values reasonable for a wide range of simulations.
Add files with your examples in the Sybil configuration in ``conftest.py`` and Sybil will test
them. Document examples in this format so Sybil will parse them::

    .. rubric:: Example:

    .. code-block:: python

        obj = MyObject(parameters ...)
        ... more example code ...

C++/CUDA
--------

* Style is set by **clang-format**

  * Whitesmith's indentation style.
  * 100 character line width.
  * Indent only with spaces.
  * 4 spaces per indent level.
  * See :file:`.clang-format` for the full **clang-format** configuration.

* Naming conventions:

  * Namespaces: All lowercase ``somenamespace``
  * Class names: ``UpperCamelCase``
  * Methods: ``lowerCamelCase``
  * Member variables: ``m_`` prefix followed by lowercase with words
    separated by underscores ``m_member_variable``
  * Constants: all upper-case with words separated by underscores
    ``SOME_CONSTANT``
  * Functions: ``lowerCamelCase``

Tools
^^^^^

* Autoformatter: `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_.

Documentation
^^^^^^^^^^^^^

Documentation comments should be in Javadoc format and precede the item they document for
compatibility with many source code editors. Multi-line documentation comment blocks start with
``/**`` and single line ones start with
``///``.

.. code:: c++

    /** Describe a class
     *
     *  Note the second * above makes this a documentation comment. Some
     *  editors like to add the additional *'s on each line. These may be
     * omitted
    */
    class SomeClass
        {
        public:
            /// Single line doc comments should have three /'s
            Trigger() { }

            /** This is a brief description of a method

                This is a longer description.

                @param arg This is an argument.
                @returns This describes the return value
            */
            virtual bool method(int arg)
                {
                return false;
                }
        private:

            /// This is a member variable
            int m_var;
        };

See ``Trigger.h`` for a good example.

Other file types
----------------

Use your best judgment and follow existing patterns when styling CMake,
restructured text, markdown, and other files. The following general guidelines
apply:

* 100 character line width.
* 4 spaces per indent level.
* 4 space indent.

Editor configuration
--------------------

`Visual Studio Code <https://code.visualstudio.com/>`_ users: Open the provided
workspace file (``hoomd.code-workspace``) which provides configuration
settings for these style guidelines.
