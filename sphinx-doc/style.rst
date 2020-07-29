Code style
==========

All code in HOOMD-blue follows a consistent style to ensure readability. We
provide configuration files for linters (specified below) so that developers can
automatically validate and format files.

At this time, the codebase is in a transition period to strict style guidelines.
New code should follow these guidelines. In a pull request, **DO NOT** re-style
code unrelated to your changes. Please ignore any linter errors in areas of a
file that you do not modify.

Python
------

Python code in HOOMD should follow `PEP8
<https://www.python.org/dev/peps/pep-0008>`_ with  with the formatting performed
by `yapf <https://github.com/google/yapf>`_ (configuration in ``setup.cfg``).
Code should pass all **flake8** tests and formatted by **yapf**.

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_

  * With these plugins:

    * `pep8-naming <https://github.com/PyCQA/pep8-naming>`_
    * `flake8-docstrings <https://gitlab.com/pycqa/flake8-docstrings>`_
    * `flake8-rst-docstrings <https://github.com/peterjc/flake8-rst-docstrings>`_

  * Run: ``flake8`` to see a list of linter violations.

* Autoformatter: `yapf <https://github.com/google/yapf>`_

  * Run: ``yapf -d -r .`` to see needed style changes.
  * Run: ``yapf -i file.py`` to apply style changes to a whole file, or use
    your IDE to apply **yapf** to a selection.

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``sphinx-doc/``. Docstrings should follow `Google style
<https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html#example-google>`_
formatting for use in `Napoleon
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.


C++/CUDA
--------

* Style is set by **clang-format** >= 10

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

  * Run: ``./run-clang-format.py -r .`` to see needed changes.
  * Run: ``clang-format -i file.c`` to apply the changes.


Documentation
^^^^^^^^^^^^^

Documentation comments should be in Javadoc format and precede the item they
document for compatibility with Doxygen and most source code editors. Multi-line
documentation comment blocks start with ``/**`` and single line ones start with
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
