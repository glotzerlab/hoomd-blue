Code style
==========

All code in HOOMD-blue must follow a consistent style to ensure readability.
We provide configuration files for linters (specified below) so that developers
can automatically validate and format files.

At this time, the codebase is in a transition period to strict style guidelines.
New code should follow these guidelines. In a pull request, **DO NOT** re-style
code unrelated to your changes. Please ignore any linter errors in areas of a
file that you do not modify.

Python
------

Python code in HOOMD should follow `PEP8
<https://www.python.org/dev/peps/pep-0008>`_ with the following choices:

* 80 character line widths.
* Hang closing brackets.
* Break before binary operators.

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_ with
  `pep8-naming <https://pypi.org/project/pep8-naming/>`_
* Autoformatter: `autopep8 <https://pypi.org/project/autopep8/>`_
* See ``setup.cfg`` for the **flake8** configuration (also used by
  **autopep8**).

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``sphinx-doc/``. Docstrings should follow Google style
formatting for use in `Napoleon
<https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

C++/CUDA
--------

See ``SourceConventions.md``.

* 100 character line width.
* Indent only with spaces.
* 4 spaces per indent level.
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

* Autoformatter: We plan to provide a style configuration file once
  **clang-format** 10 is more widely available.

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
