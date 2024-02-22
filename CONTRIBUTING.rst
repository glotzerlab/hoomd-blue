.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Contributing
============

Contributions are welcomed via `pull requests on GitHub
<https://github.com/glotzerlab/hoomd-blue/pulls>`__. Contact the **HOOMD-blue** developers before
starting work to ensure it meshes well with the planned development direction and standards set for
the project.

Features
--------

Implement functionality in a general and flexible fashion
_________________________________________________________

New features should be applicable to a variety of use-cases. The **HOOMD-blue** developers can
assist you in designing flexible interfaces.

Maintain performance of existing code paths
___________________________________________

Expensive code paths should only execute when requested.

Optimize for the current GPU generation
_______________________________________

Write, test, and optimize your GPU kernels on the latest generation of GPUs.

Version control
---------------

Base your work off the correct branch
_____________________________________

- Base backwards compatible bug fixes on ``trunk-patch``.
- Base additional functionality on ``trunk-minor``.
- Base API incompatible changes on ``trunk-major``.

Propose a minimal set of related changes
________________________________________

All changes in a pull request should be closely related. Multiple change sets that are loosely
coupled should be proposed in separate pull requests.

Agree to the Contributor Agreement
__________________________________

All contributors must agree to the Contributor Agreement before their pull request can be merged.

Source code
-----------

Use a consistent style
______________________

The **Code style** section of the documentation sets the style guidelines for **HOOMD-blue** code.

Document code with comments
___________________________

Use doxygen header comments for classes, functions, etc. Also comment complex sections of code so
that other developers can understand them.

Compile without warnings
________________________

Your changes should compile without warnings.

Tests
-----

Write unit tests
________________

Add unit tests for all new functionality.

Validity tests
______________

The developer should run research-scale simulations using the new functionality and ensure that it
behaves as intended.

User documentation
------------------

Write user documentation
________________________

Document public-facing API with Python docstrings in Google style.

Document version status
_______________________

Add `versionadded, versionchanged, and deprecated Sphinx directives
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded>`__
to each user-facing Python class, method, etc., so that users will be aware of how functionality
changes from version to version. Remove this when breaking APIs in major releases.

Add developer to the credits
____________________________

Update the credits documentation to list the name and affiliation of each individual that has
contributed to the code.

Propose a change log entry
__________________________

Propose a short concise entry describing the change in the pull request description.
