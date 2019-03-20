Contributions are welcomed via [pull requests on GitHub](https://github.com/glotzerlab/hoomd-blue/pulls). Contact
the **HOOMD-blue** developers before starting work to ensure it meshes well with the planned development direction and
standards set for the project.

# Features

## Implement functionality in a general and flexible fashion

New features should be applicable to a variety of use-cases. The **HOOMD-blue** developers can assist you in designing
flexible interfaces.

## Maintain performance of existing code paths

Expensive code paths should only execute when requested.

## Optimize for the current GPU generation

Write, test, and optimize your GPU kernels on the latest generation of GPUs.

# Version control

## Base your work off the correct branch

Bug fixes should be based on `maint`. New features should be based on `master`.

## Propose a minimal set of related changes

All changes in a pull request should be closely related. Multiple change sets that
are loosely coupled should be proposed in separate pull requests.

## Agree to the Contributor Agreement

All contributors must agree to the Contributor Agreement ([ContributorAgreement.md](ContributorAgreement.md)) before
their pull request can be merged.

# Source code

## Use a consistent style

[SourceConventions.md](SourceConventions.md) defines the style guidelines for **HOOMD-blue** code.

## Document code with comments

Use doxygen header comments for classes, functions, etc. Also comment complex sections of code so that other
developers can understand them.

## Compile without warnings

Your changes should compile without warnings.

# Tests

## Write unit tests

Add unit tests for all new functionality.

## Validity tests

The developer should run research-scale simulations using the new functionality and ensure that it behaves as intended.

# User documentation

## Write user documentation

Document public-facing API with Python docstrings in Google style.

## Example notebooks

Add demonstrations of new functionality to [hoomd-examples](https://github.com/glotzerlab/hoomd-examples).

## Document version status

Each user-facing Python class, method, etc. with a docstring should have [versionadded, versionchanged, and
deprecated Sphinx directives](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded)
so that users will be aware of how functionality changes from version to version.

## Add developer to the credits

Update the credits documentation to reference what each developer contributed to the code.

## Propose a change log entry

Propose a short concise entry describing the change in the pull request description.
