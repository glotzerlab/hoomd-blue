Contributions are welcomed via pull requests on Github. First contact the HOOMD-blue developers prior to beginning
your work to ensure that your plans mesh well with the planned development direction and standards set for the project.
Then implement your code.

Submit a pull request on Github. Multiple developers and/or users will review requested changes and make comments.
This The rest of this file will be used as a checklist to review the pull request. The lead developer will merge into
the mainline after the review is complete and approved.

# Features

## Implement functionality in a general and flexible fashion

HOOMD-blue provides a lot of flexibility to the user. Your pull request should provide something that is applicable
to a variety of use-cases and not just the one thing you might need it to do for your research. Speak to the lead
developers before writing your code, and they will help you make design choices that allow flexibility.

## Do not degrade performance of existing code paths

New functionalities should only activate expensive code paths when they are requested by the user. Do not slow down
existing code.

## Optimize for the current GPU generation

Write and test your GPU kernels on the latest generation of GPUs, and take advantage of all they have to offer.

## Connect GPU kernels to autotuners

Autotuning kernel launch parameters boosts performance significantly. Use autotuners on all new GPU kernels.

# Version control

## Base your work off the correct branch

Bug fixes should be based on `maint`. New features should be based on `master`.

## Propose a single set of related changes

Changes proposed in a single topic branch / pull request should all be related to each other. Don't propose too
many changes at once, review becomes challenging. Multiple new features that are loosely coupled should be completed
in separate topic branches. It is OK if the branch for `feature2` is based on `feature1` - as long as it is made clear
that `feature1` should be merged before the review of `feature2`. It is better to merge both `feature1` and `feature1`
into a temporary integration branch during testing.

## Keep changes to a minimum

Don't go and "fix" spelling errors all over the code, or make lots of whitespace changes along with a new feature.
If there are spelling errors to fix, propose that in a separate pull request :)

## Agree to the contributor agreement

All contributors must agree to the Contributor Agreement ([ContributorAgreement.md](ContributorAgreement.md)) before their pull request can be merged.

# Source code

## Use a consistent style

It is important to have a consistent style throughout the source code. See [SourceConventions.md](SourceConventions.md)
for the defined style guidelines for hoomd code.

## Document code with comments

Use doxygen header comments for classes, functions, etc... Also comment complex sections of code so that other
developers can understand them.

## Compiles without warnings

Your changes should compile without warnings.

# Tests

## Write unit tests

All new functionality in hoomd should be tested with automatic unit tests that execute in a few seconds. High level
features should be tested from python, and the python tests should attempt to cover all options that the user can
select.

## Validity tests

In addition to the unit tests, the developer should run research-scale simulations using the new functionality and
ensure that it behaves as intended.

# User documentation

## Write user documentation

User documentation for the user facing script commands should be documented with docstrings in napoleon format.
Include examples on using new functionality.

## Link new commands into the documentation index

The master command index needs a reference to new script commands so they are easy to find for users.

## Document version status

Each user-facing python class, method, etc... with a docstring should have [versionadded, versionchanged, and
deprecated sphinx paragraphs](www.sphinx-doc.org/en/stable/markup/para.html) so that users will be aware of
how functionality changes from version to version (post-2.0).

## Add developer to the credits

Developers need to be credited for their work. Update the credits documentation to reference what each developer
contributed to the code.

## Propose a change log entery in the pull request comments

Propose a short concise entry describing the change for use in the change log.
