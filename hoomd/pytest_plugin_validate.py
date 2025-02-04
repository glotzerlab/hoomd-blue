# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Command line options for pytest."""

import pytest


def pytest_addoption(parser):
    """Add HOOMD specific options to the pytest command line.

    * validate - run validation tests
    """
    parser.addoption(
        "--validate",
        action="store_true",
        default=False,
        help="Enable long running validation tests.",
    )


@pytest.fixture(autouse=True)
def skip_validate(request):
    """Skip validation tests by default.

    Pass the command line option --validate to enable these tests.
    """
    if request.node.get_closest_marker('validate'):
        if not request.config.getoption("validate"):
            pytest.skip('Validation tests not requested.')


def pytest_configure(config):
    """Define the ``validate`` marker."""
    config.addinivalue_line(
        "markers", "validate: Tests that perform long-running validations.")
