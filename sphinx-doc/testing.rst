Testing
=======

All code in HOOMD must be tested to ensure that it operates correctly.

**Unit tests** check that basic functionality works, one class at a time. Unit
tests assume internal knowledge about how classes work and may use unpublished
APIs to stress test all possible input and outputs of a given class in order to
exercise all code paths. For example, test that the box class properly wraps
vectors back into the minimum image. Unit tests should complete in a fraction of
a second.

**System integration tests** check that many classes work together to produce
correct output. These tests are black box tests and should only use user-facing
APIs to provide inputs and check for correct outputs. For example, test that the
hard sphere HPMC simulation executes for several steps. System integration tests
may take several seconds.

**Validation tests** rigorously check that HOOMD simulations sample the correct
statistical ensembles. For example, validate the a Lennard-Jones simulation at a
given density matches the pressure in the NIST reference. Validation tests
should run long enough to ensure reasonable sampling, but not too long. These
test run in a CI environment on every pull request. Try to keep validation tests
under 10 minutes whenever possible.

Running tests
-------------

Execute the following commands to run the tests:

* ``ctest`` - Executes C++ tests
* ``python3 -m pytest --pyargs hoomd``

**ctest** must be run in the **build** directory. When you run **pytest**
outside of the **build** directory, it will test the HOOMD installed by ``make
install``.

See the `pytest invocation docs <https://docs.pytest.org/en/latest/usage.html>`_
for information on how to control **pytest** output, select specific tests, and
more.

Running tests with MPI
----------------------

When **ENABLE_MPI=ON**, **ctest** will execute some tests with ``mpirun -n 1``,
some with ``-n 2`` and some with ``-n 8``. Make sure your test environment (e.g.
interactive cluster job) is correctly configured before running ``ctest``.

**pytest** tests may also be executed with MPI with 2 ranks. **pytest** does not
natively support MPI. Execute it with the provided wrapper script. In the
**build** directory::

    mpirun -n 2 hoomd/pytest/pytest-openmpi.sh -v -x --pyargs hoomd

The wrapper script displays the outout of rank 0 and redirects rank 1's output
to a file. Inspect this file when a test fails on rank 1. This will result in an
**MPI_ABORT** on rank 0 (assuming the ``-x`` argument is passed)::

    cat pytest.out.1

.. warning::

    Pass the ``-x`` option to prevent deadlocks when tests fail on only 1 rank.

.. note::

    The provided wrapper script supports **OpenMPI**. It will require
    modifications to work with other MPI stacks.

Implementing tests
------------------

Most tests should be implemented in pytest. See ``hoomd/pytest/test_example.py``
for some examples and the `pytest documentation <https://docs.pytest.org>`_ for
more details. HOOMD's test rig provides a ``device`` fixture that most tests
should use to cache the execution device across multiple tests and reduce test
execution time.

.. note::

    Add any new ``test_*.py`` files to the list in the corresponding
    ``CMAKELists.txt`` file.

Only add C++ tests for classes that have no Python interface or otherwise
require low level testing. If you are unsure, please check with the lead
developers prior to adding new C++ tests.
