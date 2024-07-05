.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

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

**Long running tests** check for correct behavior, but require up to a minute to execute. Mark long
running tests with the ``validate`` label.

**Validation tests** rigorously check that simulations sample the correct statistical ensembles.
These tests take hours to execute on many CPU cores or GPUs. Find HOOMD's validation tests in the
hoomd-validation_ repository.

Requirements
------------

The following Python packages are required to execute tests. Some tests will be skipped when
optional requirements are missing.

- `CuPy`_ (optional)
- `gsd`_ (optional)
- `h5py`_ (optional)
- `mpi4py`_ (optional)
- `pytest`_
- `rowan`_ (optional)
- `sybil`_ (optional)

.. _CuPy: https://cupy.dev/
.. _gsd: https://gsd.readthedocs.io/
.. _h5py: https://www.h5py.org/
.. _mpi4py: https://mpi4py.readthedocs.io/
.. _pytest: https://docs.pytest.org/
.. _rowan: https://rowan.readthedocs.io/
.. _sybil: https://sybil.readthedocs.io/

Running tests
-------------

Change to the build directory and execute the following commands to run the tests:

* ``ctest`` - Executes C++ tests
* ``python3 -m pytest hoomd``

pytest_ may be run outside the build directory by:

* Passing a full path to the build: ``python3 -m pytest <build-directory>/hoomd``
* After installing to an environment: ``python3 -m pytest --pyargs hoomd``

.. note::

    ``python3 -m pytest --pyargs hoomd`` tests the hoomd installation it finds by ``import hoomd``,
    which may not be the one you just built. You must also change to a directory outside the
    source, otherwise ``import hoomd`` attempts to import the uncompiled source.

.. seealso::

    See the pytest_ documentation for information on how to control output, select specific tests,
    and more.

.. _CTest: https://cmake.org/cmake/help/latest/manual/ctest.1.html
.. _pytest: https://docs.pytest.org/

Running tests with MPI
----------------------

When ``ENABLE_MPI=ON``, CTest_ will execute some tests with ``mpirun -n 1``, some with ``-n 2``
and some with ``-n 8``. Make sure your test environment (e.g. interactive cluster job) is correctly
configured before running ``ctest``.

pytest_ tests may also be executed with MPI with 2 ranks. pytest_ does not natively support
MPI. Execute it with the provided wrapper script in the build directory::

    mpirun -n 2 build/hoomd/hoomd/pytest/pytest-openmpi.sh -v -x build/hoomd

The wrapper script displays the outout of rank 0 and redirects rank 1's output to a file. Inspect
this file when a test fails on rank 1. This will result in an ``MPI_ABORT`` on rank 0 (assuming the
``-x`` argument is passed)::

    cat pytest.out.1

.. warning::

    Pass the ``-x`` option to prevent deadlocks when tests fail on only 1 rank.

.. note::

    The provided wrapper script supports OpenMPI_.

.. _OpenMPI: https://www.open-mpi.org/

Executing long runing tests
---------------------------

Longer running  tests do not execute by default. Run these with the ``--validate`` command line
option to pytest::

    $ python3 -m pytest build/hoomd --validate -m validate
    $ mpirun -n 2 hoomd/pytest/pytest-openmpi.sh build/hoomd -v -x -ra --validate -m validate

.. note::

    The ``-m validate`` option selects *only* the long running tests.

.. note::

    To run long running tests on an installed ``hoomd`` package, you need to specify additional
    options::

        python3 -m pytest --pyargs hoomd -p hoomd.pytest_plugin_validate -m validate --validate

Implementing tests
------------------

Most tests should be implemented in pytest_. HOOMD's test rig provides a ``device`` fixture that
most tests should use to cache the execution device across multiple tests and reduce test execution
time.

.. important::

    Add any new ``test_*.py`` files to the list in the corresponding ``CMakeLists.txt`` file.

Only add C++ tests for classes that have no Python interface or otherwise require low level testing.
If you are unsure, please check with the lead developers prior to adding new C++ tests. Add
new validation tests to the hoomd-validation_ repository.

.. _hoomd-validation: https://github.com/glotzerlab/hoomd-validation/
