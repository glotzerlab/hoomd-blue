.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Building from source
====================

To build the **HOOMD-blue** from source:

1. `Install prerequisites`_:

   .. code-block:: bash

       micromamba install cmake eigen git python numpy pybind11

2. `Obtain the source`_:

   .. code-block:: bash

       git clone --recursive git@github.com:glotzerlab/hoomd-blue.git

3. Change to the repository directory:

   .. code-block:: bash

       cd hoomd-blue

3. `Configure`_:

   .. code-block:: bash

       cmake -B build -S . -GNinja

4. `Build the package`_:

   .. code-block:: bash

       cd build

   .. code-block:: bash

       ninja

6. `Run tests`_:

   .. code-block:: bash

       python3 -m pytest hoomd

5. `Install the package`_ (optional):

   .. code-block:: bash

       ninja install

To build the documentation from source (optional):

1. `Install prerequisites`_:

   .. code-block:: bash

       micromamba install sphinx sphinx-copybutton furo nbsphinx ipython

2. `Build the documentation`_:

   .. code-block:: bash

       sphinx-build -b html sphinx-doc html

The sections below provide details on each of these steps.

.. _Install prerequisites:

Install prerequisites
---------------------

You will need to install a number of tools and libraries to build **HOOMD-blue**. The options 
``ENABLE_MPI``, ``ENABLE_GPU``, ``ENABLE_TBB``, and ``ENABLE_LLVM`` each require additional
libraries when enabled.

Install the required dependencies:

.. code-block:: bash

   micromamba install cmake eigen git python numpy pybind11

Install additional packages needed to run the unit tests:

.. code-block:: bash

   micromamba install pytest

Install additional packages needed to build the documentation:

.. code-block:: bash

   micromamba install sphinx sphinx-copybutton furo nbsphinx ipython

.. note::

    This guide assumes that you use the micromamba_ package manager. Adjust the commands
    appropriately for the package manager of your choice.

.. warning::

    When using a ``conda-forge`` environment for development, make sure that the environment does
    not contain ``clang``, ``gcc``, or any other compiler or linker. These interfere with the native
    compilers on your system and will result in compiler errors when building, linker errors when
    running, or segmentation faults.

.. _micromamba: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html

**General requirements:**

- **C++17** capable compiler
- **CMake**
- **NumPy**
- **pybind11**
- **Python**
- **Eigen**

**For MPI parallel execution** (required when ``ENABLE_MPI=on``):

- A **MPI** library (tested with OpenMPI)
- **cereal**

**For GPU execution** (required when ``ENABLE_GPU=on``):

- **NVIDIA CUDA Toolkit**

  *OR*

- AMD ROCm
- HIP [with ``hipcc`` and ``hcc`` as backend]
- rocFFT
- rocPRIM
- rocThrust
- hipCUB
- roctracer-dev

.. note::

    When ``ENABLE_GPU=on``, HOOMD-blue will default to CUDA. Set ``HOOMD_GPU_PLATFORM=HIP`` to
    choose HIP.

**For threaded parallelism on the CPU** (required when ``ENABLE_TBB=on``):

- **Intel Threading Building Blocks**

**For runtime code generation** (required when ``ENABLE_LLVM=on``):

- **LLVM**
- **libclang-cpp**

**To build the documentation:**

- **sphinx**
- **sphinx-copybutton**
- **furo**
- **nbsphinx**
- **ipython**

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_:

.. code-block:: bash

   git clone --recursive git@github.com:glotzerlab/hoomd-blue.git

Release tarballs are also available as `GitHub release`_ assets.

.. seealso::

    See the `git book`_ to learn how to work with Git repositories.

.. important::

    **HOOMD-blue** uses Git submodules. Clone with the ``--recursive`` to clone the submodules.

    Execute ``git submodule update --init`` to fetch the submodules each time you switch branches
    and the submodules show as modified.

.. _GitHub release: https://github.com/glotzerlab/hoomd-blue/releases
.. _git book: https://git-scm.com/book
.. _Git: https://git-scm.com/

.. _Configure:

Configure
---------

Use CMake_ to configure the **HOOMD-blue** build directory:

.. code-block:: bash

    cd {{ path/to/hoomd-blue/repository }}

.. code-block:: bash

    cmake -B build -S . -GNinja

Pass ``-D<option-name>=<value>`` to ``cmake`` to set options on the command line.

Options that find libraries and executables take effect only on a clean invocation of CMake. To set
these options, first remove ``CMakeCache.txt`` from the build directory and then run ``cmake`` with
these options on the command line.

- ``Python_EXECUTABLE`` - Specify which ``python`` to build against. Example: ``/usr/bin/python3``.

  - Default: ``python3.x`` found by `CMake's FindPython
    <https://cmake.org/cmake/help/latest/module/FindPython.html>`__.

- ``CMAKE_CUDA_COMPILER`` - Specify which ``nvcc`` or ``hipcc`` to build with.

  - Default: location of ``nvcc`` detected on ``$PATH``.

- ``MPI_HOME`` (env var) - Specify the location where MPI is installed.

  - Default: location of ``mpicc`` detected on the ``$PATH``.

- ``<package-name>_ROOT`` - Specify the location of a package.

  - Default: Found on the `CMake`_ search path.

Other option changes take effect at any time:

- ``BUILD_HPMC`` - When enabled, build the ``hoomd.hpmc`` module (default: ``on``).
- ``BUILD_MD`` - When enabled, build the ``hoomd.md`` module (default: ``on``).
- ``BUILD_METAL`` - When enabled, build the ``hoomd.metal`` module (default: ``on``).
- ``BUILD_MPCD`` - When enabled, build the ``hoomd.mpcd`` module. ``hoomd.md`` must also be built.
  (default: same as ``BUILD_MD``).
- ``BUILD_TESTING`` - When enabled, build unit tests (default: ``on``).
- ``CMAKE_BUILD_TYPE`` - Sets the build type (case sensitive) Options:

  - ``Debug`` - Compiles debug information into the library and executables. Enables asserts to
    check for programming mistakes. **HOOMD-blue** will run slow when compiled in ``Debug`` mode,
    but problems are easier to identify.
  - ``RelWithDebInfo`` - Compiles with optimizations and debug symbols.
  - ``Release`` - (default) All compiler optimizations are enabled and asserts are removed.
    Recommended for production builds.

- ``CMAKE_INSTALL_PREFIX`` - Directory to install **HOOMD-blue**. Defaults to the root path of the
  found Python executable.
- ``ENABLE_LLVM`` - Enable run time code generation with LLVM.
- ``ENABLE_GPU`` - When enabled, compiled GPU accelerated computations (default: ``off``).
- ``HOOMD_GPU_PLATFORM`` - Choose either ``CUDA`` or ``HIP`` as a GPU backend (default: ``CUDA``).
- ``HOOMD_SHORTREAL_SIZE`` - Size in bits of the ``ShortReal`` type (default: ``32``).

  - When set to ``32``, perform force computations, overlap checks, and other local calculations
    in single precision.
  - When set to ``64``, perform **all** calculations in double precision.

- ``HOOMD_LONGREAL_SIZE`` - Size in bits of the ``LongReal`` type (default: ``64``).

  - When set to ``64``, store particle coordinates, sum quantities, and perform integration in
    double precision.
  - When set to ``32``, store particle coordinates, sum quantities, and perform integration in
    single precision. **NOT RECOMMENDED**, HOOMD-blue fails validation tests when
    ``HOOMD_LONGREAL_SIZE == HOOMD_SHORTREAL_SIZE == 32``.

- ``ENABLE_MPI`` - Enable multi-processor/GPU simulations using MPI.

  - When set to ``on``, multi-processor/multi-GPU simulations are supported.
  - When set to ``off`` (the default), always run in single-processor/single-GPU mode.

- ``ENABLE_TBB`` - Enable support for Intel's Threading Building Blocks (TBB).

  - When set to ``on``, **HOOMD-blue** will use TBB to speed up calculations in some classes on
    multiple CPU cores.

- ``PYTHON_SITE_INSTALL_DIR`` - Directory to install ``hoomd`` to relative to
  ``CMAKE_INSTALL_PREFIX``. Defaults to the ``site-packages`` directory used by the found Python
  executable.

These options control CUDA compilation via ``nvcc``:

- ``CUDA_ARCH_LIST`` - A semicolon-separated list of GPU architectures to compile.

.. tip::

    Pass the following options to CMake_ to optimize the build for your processor:
    ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native``

.. _CMake: https://cmake.org/
.. _Ninja: https://ninja-build.org/

.. _Build the package:

Build the package
-----------------

After configuring, build **HOOMD-blue** with:

.. code-block:: bash

    cd build

.. code-block:: bash

    ninja

The ``build`` directory now contains a fully functional **HOOMD-blue** package.
Execute ``ninja`` again any time you modify the code, test scripts, or CMake scripts.

.. tip::

    ``ninja`` will automatically execute ``cmake`` as needed. You do **NOT** need to execute
    ``cmake`` yourself every time you build **HOOMD-blue**.

Run tests
---------

Use `pytest`_ to execute unit tests:

.. code-block:: bash

   python3 -m pytest hoomd

.. _pytest: https://docs.pytest.org/

.. _Install the package:

Install the package
-------------------

Execute:

.. code-block:: bash

    ninja install

to install **HOOMD-blue** into your Python environment.

.. warning::

    This will *overwrite* any **HOOMD-blue** that you may have installed by other means.

To use the compiled **HOOMD-blue** without modifying your environment, set ``PYTHONPATH``::

    export PYTHONPATH={{ path/to/hoomd-blue/repository/build }}

.. _Build the documentation:

Build the documentation
-----------------------

Run `Sphinx`_ to build HTML documentation:

.. code-block:: bash

    sphinx-build -b html sphinx-doc html

Open the file :file:`html/index.html` in your web browser to view the documentation.

.. tip::

    Add the sphinx options ``-a -n -W -T --keep-going`` to produce docs with consistent links in
    the side panel and provide more useful error messages.

.. _Sphinx: https://www.sphinx-doc.org/
