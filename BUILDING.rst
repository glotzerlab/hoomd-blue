.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Building from source
====================

To build the **HOOMD-blue** Python package from source:

1. `Install prerequisites`_::

   $ <package-manager> install cmake eigen git python numpy pybind11

2. `Obtain the source`_::

   $ git clone --recursive https://github.com/glotzerlab/hoomd-blue

3. `Configure`_::

   $ cmake -B build/hoomd -S hoomd-blue

4. `Build the package`_::

   $ cmake --build build/hoomd

5. `Install the package`_ (optional)::

   $ cmake --install build/hoomd

To build the documentation from source (optional):

1. `Install prerequisites`_::

   $ <package-manager> install sphinx sphinx-copybutton furo nbsphinx ipython

.. note::

   ``nbsphinx`` requires ``pandoc>=1.12.1``, which you may need to install separately.

2. `Build the documentation`_::

   $ sphinx-build -b html hoomd-blue/sphinx-doc build/hoomd-documentation

The sections below provide details on each of these steps.

.. _Install prerequisites:

Install prerequisites
---------------------

**HOOMD-blue** requires a number of tools and libraries to build. The options ``ENABLE_MPI``,
``ENABLE_GPU``, ``ENABLE_TBB``, and ``ENABLE_LLVM`` each require additional libraries when enabled.

.. note::

    This documentation is generic. Replace ``<package-manager>`` with your package or module
    manager. You may need to adjust package names and/or install additional packages, such as
    ``-dev`` packages that provide headers needed to build hoomd.

.. tip::

    Create a `virtual environment`_, one place where you can install dependencies and
    **HOOMD-blue**::

        $ python3 -m venv hoomd-venv

    You will need to activate your environment before configuring **HOOMD-blue**::

        $ source hoomd-venv/bin/activate

.. note::

    Some package managers (such as *pip*) and many clusters are missing some or all of **pybind11**,
    **eigen**, and **cereal**. ``install-prereq-headers.py`` will install these packages into your
    virtual environment::

    $ python3 hoomd-blue/install-prereq-headers.py

**General requirements:**

- C++17 capable compiler (tested with ``gcc`` 9 - 14 and ``clang`` 10 - 18)
- Python >= 3.9
- NumPy >= 1.19
- pybind11 >= 2.12
- Eigen >= 3.2
- CMake >= 3.15

**For MPI parallel execution** (required when ``ENABLE_MPI=on``):

- MPI (tested with OpenMPI)
- cereal >= 1.1

**For GPU execution** (required when ``ENABLE_GPU=on``):

- NVIDIA CUDA Toolkit >= 9.0

  *OR*

- AMD ROCm >= 3.5.0 with additional dependencies:

  - HIP [with ``hipcc`` and ``hcc`` as backend]
  - rocFFT
  - rocPRIM
  - rocThrust
  - hipCUB, included for NVIDIA GPU targets, but required as an
    external dependency when building for AMD GPUs
  - roctracer-dev
  - Linux kernel >= 3.5.0
  - CMake >= 3.21

  For **HOOMD-blue** on AMD GPUs, the following limitations currently apply.

  1. Certain kernels trigger an `unknown HSA error <https://github.com/ROCm-Developer-Tools/HIP/issues/1662>`_.
  2. Multi-GPU execution via unified memory is not available.

.. note::

    When ``ENABLE_GPU=on``, HOOMD-blue will default to CUDA. Set ``HOOMD_GPU_PLATFORM=HIP`` to
    choose HIP.

**For threaded parallelism on the CPU** (required when ``ENABLE_TBB=on``):

- Intel Threading Building Blocks >= 4.3

**For runtime code generation** (required when ``ENABLE_LLVM=on``):

- LLVM >= 10.0
- libclang-cpp >= 10.0

**To build the documentation:**

- sphinx
- sphinx-copybutton
- furo
- nbsphinx
- ipython

.. _virtual environment: https://docs.python.org/3/library/venv.html

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_::

   $ git clone --recursive https://github.com/glotzerlab/hoomd-blue

Release tarballs are also available as `GitHub release`_ assets: `Download hoomd-4.7.0.tar.gz`_.

.. seealso::

    See the `git book`_ to learn how to work with Git repositories.

.. warning::

    **HOOMD-blue** uses Git submodules. Clone with the ``--recursive`` to clone the submodules.

    Execute ``git submodule update --init`` to fetch the submodules each time you switch branches
    and the submodules show as modified.

.. _Download hoomd-4.7.0.tar.gz: https://github.com/glotzerlab/hoomd-blue/releases/download/v4.7.0/hoomd-4.7.0.tar.gz
.. _GitHub release: https://github.com/glotzerlab/hoomd-blue/releases
.. _git book: https://git-scm.com/book
.. _Git: https://git-scm.com/

.. _Configure:

Configure
---------

Use CMake_ to configure a **HOOMD-blue** build in the given directory. Pass
``-D<option-name>=<value>`` to ``cmake`` to set options on the command line. When modifying code,
you only need to repeat the build step to update your build - it will automatically reconfigure
as needed.

.. tip::

    Use Ninja_ to perform incremental builds in less time::

        $ cmake -B build/hoomd -S hoomd-blue -GNinja

.. tip::

    Place your build directory in ``/tmp`` or ``/scratch`` for faster builds. CMake_ performs
    out-of-source builds, so the build directory can be anywhere on the filesystem.

.. tip::

    Pass the following options to ``cmake`` to optimize the build for your processor:
    ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native``.

.. important::

    When using a virtual environment, activate the environment and set the cmake prefix path
    before running CMake_: ``$ export CMAKE_PREFIX_PATH=<path-to-environment>``.

**HOOMD-blue**'s cmake configuration accepts a number of options.

Options that find libraries and executables only take effect on a clean invocation of CMake. To set
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

.. _CMake: https://cmake.org/
.. _Ninja: https://ninja-build.org/

.. _Build the package:

Build the package
-----------------

The command ``cmake --build build/hoomd`` will build the **HOOMD-blue** Python package in the given
build directory. After the build completes, the build directory will contain a functioning Python
package.

.. _Install the package:

Install the package
-------------------

The command ``cmake --install build/hoomd`` installs the given **HOOMD-blue** build to
``${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}``. CMake autodetects these paths, but you can
set them manually in CMake.

.. _Build the documentation:

Build the documentation
-----------------------

Run `Sphinx`_ to build the documentation with the command
``sphinx-build -b html hoomd-blue/sphinx-doc build/hoomd-documentation``. Open the file
:file:`build/hoomd-documentation/index.html` in your web browser to view the documentation.

.. tip::

    When iteratively modifying the documentation, the sphinx options ``-a -n -W -T --keep-going``
    are helpful to produce docs with consistent links in the side panel and to see more useful error
    messages::

        $ sphinx-build -a -n -W -T --keep-going -b html \
            hoomd-blue/sphinx-doc build/hoomd-documentation

.. _Sphinx: https://www.sphinx-doc.org/
