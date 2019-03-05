Installing binaries
===================

**HOOMD-blue** binaries are available as containers (`Docker Hub
<https://hub.docker.com/r/glotzerlab/software>`_, `Singularity Hub
<https://singularity-hub.org/collections/1663>`_) and for Linux and macOS via
the `hoomd package on conda-forge <https://anaconda.org/conda-forge/hoomd>`_.

Using Singularity / Docker containers
-------------------------------------

Singularity::

    $ singularity pull --name "software.simg" shub://glotzerlab/software

Docker::

    $ docker pull glotzerlab/software

Installing with conda
---------------------

**HOOMD-blue** is available on `conda-forge <https://conda-forge.org>`_. To
install, first download and install `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_. Then install ``hoomd``
from the ``conda-forge`` channel::

    $ conda install -c conda-forge hoomd

If ``hoomd`` has already been installed, you can upgrade to the latest version::

    $ conda update hoomd

Compiling from source
=====================

Prerequisites
-------------

Compiling **HOOMD-blue** requires a number of software packages and libraries.

- Required:

  - Python >= 2.7
  - NumPy >= 1.7
  - CMake >= 2.8.0
  - C++11 capable compiler (tested with ``gcc`` 4.8, 4.9, 5.4, 6.4, 7.0,
    8.0, ``clang`` 5.0, 6.0)

- Optional:

  - Git >= 1.7.0
  - NVIDIA CUDA Toolkit >= 8.0
  - Intel Threading Building Blocks >= 4.3
  - MPI (tested with OpenMPI, MVAPICH)
  - LLVM >= 3.6

- Useful developer tools

  - Doxygen >= 1.8.5

Software prerequisites on clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most cluster administrators provide versions of Git, Python, NumPy, MPI, and
CUDA as modules. You will need to consult the documentation or ask the system
administrators for instructions to load the appropriate modules.

Prerequisites on workstations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a workstation, use the system's package manager to install all of the
prerequisites. Some Linux distributions separate ``-dev`` and normal packages,
you need the development packages to build **HOOMD-blue**.

On macOS systems, you can use `MacPorts <https://www.macports.org/>`_ or
`Homebrew <https://brew.sh/>`_ to install prerequisites. You will need to
install Xcode (free) through the Mac App Store to supply the C++ compiler.

Installing prerequisites with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. caution::

    *Using conda to provide build prerequisites is not recommended.* Conda is
    very useful as a delivery platform for `stable binaries
    <http://glotzerlab.engin.umich.edu/hoomd-blue/download.html>`_, but there
    are many pitfalls when using it to provide development prerequisites.

Despite this warning, many users wish to use conda to provide those development
prerequisites. There are a few additional steps required to build
**HOOMD-blue** against a conda software stack, as you must ensure that all
libraries (MPI, Python, etc.) are linked from the conda environment. First,
install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
Then, uninstall the ``hoomd`` package if it is installed,
and install the prerequisite libraries and tools. On linux, run::

    conda install sphinx git mpich2 numpy cmake pkg-config

On macOS::

    conda install sphinx git numpy cmake pkg-config

After configuring, check the CMake configuration to ensure that it finds Python,
NumPy, and MPI from within the conda installation. If any of these library or
include files reference directories other than your conda environment, you will
need to set the appropriate setting for ``PYTHON_EXECUTABLE``, etc.

.. note::

    The ``mpich2`` package is not available on macOS. Without it,
    **HOOMD-blue** will only build without MPI support.

.. warning::

    On macOS, installing gcc with conda is not sufficient to build
    **HOOMD-blue**. Update Xcode to the latest version using the Mac App
    Store.

.. _compile-hoomd:

Compile HOOMD-blue
------------------

Set the environment variable ``SOFTWARE_ROOT`` to the location you wish to
install **HOOMD-blue**::

    $ export SOFTWARE_ROOT=/path/to/prefix

Clone the Git repository to get the source::

    $ git clone --recursive https://github.com/glotzerlab/hoomd-blue.git

By default, the ``maint`` branch will be checked out. This branch includes all
bug fixes since the last stable release. **HOOMD-blue** uses submodules, using
the ``--recursive`` option to clone instructs Git to fetch all of the
submodules. Alternatively, call ``git submodule update --init`` in the
repository directory. When you update this Git repository with ``git pull``,
run ``git submodule update`` to update all of the submodules.

Configure::

    $ cd hoomd-blue
    $ mkdir build
    $ cd build
    $ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python

By default, **HOOMD-blue** configures a *Release* optimized build type for a
generic CPU architecture and with no optional libraries. Specify
``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native`` (or the
appropriate option for your compiler) to enable optimizations specific to your
CPU. Specify ``-DENABLE_CUDA=ON`` to compile code for the GPU (requires CUDA)
and ``-DENABLE_MPI=ON`` to enable parallel simulations with MPI. See the build
options section below for a full list of options::

    $ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON

Compile::

    $ make -j4

Test your build (requires a GPU to pass if **HOOMD-blue** was built with CUDA support)::

    $ make test

.. attention::

    On a cluster, run ``make test`` within a job on a GPU compute node.

To install a stable version for general use, run::

    make install

Then set your ``PYTHONPATH`` so that Python can find ``hoomd``::

    export PYTHONPATH=$PYTHONPATH:${SOFTWARE_ROOT}/lib/python

Build options
^^^^^^^^^^^^^

Here is a list of all the build options that can be changed by CMake. To
change these settings, navigate to the ``build`` directory and run::

    $ ccmake .

After changing an option, press ``c`` to configure, then press ``g`` to
generate. The ``Makefile`` is now updated with the newly selected
options. You can also set these parameters on the command line with
``cmake``::

    cmake $HOME/devel/hoomd -DENABLE_CUDA=ON

Options that specify library versions only take effect on a clean invocation of
CMake. To set these options, first remove ``CMakeCache.txt`` and then run CMake
and specify these options on the command line:

- ``PYTHON_EXECUTABLE`` - Specify which ``python`` to build against. Example: ``/usr/bin/python3``.

  - Default: ``python3`` or ``python`` detected on ``$PATH``

- ``CUDA_TOOLKIT_ROOT_DIR`` - Specify the root direction of the CUDA installation.

  - Default: location of ``nvcc`` detected on ``$PATH``

- ``MPI_HOME`` (env var) - Specify the location where MPI is installed.

  - Default: location of ``mpicc`` detected on the ``$PATH``


Other option changes take effect at any time. These can be set from within
``ccmake`` or on the command line:

- ``CMAKE_INSTALL_PREFIX`` - Directory to install the ``hoomd`` Python module.
  All files will be under ``${CMAKE_INSTALL_PREFIX}/hoomd``.
- ``BUILD_CGCMM`` - Enables building the ``hoomd.cgcmm`` module.
- ``BUILD_DEPRECATED`` - Enables building the ``hoomd.deprecated`` module.
- ``BUILD_HPMC`` - Enables building the ``hoomd.hpmc`` module.
- ``BUILD_MD`` - Enables building the ``hoomd.md`` module.
- ``BUILD_METAL`` - Enables building the ``hoomd.metal`` module.
- ``BUILD_TESTING`` - Enables the compilation of unit tests.
- ``CMAKE_BUILD_TYPE`` - Sets the build type (case sensitive) Options:

  - ``Debug`` - Compiles debug information into the library and executables.
    Enables asserts to check for programming mistakes. HOOMD-blue will run
    slow when compiled in Debug mode, but problems are easier to identify.
  - ``RelWithDebInfo`` - Compiles with optimizations and debug symbols.
    Useful for profiling benchmarks.
  - ``Release`` - (default) All compiler optimizations are enabled and
    asserts are removed. Recommended for production builds: required for any
    benchmarking.

- ``ENABLE_CUDA`` - Enable compiling of the GPU accelerated computations. Default: ``OFF``.
- ``ENABLE_DOXYGEN`` - Enables the generation of developer documentation
  Default: ``OFF``.
- ``SINGLE_PRECISION`` - Controls precision. Default: ``OFF``.

  - When set to ``ON``, all calculations are performed in single precision.
  - When set to ``OFF``, all calculations are performed in double precision.

- ``ENABLE_HPMC_MIXED_PRECISION`` - Controls mixed precision in the hpmc
  component. When on, single precision is forced in expensive shape overlap
  checks.
- ``ENABLE_MPI`` - Enable multi-processor/GPU simulations using MPI.

  - When set to ``ON``, multi-processor/multi-GPU simulations are supported.
  - When set to ``OFF`` (the default), always run in single-processor/single-GPU mode.

- ``ENABLE_MPI_CUDA`` - Enable CUDA-aware MPI library support.

  - Requires a MPI library with CUDA support to be installed.
  - When set to ``ON`` (default if a CUDA-aware MPI library is detected),
    **HOOMD-blue** will make use of the capability of the MPI library to
    accelerate CUDA-buffer transfers.
  - When set to ``OFF``, standard MPI calls will be used.
  - *Warning:* Manually setting this feature to ``ON`` when the MPI library
    does not support CUDA may cause **HOOMD-blue** to crash.

- ``ENABLE_TBB`` - Enable support for Intel's Threading Building Blocks (TBB).

  - Requires TBB to be installed.
  - When set to ``ON``, HOOMD will use TBB to speed up calculations in some
    classes on multiple CPU cores.

- ``UPDATE_SUBMODULES`` - When ``ON`` (the default), CMake will execute
  ``git submodule update --init`` whenever it runs.
- ``COPY_HEADERS`` - When ``ON`` (``OFF`` is default), copy header files into
  the build directory to make it a valid plugin build source.

These options control CUDA compilation:

- ``CUDA_ARCH_LIST`` - A semicolon-separated list of GPU architectures to
  compile in.
- ``NVCC_FLAGS`` - Allows additional flags to be passed to ``nvcc``.
