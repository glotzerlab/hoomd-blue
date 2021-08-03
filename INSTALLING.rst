Installing binaries
===================

**HOOMD-blue** binaries are available in the `glotzerlab-software <https://glotzerlab-software.readthedocs.io>`_
`Docker <https://hub.docker.com/>`_/`Singularity <https://www.sylabs.io/>`_ images and for Linux and macOS via the
`hoomd package on conda-forge <https://anaconda.org/conda-forge/hoomd>`_.

Singularity / Docker images
---------------------------

See the `glotzerlab-software documentation <https://glotzerlab-software.readthedocs.io/>`_ for container usage
information and cluster specific instructions.


Conda package
---------------------

**HOOMD-blue** is available on `conda-forge <https://conda-forge.org>`_. To
install, first download and install `miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_. Then install ``hoomd``
from the ``conda-forge`` channel::

    $ conda install -c conda-forge hoomd

Compiling from source
=====================

Prerequisites
-------------

Compiling **HOOMD-blue** requires a number of software packages and libraries.

- Required:

  - Python >= 3.5
  - NumPy >= 1.7
  - CMake >= 2.8.10.1
  - C++11 capable compiler (tested with ``gcc`` 4.8, 5.4, 5.5, 6.4, 7,
    8, 9, ``clang`` 5, 6, 7, 8)

- Optional:

  - Git >= 1.7.0
  - NVIDIA CUDA Toolkit >= 9.0
  - Intel Threading Building Blocks >= 4.3
  - MPI (tested with OpenMPI, MVAPICH)
  - LLVM >= 5.0

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
and install the prerequisite libraries and tools. On Linux or macOS, run::

    conda install -c conda-forge sphinx git openmpi numpy cmake

After configuring, check the CMake configuration to ensure that it finds Python,
NumPy, and MPI from within the conda installation. If any of these library or
include files reference directories other than your conda environment, you will
need to set the appropriate setting for ``PYTHON_EXECUTABLE``, etc.

.. warning::

    On macOS, installing gcc with conda is not sufficient to build
    **HOOMD-blue**. Update Xcode to the latest version using the Mac App
    Store.

.. _compile-hoomd:

Compile HOOMD-blue
------------------

Download source releases directly from the web:
https://glotzerlab.engin.umich.edu/Downloads/hoomd

.. code-block:: bash

   $ curl -O https://glotzerlab.engin.umich.edu/Downloads/hoomd/hoomd-v2.9.7.tar.gz

Or, clone using Git:

.. code-block:: bash

   $ git clone --recursive https://github.com/glotzerlab/hoomd-blue

**HOOMD-blue** uses Git submodules. Either clone with the ``--recursive``
option, or execute ``git submodule update --init`` to fetch the submodules.

.. note::

    When using a shared (read-only) Python installation, such as a module on a
    cluster, create a `virtual environment
    <https://docs.python.org/3/library/venv.html>`_ where you can install
    **HOOMD-blue**::

        python3 -m venv /path/to/new/virtual/environment --system-site-packages

    Activate the environment before configuring and before executing
    **HOOMD-blue** scripts::

        source /path/to/new/virtual/environment/bin/activate

Configure::

    $ cd hoomd-blue
    $ mkdir build
    $ cd build
    $ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"`

By default, **HOOMD-blue** configures a *Release* optimized build type for a
generic CPU architecture and with no optional libraries. Specify::

    -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native

(or the appropriate option for your compiler) to enable optimizations specific
to your CPU. Specify ``-DENABLE_CUDA=ON`` to compile code for the GPU (requires
CUDA) and ``-DENABLE_MPI=ON`` to enable parallel simulations with MPI.
Configure a performance optimized build::

    $ cmake ../ -DCMAKE_INSTALL_PREFIX=`python3 -c "import site; print(site.getsitepackages()[0])"` -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON

See the build options section below for a full list of options.

Compile::

    $ make -j4

Test your build (requires a GPU to pass if **HOOMD-blue** was built with CUDA support)::

    $ ctest

.. attention::

    On a cluster, run ``ctest`` within a job on a GPU compute node.

To install **HOOMD-blue** into your Python environment, run::

    make install

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

  - Default: ``python3.X`` detected on ``$PATH``

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
