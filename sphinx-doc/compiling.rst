Compiling HOOMD-blue
====================

Software Prerequisites
----------------------

HOOMD-blue requires a number of prerequisite software packages and libraries.

 * Required:
     * Git >= 1.7.0
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 2.8.0
     * C++ 11 capable compiler (tested with gcc 4.8, 4.9, 5.4, 6.4, 7.0, 8.0, clang 3.8, 5.0, 6.0)
 * Optional:
     * NVIDIA CUDA Toolkit >= 7.0
     * Intel Threaded Building Blocks >= 4.3
     * MPI (tested with OpenMPI, MVAPICH)
     * sqlite3
 * Useful developer tools
     * Doxygen  >= 1.8.5

Software prerequisites on clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most cluster administrators provide versions of Git, Python, NumPy, MPI, and CUDA as modules.
You will need to consult the documentation or ask the system administrators
for instructions to load the appropriate modules.

Prerequisites on workstations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On a workstation, use the system's package manager to install all of the prerequisites. Some Linux
distributions separate ``-dev`` and normal packages, you need the development packages to build hoomd.

Mac OS systems do not come with a package manager or the necessary prerequisites. Use
[macports](https://www.macports.org/) or [homebrew](https://brew.sh/) to install them. You will need to install
XCode (free) through the Mac app store to supply the C++ compiler.

Installing prerequisites with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. caution::

    *Not recommended.* Conda is very useful as a delivery platform for
    `stable binaries <http://glotzerlab.engin.umich.edu/hoomd-blue/download.html>`_, but there are many pitfalls when using
    it to provide development prerequisites.

Despite this warning: many users wish to use conda to those provide development
prerequisites. There are a few additional steps required to build hoomd against a conda software stack, as you must
ensure that all libraries (mpi, python, etc...) are linked from the conda environment. First, install miniconda.
Then, uninstall the hoomd binaries if you have them installed and install the prerequisite libraries and tools::

    # if using linux
    conda install sphinx git mpich2 numpy cmake pkg-config sqlite
    # if using mac
    conda install sphinx git numpy cmake pkg-config sqlite

Check the CMake configuration to ensure that it finds python, numpy, and MPI from within the conda installation.
If any of these library or include files reference directories other than your conda environment, you will need to
set the appropriate setting for ``PYTHON_EXECUTABLE``, etc...

.. note::

    The ``mpich2`` package is not available on Mac. Without it, HOOMD will build without MPI support.

.. warning::

    On Mac OS, installing gcc with conda is not sufficient to build HOOMD. Update XCode to the latest version using the
    Mac OS app store.

.. _compile-hoomd:

Compile HOOMD-blue
------------------

Set the environment variable SOFTWARE_ROOT to the location you wish to install HOOMD::

    $ export SOFTWARE_ROOT=/path/to/prefix

Clone the git repository to get the source::

    $ git clone --recursive https://bitbucket.org/glotzer/hoomd-blue

By default, the *maint* branch will be checked out. This branch includes all bug fixes since the last stable release.
HOOMD-blue uses submodules, you the ``--recursive`` option to clone instructs git to fetch all of the submodules.
When you later update this git repository with ``git pull``, run ``git submodule update`` update all of the submodules.

Configure::

    $ cd hoomd-blue
    $ mkdir build
    $ cd build
    $ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python

By default, HOOMD configures a *Release* optimized build type for a generic CPU architecture and with no optional
libraries. Specify ``-DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native`` (or whatever is the appropriate
option for your compiler) to enable optimizations specific to your CPU. Specify ``-DENABLE_CUDA=ON`` to compile code
for the GPU (requires CUDA) and ``-DENABLE_MPI=ON`` to enable parallel simulations with MPI. See the build options
section below for a full list of options::

    $ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native -DENABLE_CUDA=ON -DENABLE_MPI=ON

Compile::

    $ make -j4

Run::

    $ make test

to test your build. If you built with CUDA support, you need a GPU for all tests to pass.

.. attention::
    On a cluster, run ``make test`` within a job on a GPU compute node.

To install a stable version for general use, run::

    make install

Then set your PYTHONPATH so that python can find hoomd::

    export PYTHONPATH=$PYTHONPATH:${SOFTWARE_ROOT}/lib/python

Build options
-------------

Here is a list of all the build options that can be changed by CMake. To changes these settings, cd to your *build*
directory and run::

    $ ccmake .

After changing an option, press *c* to configure then press *g* to generate. The makefile/IDE project is now updated with
the newly selected options. Alternately, you can set these parameters on the command line with cmake::

    cmake $HOME/devel/hoomd -DENABLE_CUDA=on

Options that specify library versions only take effect on a clean invocation of cmake. To set these options, first
remove `CMakeCache.txt` and then run cmake and specify these options on the command line:

* **PYTHON_EXECUTABLE** - Specify which python to build against. Example: /usr/bin/python2.
    * Default: ``python3`` or ``python`` detected on ``$PATH``
* **CUDA_TOOLKIT_ROOT_DIR** - Specify the root direction of the CUDA installation.
    * Default: location of ``nvcc`` detected on ``$PATH``
* **MPI_HOME (env var)** - Specify the location where MPI is installed.
    * Default: location of ``mpicc`` detected on the ``$PATH``

Other option changes take effect at any time. These can be set from within `ccmake` or on the command line:

* **CMAKE_INSTALL_PREFIX** - Directory to install the hoomd python module. All files will be under
  ${CMAKE_INSTALL_PREFIX}/hoomd
* **BUILD_CGCMM** - Enables building the cgcmm component
* **BUILD_DEPRECATED** - Enables building the deprecated component
* **BUILD_HPMC** - Enables building the hpmc component.
* **BUILD_MD** - Enables building the md component
* **BUILD_METAL** - Enables building the metal component
* **BUILD_TESTING** - Enables the compilation of unit tests
* **CMAKE_BUILD_TYPE** - sets the build type (case sensitive)
    * **Debug** - Compiles debug information into the library and executables.
      Enables asserts to check for programming mistakes. HOOMD-blue will run
      slow when compiled in Debug mode, but problems are easier to
      identify.
    * **RelWithDebInfo** - Compiles with optimizations and debug symbols. Useful for profiling benchmarks.
    * **Release** - (default) All compiler optimizations are enabled and asserts are removed.
      Recommended for production builds: required for any benchmarking.
* **ENABLE_CUDA** - Enable compiling of the GPU accelerated computations using CUDA. Defaults *on* if the CUDA toolkit
  is found. Defaults *off* if the CUDA toolkit is not found.
* **ENABLE_DOXYGEN** - enables the generation of developer documentation (Defaults *off*)
* **SINGLE_PRECISION** - Controls precision
    - When set to **ON**, all calculations are performed in single precision.
    - When set to **OFF**, all calculations are performed in double precision.
* **ENABLE_HPMC_MIXED_PRECISION** - Controls mixed precision in the hpmc component. When on, single precision is forced
      in expensive shape overlap checks.
* **ENABLE_MPI** - Enable multi-processor/GPU simulations using MPI
    - When set to **ON** (default if any MPI library is found automatically by CMake), multi-GPU simulations are supported
    - When set to **OFF**, HOOMD always runs in single-GPU mode
* **ENABLE_MPI_CUDA** - Enable CUDA-aware MPI library support
    - Requires a MPI library with CUDA support to be installed
    - When set to **ON** (default if a CUDA-aware MPI library is detected), HOOMD-blue will make use of the capability of the MPI library to accelerate CUDA-buffer transfers
    - When set to **OFF**, standard MPI calls will be used
    - *Warning:* Manually setting this feature to ON when the MPI library does not support CUDA may
      result in a crash of HOOMD-blue
* **ENABLE_TBB** - Enable support for Intel's Threading Building Blocks (TBB)
    - Requires TBB to be installed
    - When set to **ON**, HOOMD will use TBB to speed up calculations in some classes on multiple CPU cores
* **UPDATE_SUBMODULES** - When ON (the default), execute ``git submodule update --init`` whenever cmake runs.
* **COPY_HEADERS** - When ON (OFF is default), copy header files into the build directory to make it a valid plugin build source

These options control CUDA compilation:

* **CUDA_ARCH_LIST** - A semicolon separated list of GPU architecture to compile in.
* **NVCC_FLAGS** - Allows additional flags to be passed to nvcc.
