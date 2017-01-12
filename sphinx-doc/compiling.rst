Compiling HOOMD-blue
====================

Software Prerequisites
----------------------

HOOMD-blue requires a number of prerequisite software packages and libraries.

 * Required:
     * Python >= 2.7
     * numpy >= 1.7
     * CMake >= 2.8.0
     * C++ 11 capable compiler (tested with gcc >= 4.8.5, clang 3.5)
 * Optional:
     * NVIDIA CUDA Toolkit >= 7.0
     * MPI (tested with OpenMPI, MVAPICH)
     * sqlite3
 * Useful developer tools
     * Git >= 1.7.0
     * Doxygen  >= 1.8.5

Software prerequisites on clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most cluster administrators provide versions of python, numpy, mpi, and cuda as modules.
Here are the module commands necessary to load prerequisites at national
supercomputers. Each code block also specifies a recommended install location ``${SOFTWARE_ROOT}`` where hoomd can
be loaded on the compute nodes with minimal file system impact. On many clusters, administrators will block your account
without warning if you launch hoomd from ``$HOME``.

**NCSA Blue waters**::

    module switch PrgEnv-cray PrgEnv-gnu
    module load cudatoolkit
    module load bwpy
    export SOFTWARE_ROOT=${HOME}/software
    export CPATH="${BWPY_DIR}/usr/include"
    export LIBRARY_PATH="${BWPY_DIR}/lib64:${BWPY_DIR}/usr/lib64"
    export LD_LIBRARY_PATH="${BWPY_DIR}/lib64:${BWPY_DIR}/usr/lib64:${LD_LIBRARY_PATH}"

Put these commands in your ``~/.modules`` file to have a working environment available every time you log in.

You can select python2 or python3::

    cmake /path/to/hoomd -DPYTHON_EXECUTABLE=`which python3` -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python

To run hoomd on blue waters, set ``PYTHONPATH``, and execute aprun::

    PYTHONPATH=$PYTHONPATH:${SOFTWARE_ROOT}/lib/python
    aprun <aprun parameters> python3 script.py

**OLCF Titan**::

    module unload PrgEnv-pgi
    module load PrgEnv-gnu
    module load cmake
    module load git
    module load cudatoolkit
    module load python/3.5.1
    module load python_numpy/1.9.2
    # need gcc first on the search path
    module unload gcc/4.9.0
    module load gcc/4.9.0
    export SOFTWARE_ROOT=${PROJWORK}/${your_project}/software/titan

For more information, see: https://www.olcf.ornl.gov/support/system-user-guides/titan-user-guide/

**OLCF Eos**::

    module unload PrgEnv-intel
    module load PrgEnv-gnu
    module load cmake
    module load git
    module load python/3.4.3
    module load python_numpy/1.9.2
    export SOFTWARE_ROOT=${PROJWORK}/${your_project}/software/eos
    # need gcc first on the search path
    module unload gcc/4.9.0
    module load gcc/4.9.0

    export CC="cc -dynamic"
    export CXX="CC -dynamic"

For more information, see: https://www.olcf.ornl.gov/support/system-user-guides/eos-user-guide/

**XSEDE SDSC Comet**::

    module purge
    module load python
    module load gnu
    module load mvapich2_ib
    module load gnutools
    module load scipy
    module load cmake
    module load cuda/7.0

    export CC=`which gcc`
    export CXX=`which g++`
    export SOFTWARE_ROOT=/oasis/projects/nsf/${your_project}/${USER}/software

.. note::
    CUDA libraries are only available on GPU nodes on Comet. To run on the CPU-only nodes, you must build hoomd
    with ENABLE_CUDA=off.

.. warning::
    Make sure to set CC and CXX. Without these, cmake will use /usr/bin/gcc and compilation will fail.

For more information, see: http://www.sdsc.edu/support/user_guides/comet.html

XSEDE TACC Stampede::

    module unload mvapich
    module load intel/15.0.2
    module load impi
    module load cuda/7.0
    module load cmake
    module load git
    module load python/2.7.9

    export CC=`which icc`
    export CXX=`which icpc`
    export SOFTWARE_ROOT=${WORK}/software

.. note::
    CUDA libraries are only available on GPU nodes on Stampede. To run on the CPU-only nodes, you must build hoomd
    with ENABLE_CUDA=off.

.. note::
    Make sure to set CC and CXX. Without these, cmake will use /usr/bin/gcc and compilation will fail.

.. warning:
    HOOMD-blue is no longer tested with the intel compiler. Numerous compiler bugs in Intel 2015 prevent it from
    building hoomd. HOOMD developers do not currently have access to stampede to generate new build instructions
    using gcc.

For more information, see: https://portal.tacc.utexas.edu/user-guides/stampede

**STFC DiRAC Cambridge Darwin and Wilkes**:

If you are running on Darwin and will not be using GPUs::

    . /etc/profile.d/modules.sh
    module purge
    module load default-impi
    module load cmake
    module load python/2.7.10

    export CC=`which gcc`
    export CXX=`which c++`
    export SOFTWARE_ROOT=/scratch/$USER/software

To build, include the following additional `cmake` options::

    -DPYTHON_EXECUTABLE=`which python` \
    -DENABLE_MPI=ON \
    -DMPIEXEC=`which mpirun`

If you are running on Wilkes, you will need to include CUDA support::

    . /etc/profile.d/modules.sh
    module purge
    module load default-impi
    module load cmake
    module load gcc/4.9.2
    module load python/2.7.5
    module load cuda

    export CC=`which gcc`
    export CXX=`which c++`
    export SOFTWARE_ROOT=/scratch/$USER/software

To build, include the following additional `cmake` options::

    -DPYTHON_EXECUTABLE=`which python` \
    -DHOOMD_PYTHON_LIBRARY=/usr/local/Cluster-Apps/python/2.7.5/lib64/libpython2.7.so \
    -DENABLE_MPI=ON \
    -DMPIEXEC=`which mpirun`

Note that the Darwin and Wilkes clusters have the same software environment
and shared filesystems, so you can build for Wilkes and use on Darwin.
However, as of March 2016, module incompatibilities necessitate older modules
and a quirk in the python installation requires explicitly setting the `libpython` location.

Installing prerequisites on a workstation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On your workstation, use your systems package manager to install all of the prerequisite libraries. Some linux
distributions separate ``-dev`` and normal packages, you need the development packages to build hoomd.

Installing prerequisites with conda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda is very useful as a delivery platform for `stable binaries <http://glotzerlab.engin.umich.edu/hoomd-blue/download.html>`_,
and we do recommend only using it for that purpose. However, many users wish to use conda to provide development
perquisites. There are a few additional steps required to build hoomd against a conda software stack, as you must
ensure that all libraries (mpi, python, etc...) are linked from the conda environment. First, install miniconda.
Then, uninstall the hoomd binaries if you have them installed and install the prerequisite libraries and tools::

    conda uninstall hoomd
    conda install sphinx git mpich2 numpy cmake pkg-config sqlite

Check the CMake configuration to ensure that it finds python, numpy, and MPI from within the conda installation.
If any of these library or include files reference directories other than your conda environment, you will need to
set the appropriate setting for ``PYTHON_EXECUTABLE``, etc...

.. _compile-hoomd:

Compile HOOMD-blue
------------------

Clone the git repository to get the source::

    $ git clone https://bitbucket.org/glotzer/hoomd-blue

By default, the *maint* branch will be checked out. This branch includes all bug fixes since the last stable release.

Compile::

    $ cd hoomd-blue
    $ mkdir build
    $ cd build
    $ cmake ../ -DCMAKE_INSTALL_PREFIX=${SOFTWARE_ROOT}/lib/python
    $ make -j20

Run::

    $ make test

to test your build.

.. attention::
    On a cluster, ``make test`` may need to be run within a job on a compute node.

To install a stable version for general use, run::

    make install
    export PYTHONPATH=$PYTHONPATH:${SOFTWARE_ROOT}/lib/python

To run out of your build directory::

    export PYTHONPATH=$PYTHONPATH:/path/to/hoomd/build

Compiling with MPI enabled
^^^^^^^^^^^^^^^^^^^^^^^^^^

System provided MPI:

If your cluster administrator provides an installation of MPI, you need to figure out if is in your
`$PATH`. If the command::

    $ which mpicc
    /usr/bin/mpicc

succeeds, you're all set. HOOMD-blue should detect your MPI compiler automatically.

If this is not the case, set the `MPI_HOME` environment variable to the location of the MPI installation::

    $ echo ${MPI_HOME}
    /home/software/rhel5/openmpi-1.4.2/gcc

Build hoomd:

Configure and build HOOMD-blue as normal (see :ref:`compile-hoomd`). During the cmake step, MPI should
be detected and enabled. For cuda-aware MPI, additionally supply the **ENABLE_MPI_CUDA=ON** option to cmake.

Build options
-------------

Here is a list of all the build options that can be changed by CMake. To changes these settings, cd to your *build*
directory and run::

    $ ccmake .

After changing an option, press *c* to configure then press *g* to generate. The makefile/IDE project is now updated with
the newly selected options. Alternately, you can set these parameters on the initial cmake invocation::

    cmake $HOME/devel/hoomd -DENABLE_CUDA=off

Options that specify library versions only take effect on a clean invocation of cmake. To set these options, first
remove `CMakeCache.txt` and then run cmake and specify these options on the command line:

* **PYTHON_EXECUTABLE** - Specify python to build against. Example: /usr/bin/python2

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
    * **Release** - All compiler optimizations are enabled and asserts are removed.
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
    - When set to **ON** (default if a CUDA-aware MPI library is detected), HOOMD-blue will make use of  the capability of the MPI library to accelerate CUDA-buffer transfers
    - When set to **OFF**, standard MPI calls will be used
    - *Warning:* Manually setting this feature to ON when the MPI library does not support CUDA may
      result in a crash of HOOMD-blue
* **UPDATE_SUBMODULES** - When ON (the default), execute ``git submodule update --init`` whenever cmake runs.
* **COPY_HEADERS** - When ON (OFF is default), copy header files into the build directory to make it a valid plugin build source

These options control CUDA compilation:

* **CUDA_ARCH_LIST** - A semicolon separated list of GPU architecture to compile in.
* **NVCC_FLAGS** - Allows additional flags to be passed to nvcc.
