.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Installing binaries
===================

MPI parallel builds
-------------------

You must build **HOOMD-blue** from source to enable support for the native **MPI** and **CUDA**
libraries on your **HPC resource**. You can use the glotzerlab-software_ repository to manage such
builds as conda packages.

.. _glotzerlab-software: https://glotzerlab-software.readthedocs.io

Serial CPU and single GPU builds
--------------------------------

**HOOMD-blue** binaries for **serial CPU** and **single GPU** are available on conda-forge_ for the
*linux-64*, *osx-64*, and *osx-arm64* platforms. Install the ``hoomd`` package from the conda-forge_
channel into a conda environment::

    $ mamba install hoomd=4.7.0

.. _conda-forge: https://conda-forge.org/docs/user/introduction.html

``conda`` auto-detects whether your system has a GPU and attempts to install the appropriate
package. Override this and force the GPU enabled package installation with::

    $ export CONDA_OVERRIDE_CUDA="12.0"
    $ mamba install "hoomd=4.7.0=*gpu*" "cuda-version=12.0"

Similarly, you can force CPU only package installation with::

    $ mamba install "hoomd=4.7.0=*cpu*"

.. note::

    CUDA 11.8 compatible packages are also available. Replace "12.0" with "11.8" above when
    installing HOOMD-blue on systems with CUDA 11 compatible drivers.

.. note::

    Run time compilation is no longer available on conda-forge builds starting with HOOMD-blue
    4.7.0.

.. tip::

    Use miniforge_, miniconda_, or any other *minimal* conda environment provider instead of the
    full Anaconda distribution to avoid package conflicts with conda-forge_ packages. When using
    miniconda_, follow the instructions provided in the conda-forge_ documentation to configure the
    channel selection so that all packages are installed from the conda-forge_ channel.

.. _miniforge: https://github.com/conda-forge/miniforge
.. _miniconda: http://conda.pydata.org/miniconda.html
