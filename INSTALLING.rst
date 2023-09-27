.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Installing binaries
===================

**HOOMD-blue** binaries are available in the glotzerlab-software_ Docker_/Singularity_ images and in
packages on conda-forge_

.. _glotzerlab-software: https://glotzerlab-software.readthedocs.io
.. _Docker: https://hub.docker.com/
.. _Singularity: https://www.sylabs.io/
.. _conda-forge: https://conda-forge.org/docs/user/introduction.html

Singularity / Docker images
---------------------------

See the glotzerlab-software_ documentation for instructions to install and use the containers on
supported HPC clusters.

Conda package
-------------

**HOOMD-blue** is available on conda-forge_ on the *linux-64*, *osx-64*, and *osx-arm64* platforms.
Install the ``hoomd`` package from the conda-forge_ channel into a conda environment::

    $ conda install hoomd=4.2.0

``conda`` auto-detects whether your system has a GPU and attempts to install the appropriate
package. Override this and force the GPU enabled package installation with::

    $ export CONDA_OVERRIDE_CUDA="12.0"
    $ conda install "hoomd=4.2.0=*gpu*" "cuda-version=12.0"

Similarly, you can force CPU only package installation with::

    $ conda install "hoomd=4.2.0=*cpu*"

.. note::

    CUDA 11.2 compatible packages are also available. Replace "12.0" with "11.2" above when
    installing HOOMD-blue on systems with CUDA 11 compatible drivers.

.. note::

    To use :ref:`run time compilation` on **macOS**, install the ``compilers`` package::

        $ conda install compilers

    Without this package you will get *file not found* errors when HOOMD-blue performs the run time
    compilation.

.. tip::

    Use mambaforge_, miniforge_ or miniconda_ instead of the full Anaconda distribution to avoid
    package conflicts with conda-forge_ packages. When using miniconda_, follow the instructions
    provided in the conda-forge_ documentation to configure the channel selection so that all
    packages are installed from the conda-forge_ channel.

.. _mambaforge: https://github.com/conda-forge/miniforge
.. _miniforge: https://github.com/conda-forge/miniforge
.. _miniconda: http://conda.pydata.org/miniconda.html
