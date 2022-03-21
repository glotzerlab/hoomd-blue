.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

    $ conda install -c conda-forge hoomd

Recent versions of ``conda`` auto-detect whether your system has a GPU and installs the appropriate
package. Override this and force GPU package installation with::

    $ conda install -c conda-forge hoomd=*=*gpu*

.. tip::

    Use miniforge_ or miniconda_ instead of the full Anaconda distribution to avoid package
    conflicts with conda-forge_ packages.

.. _miniforge: https://github.com/conda-forge/miniforge
.. _miniconda: http://conda.pydata.org/miniconda.html
