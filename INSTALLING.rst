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
channel:

.. code-block:: bash

    micromamba install hoomd=5.2.0

.. _conda-forge: https://conda-forge.org/docs/user/introduction.html

By default, micromamba auto-detects whether your system has a GPU and attempts to install the
appropriate package. Override this and force the GPU enabled package installation with:

.. code-block:: bash

    export CONDA_OVERRIDE_CUDA="12.6"
    micromamba install "hoomd=5.2.0=*gpu*" "cuda-version=12.6"

.. note::

    conda-forge_ may update to a new version of CUDA. If the above command results in an error,
    replace ``12.6`` with the version noted in micromamba's error message.

Similarly, you can force CPU-only package installation with:

.. code-block:: bash

    micromamba install "hoomd=5.2.0=*cpu*"

.. note::

    CUDA 11.8 compatible packages are also available. Replace "12.0" with "11.8" above when
    installing HOOMD-blue on systems with CUDA 11 compatible drivers.
