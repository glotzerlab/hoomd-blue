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

.. tab:: Pixi

    .. code-block:: bash

        pixi add hoomd=5.4.0

.. tab:: Micromamba

    .. code-block:: bash

        micromamba install hoomd=5.4.0

.. tab:: Mamba

    .. code-block:: bash

        mamba install hoomd=5.4.0

.. _conda-forge: https://conda-forge.org/docs/user/introduction.html

By default, micromamba auto-detects whether your system has a GPU and attempts to install the
appropriate package. Override this and force the GPU enabled package installation with:

.. tab:: Pixi

    Add:

    .. code-block:: toml

        [system-requirements]
        cuda = "12.9"

     See `Using CUDA in Pixi`_ for more details. Then run:

    .. code-block:: bash

        export CONDA_OVERRIDE_CUDA="12.9"
        pixi add "hoomd=5.4.0=*gpu*"

.. tab:: Micromamba

    .. code-block:: bash

        export CONDA_OVERRIDE_CUDA="12.9"
        micromamba install "hoomd=5.4.0=*gpu*" "cuda-version=12.9"

.. tab:: Mamba

    .. code-block:: bash

        export CONDA_OVERRIDE_CUDA="12.9"
        mamba install "hoomd=5.4.0=*gpu*" "cuda-version=12.9"

.. _Using CUDA in Pixi: https://pixi.sh/dev/workspace/system_requirements/#using-cuda-in-pixi

.. note::

    conda-forge_ may update to a new version of CUDA after these instructions are published.
    If the above command results in an error, replace ``12.9`` with the version noted in
    micromamba's error message.

Similarly, you can force CPU-only package installation with:

.. tab:: Pixi

    .. code-block:: bash

        pixi add "hoomd=5.4.0=*cpu*"

.. tab:: Micromamba

    .. code-block:: bash

        micromamba install "hoomd=5.4.0=*cpu*"

.. tab:: Mamba

    .. code-block:: bash

        mamba install "hoomd=5.4.0=*cpu*"
