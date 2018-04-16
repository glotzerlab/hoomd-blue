HOOMD-blue
++++++++++

Welcome to the reference documentation for HOOMD-blue!

The HOOMD examples and tutorials complement this documentation. `Read the HOOMD-blue tutorial online <http://nbviewer.jupyter.org/github/joaander/hoomd-examples/blob/master/index.ipynb>`_

On laptops/workstations, you can install `stable binaries <http://glotzerlab.engin.umich.edu/hoomd-blue/download.html>`_
with conda. If you haven't already, download and install `miniconda <http://conda.pydata.org/miniconda.html>`_. Then
add the glotzer channel and install HOOMD-blue::

    $ conda config --add channels glotzer
    $ conda install hoomd

If you have already installed hoomd in conda, you can upgrade to the latest version::

    $ conda update --all

On clusters, compile HOOMD from source so that you are using the right MPI version to take advantage
of the high performance network. Your cluster may also require a specific version of CUDA.

.. toctree::
    :maxdepth: 2
    :caption: Concepts

    compiling
    command-line-options
    units
    box
    aniso
    nlist
    mpi
    autotuner
    restartable-jobs
    varperiod
    developer

.. toctree::
   :maxdepth: 3
   :caption: Stable python packages

   package-hoomd
   package-hpmc
   package-md
   package-mpcd
   package-dem

.. toctree::
   :maxdepth: 3
   :caption: Additional python packages

   package-cgcmm
   package-deprecated
   package-jit
   package-metal

.. toctree::
   :maxdepth: 3
   :caption: Reference

   license
   credits
   indices
