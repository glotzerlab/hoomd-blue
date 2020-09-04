==========
HOOMD-blue
==========

.. only:: html

    |Citing-HOOMD|
    |conda-forge|
    |conda-forge-Downloads|
    |Azure Pipelines|
    |Read-the-Docs|
    |Contributors|
    |License|


    .. |Citing-HOOMD| image:: https://img.shields.io/badge/cite-hoomd-blue.svg
        :target: https://glotzerlab.engin.umich.edu/hoomd-blue/citing.html
    .. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |conda-forge-Downloads| image:: https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |Azure Pipelines| image:: https://dev.azure.com/glotzerlab/hoomd-blue/_apis/build/status/test?branchName=master
        :target: https://dev.azure.com/glotzerlab/hoomd-blue/_build
    .. |Read-the-Docs| image:: https://img.shields.io/readthedocs/hoomd-blue/stable.svg
        :target: https://hoomd-blue.readthedocs.io/en/stable/?badge=stable
    .. |Contributors| image:: https://img.shields.io/github/contributors-anon/glotzerlab/hoomd-blue.svg?style=flat
        :target: https://hoomd-blue.readthedocs.io/en/stable/credits.html
    .. |License| image:: https://img.shields.io/badge/license-BSD--3--Clause-green.svg
        :target: https://github.com/glotzerlab/hoomd-blue/blob/maint/LICENSE

**HOOMD-blue** is a general purpose particle simulation toolkit. It performs
hard particle Monte Carlo simulations of a variety of shape classes, and
molecular dynamics simulations of particles with a range of pair, bond, angle,
and other potentials. **HOOMD-blue** runs fast on NVIDIA GPUs, and can scale
across thousands of nodes. For more information, see the `HOOMD-blue website
<https://glotzerlab.engin.umich.edu/hoomd-blue/>`_.

Resources
=========

- `GitHub Repository <https://github.com/glotzerlab/hoomd-blue>`_:
  Source code and issue tracker.
- :doc:`Installation Guide </installation>`:
  Instructions for installing and compiling **HOOMD-blue**.
- `hoomd-users Google Group <https://groups.google.com/d/forum/hoomd-users>`_:
  Ask questions to the **HOOMD-blue** community.
- `HOOMD-blue Tutorial <https://nbviewer.jupyter.org/github/glotzerlab/hoomd-examples/blob/master/index.ipynb>`_:
  Beginner's guide, code examples, and sample scripts.
- `HOOMD-blue website <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_:
  Additional information, benchmarks, and publications.

Job scripts
===========

HOOMD-blue job scripts are Python scripts. You can control system
initialization, run protocols, analyze simulation data, or develop complex
workflows all with Python code in your job.

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    migrating
    changelog
    command-line-options

.. toctree::
    :maxdepth: 2
    :caption: Concepts

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
   :caption: Stable Python packages

   package-hoomd
   package-hpmc
   package-md
   package-mpcd
   package-dem

.. toctree::
   :maxdepth: 3
   :caption: Additional Python packages

   package-jit
   package-metal

.. toctree::
    :maxdepth: 3
    :caption: Developer guide

    style
    testing

.. toctree::
   :maxdepth: 3
   :caption: Reference

   deprecated
   license
   credits
   indices
