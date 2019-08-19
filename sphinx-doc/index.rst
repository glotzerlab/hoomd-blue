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

Here is a simple example:

.. code-block:: python

   import hoomd
   from hoomd import md
   hoomd.context.initialize()

   # Create a 10x10x10 simple cubic lattice of particles with type name A
   hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)

   # Specify Lennard-Jones interactions between particle pairs
   nl = md.nlist.cell()
   lj = md.pair.lj(r_cut=3.0, nlist=nl)
   lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)

   # Integrate at constant temperature
   md.integrate.mode_standard(dt=0.005)
   hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=1.2, seed=4)

   # Run for 10,000 time steps
   hoomd.run(10e3)

Save this script as ``lj.py`` and run it with ``python lj.py`` (or
``singularity exec software.simg python3 lj.py`` if using Singularity
containers).

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
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

   package-cgcmm
   package-deprecated
   package-jit
   package-metal

.. toctree::
   :maxdepth: 3
   :caption: Reference

   deprecated
   license
   credits
   indices
