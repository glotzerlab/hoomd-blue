.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

==========
HOOMD-blue
==========

.. only:: html

    |Citing-HOOMD|
    |conda-forge|
    |conda-forge-Downloads|
    |GitHub Actions|
    |Contributors|
    |License|


    .. |Citing-HOOMD| image:: https://img.shields.io/badge/cite-hoomd-blue.svg
        :target: https://hoomd-blue.readthedocs.io/en/latest/citing.html
    .. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |conda-forge-Downloads| image:: https://img.shields.io/conda/dn/conda-forge/hoomd.svg?style=flat
        :target: https://anaconda.org/conda-forge/hoomd
    .. |GitHub Actions| image:: https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml/badge.svg
        :target: https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml
    .. |Contributors| image:: https://img.shields.io/github/contributors-anon/glotzerlab/hoomd-blue.svg?style=flat
        :target: https://hoomd-blue.readthedocs.io/en/latest/credits.html
    .. |License| image:: https://img.shields.io/badge/license-BSD--3--Clause-green.svg
        :target: https://hoomd-blue.readthedocs.io/en/latest/license.html

**HOOMD-blue** is a Python package that runs simulations of particle systems on CPUs and GPUs. It
performs hard particle Monte Carlo simulations of a variety of shape classes and molecular dynamics
simulations of particles with a range of pair, bond, angle, and other potentials. Many features are
targeted at the soft matter research community, though the code is general and capable of many
types of particle simulations.

Resources
=========

- `GitHub Repository <https://github.com/glotzerlab/hoomd-blue>`_:
  Source code and issue tracker.
- :doc:`Citing HOOMD-blue </citing>`:
  How to cite the code.
- :doc:`Installation guide </installation>`:
  Instructions for installing **HOOMD-blue** binaries.
- :doc:`Compilation guide </building>`:
  Instructions for compiling **HOOMD-blue**.
- `hoomd-users mailing list <https://groups.google.com/d/forum/hoomd-users>`_:
  Send messages to the **HOOMD-blue** user community.
- `HOOMD-blue website <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_:
  Additional information and publications.
- `HOOMD-blue benchmark scripts <https://github.com/glotzerlab/hoomd-benchmarks>`_:
  Scripts to evaluate the performance of HOOMD-blue simulations.

Related tools
=============

- `freud <https://freud.readthedocs.io/>`_:
  Analyze HOOMD-blue simulation results with the **freud** Python library.
- `signac <https://signac.io/>`_:
  Manage your workflow with **signac**.
- `Molecular Simulation Design Framework (MoSDeF)`_ tools:

  - `mbuild`_: Assemble reusable components into complex molecular systems.
  - `foyer`_: perform atom-typing and define classical molecular modeling force fields.

.. _Molecular Simulation Design Framework (MoSDeF): https://mosdef.org/
.. _mbuild: https://mbuild.mosdef.org/
.. _foyer: https://foyer.mosdef.org/

Example scripts
===============

These examples demonstrate some of the Python API.

Hard particle Monte Carlo:

.. code:: python

    import hoomd

    mc = hoomd.hpmc.integrate.ConvexPolyhedron()
    mc.shape['octahedron'] = dict(vertices=[
        (-0.5, 0, 0),
        (0.5, 0, 0),
        (0, -0.5, 0),
        (0, 0.5, 0),
        (0, 0, -0.5),
        (0, 0, 0.5),
    ])

    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=20)
    sim.operations.integrator = mc
    # See HOOMD tutorial for how to construct an initial configuration 'init.gsd'
    sim.create_state_from_gsd(filename='init.gsd')

    sim.run(1e5)

Molecular dynamics:

.. code:: python

    import hoomd

    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2.5

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.NVT(kT=1.5, filter=hoomd.filter.All(), tau=1.0)
    integrator.methods.append(nvt)

    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu)
    sim.operations.integrator = integrator
    # See HOOMD tutorial for how to construct an initial configuration 'init.gsd'
    sim.create_state_from_gsd(filename='init.gsd')
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    sim.run(1e5)

.. toctree::
    :maxdepth: 1
    :caption: Getting started

    features
    installation
    building
    migrating
    changelog
    citing

.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    tutorial/00-Introducing-HOOMD-blue/00-index
    tutorial/01-Introducing-Molecular-Dynamics/00-index
    tutorial/02-Logging/00-index
    tutorial/03-Parallel-Simulations-With-MPI/00-index
    tutorial/04-Custom-Actions-In-Python/00-index
    tutorial/05-Organizing-and-Executing-Simulations/00-index

.. toctree::
    :maxdepth: 1
    :caption: How to guides

    howto/molecular

.. toctree::
   :maxdepth: 3
   :caption: Python API

   package-hoomd
   package-hpmc
   package-md

.. toctree::
    :maxdepth: 1
    :caption: Developer guide

    contributing
    style
    testing
    components

.. toctree::
   :maxdepth: 1
   :caption: Reference

   notation
   units
   deprecated
   license
   credits
   indices
