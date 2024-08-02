.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
    .. |GitHub Actions| image:: https://github.com/glotzerlab/hoomd-blue/actions/workflows/test.yml/badge.svg?branch=trunk-patch
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
- `HOOMD-blue discussion board <https://github.com/glotzerlab/hoomd-blue/discussions/>`_:
  Ask the **HOOMD-blue** user community for help.
- `HOOMD-blue website <https://glotzerlab.engin.umich.edu/hoomd-blue/>`_:
  Additional information and publications.
- `HOOMD-blue benchmark scripts <https://github.com/glotzerlab/hoomd-benchmarks>`_:
  Scripts to evaluate the performance of HOOMD-blue simulations.
- `HOOMD-blue validation tests <https://github.com/glotzerlab/hoomd-validation>`_:
  Scripts to validate that HOOMD-blue performs accurate simulations.

Related tools
=============

- `freud <https://freud.readthedocs.io/>`_:
  Analyze HOOMD-blue simulation results with the **freud** Python library.
- `signac <https://signac.io/>`_:
  Manage your workflow with **signac**.
- `Molecular Simulation Design Framework (MoSDeF)`_ tools:

  - `mbuild`_: Assemble reusable components into complex molecular systems.
  - `foyer`_: Perform atom-typing and define classical molecular modeling force fields.
  - `gmso`_: A flexible and mutable data structure for chemical topologies that writes HOOMD-blue formats.

.. _Molecular Simulation Design Framework (MoSDeF): https://mosdef.org/
.. _mbuild: https://mbuild.mosdef.org/
.. _foyer: https://foyer.mosdef.org/
.. _gmso: https://gmso.mosdef.org/

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
    # The tutorial describes how to construct an initial configuration 'init.gsd'.
    sim.create_state_from_gsd(filename='init.gsd')

    sim.run(1e5)

Molecular dynamics:

.. code:: python

    import hoomd

    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1, sigma=1)
    lj.r_cut[('A', 'A')] = 2.5

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    bussi = hoomd.md.methods.thermostats.Bussi(kT=1.5)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(), thermostat=bussi)
    integrator.methods.append(nvt)

    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu)
    sim.operations.integrator = integrator
    # The tutorial describes how to construct an initial configuration 'init.gsd'.
    sim.create_state_from_gsd(filename='init.gsd')
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    sim.run(1e5)

.. rubric::
    Contents

.. toctree::
    :maxdepth: 2
    :caption: Guides

    getting-started
    tutorials
    how-to

.. toctree::
   :maxdepth: 1
   :caption: Python API

   package-hoomd
   package-hpmc
   package-md
   package-mpcd

.. toctree::
   :maxdepth: 2
   :caption: Reference

   documentation
   changes
   developers

   open-source

   indices
