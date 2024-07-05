.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to model molecular systems
==============================

To model molecular systems using molecular dynamics in HOOMD-blue:

1. Define the bonds, angles, dihedrals, impropers, and constraints as needed to the
   system `State <hoomd.State>`.
2. Add the needed potentials and/or constraints to the integrator (see `md.bond`, `md.angle`,
   `md.dihedral`, `md.improper`, and `md.constrain`).

This code demonstrates bonds by modelling a Gaussian chain:

.. literalinclude:: molecular.py
    :language: python

Consider using a tool to build systems, such as the `Molecular Simulation Design Framework
(MoSDeF)`_. For example, `mBuild`_ can assemble reusable components into complex molecular systems,
and `foyer`_ can perform atom-typing and define classical molecular modeling force fields. The
`mosdef-workflows`_ `demonstrates how to use these tools with HOOMD-blue
<https://github.com/mosdef-hub/mosdef-workflows/blob/master/hoomd_lj/multistate_hoomd_lj.ipynb>`_

.. _Molecular Simulation Design Framework (MoSDeF): https://mosdef.org/
.. _mbuild: https://mbuild.mosdef.org/
.. _foyer: https://foyer.mosdef.org/
.. _mosdef-workflows: https://github.com/mosdef-hub/mosdef-workflows
