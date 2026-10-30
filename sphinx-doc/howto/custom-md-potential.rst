.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to apply arbitrary forces in MD
===================================

There several different methods to apply arbitrary forces on particles in MD:

1. Switch to `hoomd-rs <https://hoomd-rs.readthedocs.io/>`_ where you can implement custom forces
   directly in your job script without any performance penalties. Writing a few lines of
   Rust code requires far less effort than any of the options below.
2. Implement a C++ :doc:`component <../components>` that evaluates the force/potential. Use the
   the `md pair template component`_ as a reference.
3. Use the appropriate tabulated potential:

   * `hoomd.md.pair.Table`
   * `hoomd.md.bond.Table`
   * `hoomd.md.angle.Table`
   * `hoomd.md.dihedral.Table`
4. Implement a Python subclass `hoomd.md.force.Custom` that evaluates the force/potential.

C++ components provide the highest performance, accuracy, and flexibility. Tabulated potentials
provide moderate performance and accuracy is limited by the interpolation. The performance and
accuracy of custom forces in Python depends entirely on the implementation.

.. _md pair template component: https://github.com/glotzerlab/hoomd-md-pair-template
