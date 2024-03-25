.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to apply arbitrary forces in MD
===================================

There several different methods to apply arbitrary forces on particles in MD:

1. Implement a C++ :doc:`component <../components>` that evaluates the force/potential. Use one of
   the `example plugins`_ as a reference.
2. Use the appropriate tabulated potential:

   * `hoomd.md.pair.Table`
   * `hoomd.md.bond.Table`
   * `hoomd.md.angle.Table`
   * `hoomd.md.dihedral.Table`
3. Implement a Python subclass `hoomd.md.force.Custom` that evaluates the force/potential.

C++ components provide the highest performance, accuracy, and flexibility. Tabulated potentials
provide moderate performance and accuracy is limited by the interpolation. The performance and
accuracy of custom forces in Python depends entirely on the implementation.

.. _example plugins: https://github.com/glotzerlab/hoomd-blue/tree/trunk-patch/example_plugins
