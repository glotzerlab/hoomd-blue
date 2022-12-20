.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Deprecated
==========

Features deprecated in v3.x may be removed in a future v4.0.0 release.

v3.x
----

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
     - Deprecated in
   * - Particle diameters
     - Potentials such as `md.pair.ExpandedLJ`.
     - v3.0.0
   * - ``hoomd.md.pair.aniso.ALJ.mode`` parameter
     - n/a: ``mode`` has no effect since v3.0.0.
     - v3.1.0
   * - ``hoomd.md.pair.aniso.Dipole.mode`` parameter
     - n/a: ``mode`` has no effect since v3.0.0.
     - v3.1.0
   * - ``hoomd.device.GPU.memory_traceback`` parameter
     - n/a: ``memory_traceback`` has no effect since v3.0.0.
     - v3.4.0
   * - ``hoomd.md.dihedral.Harmonic``
     - `hoomd.md.dihedral.Periodic` - new name.
     - v3.7.0
   * - ``ENABLE_MPI_CUDA`` CMake option
     - n/a
     - v3.7.0
   * - ``fix_cudart_rpath`` CMake macro
     - n/a
     - v3.7.0
   * - ``charges`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>`
     - Pass charges to `Rigid.create_bodies <hoomd.md.constrain.Rigid.create_bodies>` or set in system state.
     - v3.7.0
   * - ``diameters`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>`
     - Set diameters in system state.
     - v3.7.0
