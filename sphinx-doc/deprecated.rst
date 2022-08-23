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
