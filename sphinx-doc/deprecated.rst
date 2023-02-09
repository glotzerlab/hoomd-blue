.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Deprecated
==========

Features deprecated in v3.x may be removed in a future v4.0.0 release.

HOOMD may issue `FutureWarning` messages to provide warnings for breaking changes in the next major
release. Use Python's warnings module to silence warnings which may not be correctable until the
next major release. Use this filter: ``ignore::FutureWarning:hoomd``. See Python's `warnings`
documentation for more information on warning filters.

.. note::

    Where noted, suggested replacements will be first available with v4.0.0 and there  will be no
    releases with overlapping support for the two APIs.

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
   * - ``hoomd.md.methods.NVE``
     - ``hoomd.md.methods.ConstantVolume`` with ``thermostat=None`` (available in >=4.0.0).
     - v3.8.0
   * - ``hoomd.md.methods.NVT``
     - ``hoomd.md.methods.ConstantVolume`` with a ``hoomd.md.methods.thermostats.MTTK`` thermostat (available in >=4.0.0).
     - v3.8.0
   * - ``hoomd.md.methods.Berendsen``
     - ``hoomd.md.methods.ConstantVolume`` with a ``hoomd.md.methods.thermostats.Berendsen`` thermostat (available in >=4.0.0).
     - v3.8.0
   * - ``hoomd.md.methods.NPH``
     - ``hoomd.md.methods.ConstantPressure` with ``thermostat=None`` (available in >=4.0.0).
     - v3.8.0
   * - ``hoomd.md.methods.NPT``
     - ``hoomd.md.methods.ConstantPressure`` with a ``hoomd.md.methods.thermostats.MTTK`` thermostat (available in >=4.0.0).
     - v3.8.0
