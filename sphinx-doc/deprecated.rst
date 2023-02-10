.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Deprecated
==========

Features deprecated in v3.x may be removed in a future v4.0.0 release.

.. note::

    Where noted, suggested replacements will be first available with v4.0.0 and there  will be no
    releases with overlapping support for the two APIs.

v3.x
----

* Particle diameters (since v3.0.0).

  * Use potentials such as `md.pair.ExpandedLJ`.

* ``hoomd.md.pair.aniso.ALJ.mode`` parameter (since v3.1.0).

  * ``mode`` has no effect since v3.0.0.

* ``hoomd.md.pair.aniso.Dipole.mode`` parameter (since v3.1.0)

  * ``mode`` has no effect since v3.0.0.

* ``hoomd.device.GPU.memory_traceback`` parameter (since v3.4.0)

  * ``memory_traceback`` has no effect since v3.0.0.

* ``hoomd.md.dihedral.Harmonic`` (since v3.7.0).

  * Use `hoomd.md.dihedral.Periodic`.

* ``ENABLE_MPI_CUDA`` CMake option (since v3.7.0).
* ``fix_cudart_rpath`` CMake macro (since v3.7.0).
* ``charges`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>` (since v3.7.0).

  * Pass charges to `Rigid.create_bodies <hoomd.md.constrain.Rigid.create_bodies>` or set in system state.

* ``diameters`` key in `Rigid.body <hoomd.md.constrain.Rigid.body>` (since v3.7.0).

  * Set diameters in system state.

* ``hoomd.md.methods.NVE`` (since v3.8.0).

  * Use ``hoomd.md.methods.ConstantVolume`` with ``thermostat=None`` (available in >=4.0.0).

* ``hoomd.md.methods.NVT`` (since v3.8.0).

  * Use ``hoomd.md.methods.ConstantVolume`` with a ``hoomd.md.methods.thermostats.MTTK`` thermostat (available in >=4.0.0).

* ``hoomd.md.methods.Berendsen`` (since v3.8.0).

  * Use ``hoomd.md.methods.ConstantVolume`` with a ``hoomd.md.methods.thermostats.Berendsen`` thermostat (available in >=4.0.0).

* ``hoomd.md.methods.NPH`` (since v3.8.0).

  * Use ``hoomd.md.methods.ConstantPressure` with ``thermostat=None`` (available in >=4.0.0).

* ``hoomd.md.methods.NPT`` (since v3.8.0).

  * Use ``hoomd.md.methods.ConstantPressure`` with a ``hoomd.md.methods.thermostats.MTTK`` thermostat (available in >=4.0.0).

* ``log`` attribute and constructor parameter in `hoomd.write.GSD`.

  * Use ``logger``.
