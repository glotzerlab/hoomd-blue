.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Deprecated
==========

Features deprecated in HOOMD 4.x may be removed in a future 5.0.0 release.

HOOMD may issue `FutureWarning` messages to provide warnings for breaking changes in the next major
release. Use Python's warnings module to silence warnings which may not be correctable until the
next major release. Use this filter: ``ignore::FutureWarning:hoomd``. See Python's `warnings`
documentation for more information on warning filters.

4.x
---

* ``hoomd.snapshot.from_gsd_snapshot`` (since 4.0.0).

  * Use `hoomd.Snapshot.from_gsd_frame`.

* ``Device.num_cpu_threads > 1`` (since 4.4.0).

  * Set ``num_cpu_threads = 1``.
  * All TBB code will be removed in the 5.0 release.

* ``HPMCIntegrator.depletant_fugacity > 0`` (since 4.4.0).

* ``_InternalCustomUpdater.update`` (since 4.4.1)
* ``_InternalCustomTuner.tune`` (since 4.4.1)
* ``_InternalCustomWriter.write`` (since 4.4.1)
* ``HDF5Log.write`` (since 4.4.1)
