.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
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
* ``hpmc.pair.user.CPPPotentialBase``, ``hpmc.pair.user.CPPPotential``,
  ``hpmc.pair.user.CPPPotentialUnion``, ``hpmc.integrate.HPMCIntegrator.pair_potential``
  (since 4.5.0).

  * Use a `hoomd.hpmc.pair.Pair` potential with `hpmc.integrate.HPMCIntegrator.pair_potentials`.


* ``hoomd.util.GPUNotAvailableError`` (since 4.5.0).

  * use ``hoomd.error.GPUNotAvailableError``.

* ``_InternalCustomUpdater.update`` (since 4.5.0)
* ``_InternalCustomTuner.tune`` (since 4.5.0)
* ``_InternalCustomWriter.write`` (since 4.5.0)
* ``HDF5Log.write`` (since 4.5.0)
* Single-process multi-GPU code path (since 4.5.0)
* ``gpu_ids`` argument to ``GPU`` (since 4.5.0)

  * Use ``gpu_id``.

* ``GPU.devices`` (since 4.5.0)

  * Use ``device``.

* ``box1``, ``box2``, and ``variant`` arguments to ``hoomd.update.BoxResize``.

  * Use ``box``.
