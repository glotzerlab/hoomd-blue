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

* ``Device.num_cpu_threads > 1`` (since 4.4.0).

  * Set ``num_cpu_threads = 1``.
  * All TBB code will be removed in the 5.0 release.

* ``HPMCIntegrator.depletant_fugacity > 0`` (since 4.4.0).

* Single-process multi-GPU code path (since 4.5.0)
* ``gpu_ids`` argument to ``GPU`` (since 4.5.0)

  * Use ``gpu_id``.

* ``GPU.devices`` (since 4.5.0)

  * Use ``device``.

* ``HPMCIntegrator.external_potential`` (since 4.8.0).

  * Use ``HPMCIntegrator.external_potentials`` (when possible)
