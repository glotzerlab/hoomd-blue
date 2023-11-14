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
