Deprecated features
===================

Features deprecated in v3.0.0-beta.x may be removed in a future beta release and will be removed
before the final 3.0.0 release.

v3.0.0-beta.8
-------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``Snapshot.exists``
     - ``snapshot.communicator.rank == 0``
   * - ``State.snapshot``
     - ``get_snapshot`` and ``set_snapshot``
   * - Settable ``State.box``
     - ``State.set_box``
