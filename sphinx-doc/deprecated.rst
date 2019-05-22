Deprecated features
===================

v2.x
----

Commands and features deprecated in v2.x will be removed in v3.0.

:py:mod:`hoomd`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - Python 2.7
     - Python >= 3.6
   * - ``static`` parameter in :py:class:`dump.gsd <hoomd.dump.gsd>`
     - ``dynamic`` parameter
   * - ``set_params`` and other ``set_*`` methods
     - Properties (*under development*)
   * - ``context.initialize``
     - New context API (*under development*)

:py:mod:`hoomd.deprecated`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``deprecated.analyze.msd``
     - Offline analysis: e.g. `Freud's msd module <https://freud.readthedocs.io>`_.
   * - ``deprecated.dump.xml``
     - :py:class:`dump.gsd <hoomd.dump.gsd>`
   * - ``deprecated.dump.pos``
     - :py:class:`dump.gsd <hoomd.dump.gsd>` with on-demand conversion to ``.pos``.
   * - ``deprecated.init.read_xml``
     - :py:class:`init.read_gsd <hoomd.init.read_gsd>`
   * - ``deprecated.init.create_random``
     - `mBuild <https://mosdef-hub.github.io/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.
   * - ``deprecated.init.create_random_polymers``
     - `mBuild <https://mosdef-hub.github.io/mbuild/>`_, `packmol <https://www.ime.unicamp.br/~martinez/packmol/userguide.shtml>`_, or user script.

:py:mod:`hoomd.hpmc`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``ignore_overlaps`` parameter
     - :py:class:`interaction_matrix <hoomd.hpmc.integrate.interaction_matrix>`
   * - ``sphere_union::max_members`` parameter
     - no longer needed
   * - ``convex_polyhedron_union``
     - :py:class:`convex_spheropolyhedron_union <hoomd.hpmc.integrate.convex_spheropolyhedron_union>`, ``sweep_radius=0``
   * - ``setup_pos_writer`` member
     - n/a
   * - ``depletant_mode='circumsphere'``
     - no longer needed
   * - ``max_verts`` parameter
     - no longer needed
   * - ``depletant_mode`` parameter
     - no longer needed
   * - ``ntrial`` parameter
     - no longer needed
   * - ``implicit`` boolean parameter
     - set ``fugacity`` non-zero

:py:mod:`hoomd.cgcmm`:

.. list-table::
   :header-rows: 1

   * - Feature
     - Replace with
   * - ``cgcmm.angle.cgcmm``
     - no longer needed
   * - ``cgcmm.pair.cgcmm``
     - no longer needed
