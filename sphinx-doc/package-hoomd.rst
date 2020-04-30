hoomd
=====

.. rubric:: Overview

.. py:currentmodule:: hoomd

.. autosummary::
    :nosignatures:

    Simulation
    State
    Snapshot
    Operations
    Snapshot
    Logger
    _CustomOperation
    CustomAction

.. rubric:: Details

.. automodule:: hoomd
    :synopsis: HOOMD-blue main package.
    :undoc-members:
    :imported-members:
    :members: Simulation,
              State,
              Snapshot,
              Operations,
              Logger,
              _CustomOperation

.. autoclass:: CustomAction
   :exclude-members: flags, log_quantities
   :members: attach, detach, act

.. rubric:: Modules

.. toctree::
   :maxdepth: 3

   module-hoomd-analyze
   module-hoomd-cite
   module-hoomd-comm
   module-hoomd-compute
   module-hoomd-device
   module-hoomd-dump
   module-hoomd-filters
   module-hoomd-triggers
   module-hoomd-update
   module-hoomd-util
   module-hoomd-variant
   hoomd-misc
