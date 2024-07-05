.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.trigger
--------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.trigger

.. autosummary::
    :nosignatures:

    After
    And
    Before
    Not
    On
    Or
    Periodic
    Trigger
    trigger_like

.. rubric:: Details

.. automodule:: hoomd.trigger
    :synopsis: Trigger events at specific time steps.
    :no-members:

    .. autoclass:: After(timestep)
        :show-inheritance:
    .. autoclass:: And(triggers)
        :show-inheritance:
    .. autoclass:: Before(timestep)
        :show-inheritance:
    .. autoclass:: Not(trigger)
        :show-inheritance:
    .. autoclass:: On(timestep)
        :show-inheritance:
    .. autoclass:: Or(triggers)
        :show-inheritance:
    .. autoclass:: Periodic(period, phase)
        :show-inheritance:
    .. autoclass:: Trigger()
    .. autodata:: trigger_like
