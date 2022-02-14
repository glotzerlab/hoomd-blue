.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.trigger
--------------

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    hoomd.trigger.After
    hoomd.trigger.And
    hoomd.trigger.Before
    hoomd.trigger.Not
    hoomd.trigger.On
    hoomd.trigger.Or
    hoomd.trigger.Periodic
    hoomd.trigger.Trigger

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
