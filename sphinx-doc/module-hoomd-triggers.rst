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
    :members:
    :undoc-members:

    .. autoclass:: After(timestep)
    .. autoclass:: And(triggers)
    .. autoclass:: Before(timestep)
    .. autoclass:: Not(trigger)
    .. autoclass:: On(timestep)
    .. autoclass:: Or(triggers)
    .. autoclass:: Periodic(period, phase)
    .. autoclass:: Trigger()
