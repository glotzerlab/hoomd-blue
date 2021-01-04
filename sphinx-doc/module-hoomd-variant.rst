hoomd.variant
-------------

.. rubric:: Overview

.. autosummary::
    :nosignatures:

    hoomd.variant.Constant
    hoomd.variant.Cycle
    hoomd.variant.Power
    hoomd.variant.Ramp
    hoomd.variant.Variant

.. rubric:: Details

.. automodule:: hoomd.variant
    :synopsis: Values that vary as a function of time step.
    :members:

    .. autoclass:: Constant(value)
    .. autoclass:: Cycle(A, B, t_start, t_A, t_AB, t_B, t_BA)
    .. autoclass:: Power(A, B, power, t_start, t_ramp)
    .. autoclass:: Ramp(A, B, t_start, t_ramp)
    .. autoclass:: Variant()
