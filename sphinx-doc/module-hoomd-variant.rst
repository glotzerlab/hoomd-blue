.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

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
    :no-members:

    .. autoclass:: Constant(value)
        :members: __eq__
    .. autoclass:: Cycle(A, B, t_start, t_A, t_AB, t_B, t_BA)
        :members: __eq__
    .. autoclass:: Power(A, B, power, t_start, t_ramp)
        :members: __eq__
    .. autoclass:: Ramp(A, B, t_start, t_ramp)
        :members: __eq__
    .. autoclass:: Variant()
