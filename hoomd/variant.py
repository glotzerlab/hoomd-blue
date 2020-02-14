# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd


class Variant(_hoomd.Variant):
    """ Variant base class.

    Variantsdefine values as a function of the simulation time step. Use one of
    the existing Variant types or define your own custom function:

    .. code:: python

        class CustomVariant(hoomd.variant.Variant):
            def __init__(self):
                hoomd.variant.Variant.__init__(self)

            def __call__(self, timestep):
                return (float(timestep)**(1 / 2))

    .. py:method:: __call__(timestep)

        Evaluate the function.

        :param timestep: The time step.
        :type timestep: int
        :return: The value of the function at the given time step.
        :rtype: float
    """
    pass


class Constant(_hoomd.VariantConstant, Variant):
    """ A constant value.

    Args:
        value (float): The value.

    :py:class:`Constant` returns *value* at all time steps.

    Attributes:
        value (float): The value.
    """
    def __init__(self, value):
        _hoomd.VariantConstant.__init__(self, value)


class Ramp(_hoomd.VariantRamp, Variant):
    """ A linear ramp.

    Args:
        A (float): The start value.
        B (float): The end value.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.

    :py:class:`Ramp` holds the value *A* until time *t_start*. Then it
    ramps linearly from *A* to *B* over *t_ramp* steps and holds the value
    *B* after that.

    .. image:: variant-ramp.svg

    Attributes:
        A (float): The start value.
        B (float): The end value.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """
    def __init__(self, A, B, t_start, t_ramp):
        _hoomd.VariantRamp.__init__(self, A, B, t_start, t_ramp)
