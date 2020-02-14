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
    """
    pass


Variant.__call__.__doc__ = """ Evaluate the function.

Args:
    timestep (int): The time step.

Returns:
    float: The value of the function at the given time step.
"""


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
