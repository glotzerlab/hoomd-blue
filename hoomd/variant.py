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
    def _min(self):
        raise NotImplementedError

    @property
    def min(self):
        return self._min()

    def _max(self):
        raise NotImplementedError

    @property
    def max(self):
        return self._max()


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


class Cycle(_hoomd.VariantCycle, Variant):
    """ A cycle of linear ramps.

    Args:
        A (float): The first value.
        B (float): The second value.
        t_start (int): The start time step.
        t_A (int): The hold time at the first value.
        t_AB (int): The time spent ramping from A to B.
        t_B (int): The hold time at the second value.
        t_BA (int): The time spent ramping from B to A.

    :py:class:`Cycle` holds the value *A* until time *t_start*. It continues
    holding that value until *t_start + t_A*. Then it ramps linearly from *A* to
    *B* over *t_AB* steps and holds the value *B* for *t_B* steps. After this,
    it ramps back from *B* to *A* over *t_BA* steps and repeats the cycle
    starting with *t_A*. :py:class:`Cycle` repeats this cycle indefinitely.

    .. image:: variant-cycle.svg

    Attributes:
        A (float): The first value.
        B (float): The second value.
        t_start (int): The start time step.
        t_A (int): The holding time at A.
        t_AB (int): The time spent ramping from A to B.
        t_B (int): The holding time at B.
        t_BA (int): The time spent ramping from B to A.
    """
    def __init__(self, A, B, t_start, t_A, t_AB, t_B, t_BA):
        _hoomd.VariantCycle.__init__(self, A, B, t_start, t_A, t_AB, t_B, t_BA)


class Power(_hoomd.VariantPower, Variant):
    """A approach from initial to final value of x ^ (power).

    Args:
        A (float): The start value.
        B (float): The end value.
        power (float): The power of the approach to ``B``.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.

    :py:class:`Power` holds the value *A* until time *t_start*. Then it
    progresses at :math:`x^{power}` from *A* to *B* over *t_ramp* steps and
    holds the value *B* after that.

    Attributes:
        A (float): The start value.
        B (float): The end value.
        power (float): The power of the approach to ``B``.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """
    def __init__(self, A, B, power, t_start, t_ramp):
        _hoomd.VariantPower.__init__(self, A, B, power, t_start, t_ramp)
