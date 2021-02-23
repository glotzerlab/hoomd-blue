# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Define quantities that vary over the simulation.

A `Variant` object represents a scalar function of the time step. Some
**Operations** accept `Variant` values for certain parameters, such as the
``kT`` parameter to `NVT`.
"""

from hoomd import _hoomd


class Variant(_hoomd.Variant):
    """Variant base class.

    Variants define values as a function of the simulation time step. Use one of
    the built in types or define your own custom function:

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

    @property
    def min(self):
        """The minimum value of this variant."""
        return self._min()

    @property
    def max(self):
        """The maximum value of this variant."""
        return self._max()

    def __getstate__(self):
        """Get the variant's ``__dict__`` attributue."""
        return self.__dict__

    def __setstate__(self, state):
        """Restore the state of the variant."""
        _hoomd.Variant.__init__(self)
        self.__dict__ = state


class Constant(_hoomd.VariantConstant, Variant):
    """A constant value.

    Args:
        value (float): The value.

    `Constant` returns *value* at all time steps.

    Attributes:
        value (float): The value.
    """

    def __init__(self, value):
        Variant.__init__(self)
        _hoomd.VariantConstant.__init__(self, value)

    def __eq__(self, other):
        if not isinstance(other, Variant):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return other.value == self.value


class Ramp(_hoomd.VariantRamp, Variant):
    """A linear ramp.

    Args:
        A (float): The start value.
        B (float): The end value.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.

    `Ramp` holds the value *A* until time *t_start*. Then it ramps linearly from
    *A* to *B* over *t_ramp* steps and holds the value *B*.

    .. image:: variant-ramp.svg

    Attributes:
        A (float): The start value.
        B (float): The end value.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """

    def __init__(self, A, B, t_start, t_ramp):
        Variant.__init__(self)
        _hoomd.VariantRamp.__init__(self, A, B, t_start, t_ramp)

    def __eq__(self, other):
        if not isinstance(other, Variant):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        attrs = ('A', 'B', 't_start', 't_ramp')
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)


class Cycle(_hoomd.VariantCycle, Variant):
    """A cycle of linear ramps.

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
        Variant.__init__(self)
        _hoomd.VariantCycle.__init__(self, A, B, t_start, t_A, t_AB, t_B, t_BA)

    def __eq__(self, other):
        if not isinstance(other, Variant):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        attrs = ('A', 'B', 't_start', 't_A', 't_AB', 't_B', 't_BA')
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)


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

    .. code-block:: python

        p = Power(2, 8, 1 / 10, 10, 20)

    .. image:: variant-power.svg

    Attributes:
        A (float): The start value.
        B (float): The end value.
        power (float): The power of the approach to ``B``.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """

    def __init__(self, A, B, power, t_start, t_ramp):
        Variant.__init__(self)
        _hoomd.VariantPower.__init__(self, A, B, power, t_start, t_ramp)

    def __eq__(self, other):
        if not isinstance(other, Variant):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        attrs = ('A', 'B', 't_start', 't_ramp', 'power')
        return all(getattr(self, attr) == getattr(other, attr)
                   for attr in attrs)
