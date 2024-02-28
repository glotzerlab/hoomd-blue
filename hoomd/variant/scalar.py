# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement variants that return scalar values."""

import typing

from hoomd import _hoomd


class Variant(_hoomd.Variant):
    """Variant base class.

    Provides methods common to all variants and a base class for user-defined
    variants.

    Subclasses should override the ``__call__``, ``_min``, and ``_max`` methods
    and must explicitly call the base class constructor in ``__init__``:

    .. code-block:: python

        class CustomVariant(hoomd.variant.Variant):
            def __init__(self):
                hoomd.variant.Variant.__init__(self)

            def __call__(self, timestep):
                return (float(timestep)**(1 / 2))

            def _min(self):
                return 0.0

            def _max(self):
                return float('inf')

    Note:
        Provide the minimum and maximum values in the ``_min`` and ``_max``
        methods respectively.

    .. py:method:: __call__(timestep)

        Evaluate the function.

        :param timestep: The time step.
        :type timestep: int
        :return: The value of the function at the given time step.
        :rtype: float
    """

    @property
    def min(self):
        """The minimum value of this variant for :math:`t \\in [0,\\infty)`."""
        return self._min()

    @property
    def max(self):
        """The maximum value of this variant for :math:`t \\in [0,\\infty)`."""
        return self._max()

    def __getstate__(self):
        """Get the variant's ``__dict__`` attribute."""
        return self.__dict__

    def __setstate__(self, state):
        """Restore the state of the variant."""
        _hoomd.Variant.__init__(self)
        self.__dict__ = state

    def _private_eq(self, other):
        """Return whether two variants are equivalent."""
        if not isinstance(other, Variant):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in self._eq_attrs)


class Constant(_hoomd.VariantConstant, Variant):
    """A constant value.

    Args:
        value (float): The value.

    `Constant` returns `value` at all time steps.

    .. rubric:: Example:

    .. code-block:: python

            variant = hoomd.variant.Constant(1.0)

    Attributes:
        value (float): The value.
    """
    _eq_attrs = ("value",)

    def __init__(self, value):
        Variant.__init__(self)
        _hoomd.VariantConstant.__init__(self, value)

    __eq__ = Variant._private_eq


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
       :alt: Example plot of a ramp variant.

    .. rubric:: Example:

    .. code-block:: python

            variant = hoomd.variant.Ramp(A=1.0,
                                         B=2.0,
                                         t_start=10_000,
                                         t_ramp=100_000)

    Attributes:
        A (float): The start value.
        B (float): The end value.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """
    _eq_attrs = ("A", "B", "t_start", "t_ramp")

    def __init__(self, A, B, t_start, t_ramp):
        Variant.__init__(self)
        _hoomd.VariantRamp.__init__(self, A, B, t_start, t_ramp)

    __eq__ = Variant._private_eq


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

    `Cycle` holds the value *A* until time *t_start*. It continues holding that
    value until *t_start + t_A*. Then it ramps linearly from *A* to *B* over
    *t_AB* steps and holds the value *B* for *t_B* steps. After this, it ramps
    back from *B* to *A* over *t_BA* steps and repeats the cycle starting with
    *t_A*. `Cycle` repeats this cycle indefinitely.

    .. image:: variant-cycle.svg
       :alt: Example plot of a cycle variant.

    .. rubric:: Example:

    .. code-block:: python

            variant = hoomd.variant.Cycle(A=1.0,
                                          B=2.0,
                                          t_start=10_000,
                                          t_A=100_000,
                                          t_AB=1_000_000,
                                          t_B=200_000,
                                          t_BA=2_000_000)

    Attributes:
        A (float): The first value.
        B (float): The second value.
        t_start (int): The start time step.
        t_A (int): The holding time at A.
        t_AB (int): The time spent ramping from A to B.
        t_B (int): The holding time at B.
        t_BA (int): The time spent ramping from B to A.
    """
    _eq_attrs = ("A", "B", "t_start", "t_A", "t_AB", "t_B", "t_BA")

    def __init__(self, A, B, t_start, t_A, t_AB, t_B, t_BA):
        Variant.__init__(self)
        _hoomd.VariantCycle.__init__(self, A, B, t_start, t_A, t_AB, t_B, t_BA)

    __eq__ = Variant._private_eq


class Power(_hoomd.VariantPower, Variant):
    """An approach from initial to final value following ``t**power``.

    Args:
        A (float): The start value.
        B (float): The end value.
        power (float): The power of the approach to ``B``.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.

    `Power` holds the value *A* until time *t_start*. Then it progresses at
    :math:`t^{\\mathrm{power}}` from *A* to *B* over *t_ramp* steps and holds
    the value *B* after that.

    .. image:: variant-power.svg
       :alt: Example plot of a power variant.

    .. rubric:: Example:

    .. code-block:: python

        variant = hoomd.variant.Power(A=2,
                                      B=8,
                                      power=1 / 10,
                                      t_start=10, t_ramp=20)

    Attributes:
        A (float): The start value.
        B (float): The end value.
        power (float): The power of the approach to ``B``.
        t_start (int): The start time step.
        t_ramp (int): The length of the ramp.
    """
    _eq_attrs = ("A", "B", "power", "t_start", "t_ramp")

    def __init__(self, A, B, power, t_start, t_ramp):
        Variant.__init__(self)
        _hoomd.VariantPower.__init__(self, A, B, power, t_start, t_ramp)

    __eq__ = Variant._private_eq


variant_like = typing.Union[Variant, float]
"""
Objects that are like a variant.

Any subclass of `Variant` is accepted along with float instances and objects
convertible to float. They are internally converted to variants of type
`Constant` via ``Constant(float(a))`` where ``a`` is the float or float
convertible object.

Note:
    Attributes that are `Variant` objects can be set via a `variant_like`
    object.
"""
