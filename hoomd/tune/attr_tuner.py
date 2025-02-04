# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement tuner utility classes."""

from abc import ABCMeta, abstractmethod


class _TuneDefinition(metaclass=ABCMeta):
    """Internal class for defining y = f(x) relations.

    This class is designed to allow for tuning x to achieve a specified value
    for y over a domain T. It abstracts over getting and setting x and getting
    y. The class also provides helper functions for ensuring x is always set to
    a value within the specified domain.
    """

    def __init__(self, target, domain=None):
        self.domain = domain
        self._target = target

    def in_domain(self, value):
        """Check whether a value is in the domain.

        Args:
            value (``any``): A value that can be compared to the minimum and
                maximum of the domain.

        Returns:
            bool: Whether the value is in the domain of x.
        """
        if self.domain is None:
            return True
        else:
            lower_bound, upper_bound = self.domain
            return ((lower_bound is None or lower_bound <= value)
                    and (upper_bound is None or value <= upper_bound))

    def clamp_into_domain(self, value):
        """Return the closest value within the domain.

        Args:
            value (``any``): A value of the same type as x.

        Returns:
            The value clamps within the domains of x. Clamping here refers to
            returning the value or minimum or maximum of the domain if value is
            outside the domain.
        """
        if self._domain is None:
            return value
        else:
            lower_bound, upper_bound = self.domain
            if lower_bound is not None and value < lower_bound:
                return lower_bound
            elif upper_bound is not None and value > upper_bound:
                return upper_bound
            else:
                return value

    @property
    def x(self):
        """The dependent variable.

        Can be set. When set the setting value is clamped within the provided
        domain. See `clamp_into_domain` for further explanation.
        """
        return self._get_x()

    @x.setter
    def x(self, value):
        return self._set_x(self.clamp_into_domain(value))

    @property
    def max_x(self):
        """Maximum allowed x value."""
        if self.domain is None:
            return None
        else:
            return self.domain[1]

    @property
    def min_x(self):
        """Minimum allowed y value."""
        if self.domain is None:
            return None
        else:
            return self.domain[0]

    @property
    def y(self):
        """The independent variable, and is unsettable."""
        return self._get_y()

    @property
    def target(self):
        """The targetted y value, can be set."""
        return self._get_target()

    @target.setter
    def target(self, value):
        self._set_target(value)

    @abstractmethod
    def _get_x(self):
        pass

    @abstractmethod
    def _set_x(self):
        pass

    @abstractmethod
    def _get_y(self):
        pass

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    @property
    def domain(self):
        """tuple[``any``, ``any``]: A tuple pair of the minimum and maximum \
            accepted values of x.

        When the domain is None, any value of x is accepted. Either the minimum
        or maximum can be set to ``None`` as well which means there is no
        maximum or minimum. The domain is used to wrap values within the
        specified domain when setting x.
        """
        if self._domain is not None:
            return tuple(self._domain)
        else:
            return None

    @domain.setter
    def domain(self, value):
        if value is not None and not len(value) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = value

    def __hash__(self):
        raise NotImplementedError("This object is not hashable.")

    def __eq__(self, other):
        raise NotImplementedError("This object is not equatable.")


class ManualTuneDefinition(_TuneDefinition):
    """Class for defining y = f(x) relationships for tuning x for a set y \
    target.

    This class is made to be used with `hoomd.tune.SolverStep` subclasses.  Here
    y represents a dependent variable of x. In general, x and y should be of
    type `float`, but specific `hoomd.tune.SolverStep` subclasses may accept
    other types.

    A special case for the return type of y is ``None``. If the value is
    currently inaccessible or would be invalid, a `ManualTuneDefinition` object
    can return a y of ``None`` to indicate this. `hoomd.tune.SolverStep`
    objects will handle this automatically. Since we check for ``None``
    internally in `hoomd.tune.SolverStep` objects, a `ManualTuneDefinition`
    object's ``y`` property should be consistant when called multiple times
    within a timestep.

    When setting ``x`` the value is clamped between the given domain via,

    .. math::

        x &= x_{max}, \\text{ if } x_n > x_{max},\\\\
        x &= x_{min}, \\text{ if } x_n < x_{min},\\\\
        x &= x_n, \\text{ otherwise}

    Args:
        get_y (``callable``): A callable that gets the current value for y.
        target (``any``): The target y value to approach.
        get_x (``callable``): A callable that gets the current value for x.
        set_x (``callable``): A callable that sets the current value for x.
        domain (`tuple` [``any``, ``any``], optional): A tuple pair of the
            minimum and maximum accepted values of x, defaults to `None`. When,
            the domain is `None` then any value of x is accepted. Either the
            minimum or maximum can be set to `None` as well which means there is
            no maximum or minimum. The domain is used to wrap values within the
            specified domain when setting x.

    Note:
        Placing domain restrictions on x can lead to the target y value being
        impossible to converge to. This will lead to the `hoomd.tune.SolverStep`
        object passed this tunable to never finish solving regardless if all
        other `ManualTuneDefinition` objects are converged.
    """

    def __init__(self, get_y, target, get_x, set_x, domain=None):
        self._user_get_x = get_x
        self._user_set_x = set_x
        self._user_get_y = get_y
        self._target = target
        if domain is not None and not len(domain) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = domain

    def _get_x(self):
        return self._user_get_x()

    def _set_x(self, value):
        return self._user_set_x(value)

    def _get_y(self):
        return self._user_get_y()

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    def __hash__(self):
        """Compute a hash of the tune definition."""
        return hash((self._user_get_x, self._user_set_x, self._user_get_y,
                     self._target))

    def __eq__(self, other):
        """Test for equality."""
        return (self._user_get_x == other._user_get_x
                and self._user_set_x == other._user_set_x
                and self._user_get_y == other._user_get_y
                and self._target == other._target)
