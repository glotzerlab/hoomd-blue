# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement tuner utility classes."""

from math import isclose
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

    This class is made to be used with `SolverStep` subclasses.
    Here y represents a dependent variable of x. In general, x and y should be
    of type `float`, but specific `SolverStep` subclasses may accept
    other types.

    A special case for the return type of y is ``None``. If the value is
    currently inaccessible or would be invalid, a `ManualTuneDefinition` object
    can return a y of ``None`` to indicate this. `SolverStep` objects will
    handle this automatically. Since we check for ``None`` internally in
    `SolverStep` objects, a `ManualTuneDefinition` object's ``y`` property
    should be consistant when called multiple times within a timestep.

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
        impossible to converge to. This will lead to the `SolverStep` object
        passed this tunable to never finish solving regardless if all other
        `ManualTuneDefinition` objects are converged.
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


class SolverStep(metaclass=ABCMeta):
    """Abstract base class for "solving" stepwise equations of f(x) = y.

    Requires a single method `solve_one` that steps forward one iteration in
    solving the given variable relationship. Users can use subclasses of this
    with `ManualTuneDefinition` to tune attributes with a functional relation.

    Note:
        A `SolverStep` object requires manual iteration to converge. This is to
        support the use case of measuring quantities that require running the
        simulation for some amount of time after one iteration before
        remeasuring the dependent variable (i.e. the y). `SolverStep` object can
        be used in `hoomd.custom.Action` subclasses for user defined tuners and
        updaters.
    """

    @abstractmethod
    def solve_one(self, tunable):
        """Takes in a tunable object and attempts to solve x for a specified y.

        Args:
            tunable (`hoomd.tune.ManualTuneDefinition`): A tunable object that
                represents a relationship of f(x) = y.

        Returns:
            bool : Whether or not the tunable converged to the target.
        """
        pass

    def _solve_one_internal(self, tunable):
        if tunable.y is None:
            return False
        else:
            return self.solve_one(tunable)

    def solve(self, tunables):
        """Iterates towards a solution for a list of tunables.

        If a y for one of the ``tunables`` is ``None`` then we skip that
        ``tunable``. Skipping implies that the quantity is not tuned and `solve`
        will return `False`.

        Args:
            tunables (list[`hoomd.tune.ManualTuneDefinition`]): A list of
                tunable objects that represent a relationship f(x) = y.

        Returns:
            bool:
                Returns whether or not all tunables were considered tuned by
                the object.
        """
        # Need to convert tuning results to a list first to ensure we don't
        # short circuit in all.
        return all([self._solve_one_internal(tunable) for tunable in tunables])


class ScaleSolver(SolverStep):
    """Solves equations of f(x) = y using a ratio of the current y with the \
    target.

    Args:
        max_scale (`float`, optional): The maximum amount to scale the
            current x value with, defaults to 2.0.
        gamma (`float`, optional): nonnegative real number used to dampen
            or increase the rate of change in x. ``gamma`` is added to the
            numerator and denominator of the ``y / target`` ratio. Larger values
            of ``gamma`` lead to smaller changes while a ``gamma`` of 0 leads to
            scaling x by exactly the ``y / target`` ratio.
        correlation (`str`, optional): Defines whether the relationship
            between x and y is of a positive or negative correlation, defaults
            to 'positive'. This determines which direction to scale x in for a
            given y.
        tol (`float`, optional): The absolute tolerance for convergence of
            y, defaults to 1e-5.

    Note:
        This solver is only usable when quantities are strictly positive.
    """

    def __init__(self,
                 max_scale=2.0,
                 gamma=2.0,
                 correlation='positive',
                 tol=1e-5):
        self.max_scale = max_scale
        self.gamma = gamma
        self.correlation = correlation.lower()
        self.tol = tol

    def solve_one(self, tunable):
        """Solve one step."""
        x, y, target = tunable.x, tunable.y, tunable.target
        if abs(y - target) <= self.tol:
            return True

        if y > 0:
            if self.correlation == 'positive':
                scale = (self.gamma + target) / (y + self.gamma)
            else:
                scale = (y + self.gamma) / (self.gamma + target)
        else:
            # y was zero. Try a value an order of magnitude smaller
            if self.correlation == 'positive':
                scale = 0.1
            else:
                scale = 1.1

        if (scale > self.max_scale):
            scale = self.max_scale
        # Ensures we stay within the tunable's domain (i.e. we don't take on
        # values to high or low).
        tunable.x = tunable.clamp_into_domain(scale * x)
        return False

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ('max_scale', 'gamma', 'correlation', 'tol'))


class SecantSolver(SolverStep):
    """Solves equations of f(x) = y using the secant method.

    Args:
        gamma (`float`, optional): real number between 0 and 1 used to
            dampen the rate of change in x. ``gamma`` scales the corrections to
            x each iteration.  Larger values of ``gamma`` lead to larger changes
            while a ``gamma`` of 0 leads to no change in x at all.
        tol (`float`, optional): The absolute tolerance for convergence of
            y, defaults to 1e-5.

    Note:
        Tempering the solver with a smaller than 1 ``gamma`` value is crucial
        for numeric stability. If instability is found, then lowering ``gamma``
        accordingly should help.
    """

    _max_allowable_counter = 3

    def __init__(self, gamma=0.9, tol=1e-5):
        self.gamma = gamma
        self._previous_pair = dict()
        self.tol = tol
        self._counters = dict()

    def solve_one(self, tunable):
        """Solve one step."""
        # start tuning new tunable
        if tunable not in self._previous_pair:
            self._initialize_tuning(tunable)
            return False

        x, y, target = tunable.x, tunable.y, tunable.target
        # check for convergence
        if abs(y - target) <= self.tol:
            return True

        old_x, old_f_x = self._previous_pair[tunable]

        # Attempt to find the new value of x using the standard secant formula.
        # We use f(x) = y - target since this is the root we are searching for.
        f_x = y - target
        try:
            dxdf = (x - old_x) / (f_x - old_f_x)
        except ZeroDivisionError:  # Implies that y has not changed
            # Given the likelihood for use cases in HOOMD that this implies
            # a lack of equilibration of y or too small of a change.
            new_x = self._handle_static_y(tunable, x, old_x)
        else:
            # We can use the secant formula
            self._counters[tunable] = 0
            new_x = x - (self.gamma * f_x * dxdf)

        # We need to check if the new x is essentially the same as the previous.
        # If this is the case we should not update the entry in
        # `self._previous_pair` as this would prevent all future tunings. To
        # compare we must first clamp the value of the new x appropriately.
        new_x = tunable.clamp_into_domain(new_x)
        if not isclose(new_x, x):
            # We only only update x and the previous tunable information when x
            # changes. This is to allow for us gracefully handling when y is the
            # same multiple times.
            tunable.x = new_x
            self._previous_pair[tunable] = (x, y - target)
        return False

    def _initialize_tuning(self, tunable):
        """Called when a tunable is passed for the first time to solver.

        Perturbs x to allow for the calculation of df/dx.
        """
        x = tunable.x
        new_x = tunable.clamp_into_domain(x * 1.1)
        if new_x == x:
            new_x = tunable.clamp_into_domain(x * 0.9)
            if new_x == x:
                raise RuntimeError("Unable to perturb x for secant solver.")
        tunable.x = new_x
        self._previous_pair[tunable] = (x, tunable.y - tunable.target)

    def _handle_static_y(self, tunable, x, old_x):
        """Handles when y is constant for multiple calls to solve_one.

        We do nothing for the first SecantSolver._max_allowable_counter
        consecutive times, but afterwards we attempt to perturb x in the
        direction of last change, and reset the counter.

        This method is useful to handle y that vary slowly with x (such as move
        sizes and acceptance rates for low density HPMC simulations), or cases
        where y takes a while to equilibrate.
        """
        counter = self._counters.get(tunable, 0) + 1
        if counter > self._max_allowable_counter:
            # We nudge x in the direction of previous change.
            self._counters[tunable] = 0
            return tunable.clamp_into_domain(x + ((x - old_x) * 0.5))
        else:
            self._counters[tunable] = counter
            return x

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ('gamma', 'tol', '_counters', '_previous_pair'))
