from abc import ABCMeta, abstractmethod


class _TuneDefinition(metaclass=ABCMeta):
    """Internal class for defining y = f(x) relations

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
            The value wrapped within the domains of x. Wrapping here refers to
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
        """The dependent variable."""
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
        """tuple[any, any]: A tuple pair of the minimum and maximum accepted
            values of x.

            When, the domain is `None` then any value of x is accepted. Either
            the minimum or maximum can be set to `None` as well which means
            there is no maximum or minimum. The domain is used to wrap values
            within the specified domain when setting x.
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
    """
    Class for defining y = f(x) relationships for tuning x for a set y target.

    This class is made to be used with `hoomd.tune.SolverStep` subclasses.
    Here y represents a dependent variable of x. In general, x and y should be
    of type `float`, but specific `hoomd.tune.SolverStep` subclasses may accept
    other types.

    Args:
        get_y (``callable``): A callable that gets the current value for y.
        target (``any``): The target y value to approach.
        get_x (``callable``): A callable that gets the current value for x.
        set_x (``callable``): A callable that sets the current value for x.
        domain (:obj:`tuple` [``any``, ``any``], optional): A tuple pair of the
            minimum and maximum accepted values of x, defaults to `None`. When,
            the domain is `None` then any value of x is accepted. Either the
            minimum or maximum can be set to `None` as well which means there is
            no maximum or minimum. The domain is used to wrap values within the
            specified domain when setting x.
    """
    def __init__(self, get_y, target, get_x, set_x, domain=None):
        self.__get_x = get_x
        self.__set_x = set_x
        self.__get_y = get_y
        self._target = target
        if domain is not None and not len(domain) == 2:
            raise ValueError("domain must be a sequence of length two.")
        self._domain = domain

    def _get_x(self):
        return self.__get_x()

    def _set_x(self, value):
        return self.__set_x(value)

    def _get_y(self):
        return self.__get_y()

    def _get_target(self):
        return self._target

    def _set_target(self, value):
        self._target = value

    def __hash__(self):
        return hash((self.__get_x, self.__set_x, self.__get_y, self._target))

    def __eq__(self, other):
        return (self.__get_x == other.__get_x
                and self.__set_x == other.__set_x
                and self.__get_y == other.__get_y
                and self._target == other._target)


class SolverStep(metaclass=ABCMeta):
    """Abstract base class for "solving" stepwise equations of f(x) = y.

    Requires a single method `SolverStep._solve_one` that steps forward one
    iteration in solving the given variable relationship. Users can use
    subclasses of this with `hoomd.tune.ManualTuneDefinition` to tune attributes
    with a functional relation.

    Note:
        A `SolverStep` object requires manual iteration to converge. This is to
        support the use case of measuring quantities that require running the
        simulation for some amount of time after one iteration before
        remeasuring the dependent variable (i.e. the y). `SolverStep` object can
        be used in `hoomd.custom.Action` subclasses for user defined tuners and
        updaters.
    """
    @abstractmethod
    def _solve_one(self, tunable):
        """Takes in a tunable object and attempts to solve x for a specified y.

        Args:
            tunable (`hoomd.tune.ManualTuneDefinition`): A tunable object that
                represents a relationship of f(x) = y.

        Returns:
            bool : Whether or not the tunable converged to the target.
        """
        pass

    def solve(self, tunables):
        """Iterates towards a solution for a list of tunables.

        Args:
            tunables (list[`hoomd.tune.ManualTuneDefinition`]): A list of
                tunable objects that represent a relationship f(x) = y.
        """
        return all(self._solve_one(tunable) for tunable in tunables)


class ScaleSolver(SolverStep):
    """
    Solves equations of f(x) = y using a ratio of the current y with the target.

    Args:
        max_scale (:obj:`float`, optional): The maximum amount to scale the
            current x value with, defaults to 2.0.
        gamma (:obj:`float`, optional): nonnegative real number used to dampen
            or increase the rate of change in x. ``gamma`` is added to the
            numerator and denominator of the ``y / target`` ratio. Larger values
            of ``gamma`` lead to smaller changes while a ``gamma`` of 0 leads to
            scaling x by exactly the ``y / target`` ratio.
        correlation (:obj:`str`, optional): Defines whether the relationship
            between x and y is of a positive or negative correlation, defaults
            to 'positive'. This determines which direction to scale x in for a
            given y.
        tol (:obj:`float`, optional): The absolute tolerance for convergence of
            y, defaults to 1e-5.
    Note:
        This solver is only usable when quantities are strictly positive.
    """
    def __init__(self, max_scale=2.0, gamma=2.0,
                 correlation='positive', tol=1e-5):
        self.max_scale = max_scale
        self.gamma = gamma
        self.correlation = correlation.lower()
        self.tol = tol

    def _solve_one(self, tunable):
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


class SecantSolver(SolverStep):
    """
    Solves equations of f(x) = y using the secant method.

    Args:
        gamma (:obj:`float`, optional): real number between 0 and 1 used to
            dampen the rate of change in x. ``gamma`` scales the corrections to
            x each iteration.  Larger values of ``gamma`` lead to larger changes
            while a ``gamma`` of 0 leads to no change in x at all.
        tol (:obj:`float`, optional): The absolute tolerance for convergence of
            y, defaults to 1e-5.
    Note:
        Tempering the solver with a smaller than 1 ``gamma`` value is crucial
        for numeric stability. If instability is found, then lowering ``gamma``
        accordingly should help.
    """
    def __init__(self, gamma=0.9, tol=1e-5):
        self.gamma = gamma
        self._previous_pair = dict()
        self.tol = tol

    def _solve_one(self, tunable):
        x, y, target = tunable.x, tunable.y, tunable.target
        if abs(y - target) <= self.tol:
            return True

        if tunable not in self._previous_pair:
            # We must perturb x some to get a second point to find the correct
            # root.
            new_x = tunable.clamp_into_domain(x * 1.1)
            if new_x == x:
                new_x = tunable.clamp_into_domain(x * 0.9)
                if new_x == x:
                    raise RuntimeError("Unable to perturb x for secant solver.")
        else:
            # standard secant formula. A brief note, we use f(x) = y - target
            # since this is the root we are searching for.
            old_x, old_f_x = self._previous_pair[tunable]
            self._previous_pair[tunable] = (x, y - target)
            f_x = y - target
            dxdf = (x - old_x) / (f_x - old_f_x)
            new_x = x - (self.gamma * f_x * dxdf)

        self._previous_pair[tunable] = (x, y - target)
        tunable.x = new_x
        return False
