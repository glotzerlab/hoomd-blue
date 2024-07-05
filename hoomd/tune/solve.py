# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement HOOMD's solver infrastructure.

This module does not intent to be a full optimzation, root finding, etc library,
but offers basic facilities for tuning problems faced by various HOOMD objects.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from math import isclose
import typing

import numpy as np

import hoomd.variant


class SolverStep(metaclass=ABCMeta):
    """Abstract base class various solver types.

    Requires a single method `solve_one` that steps forward one iteration in
    solving the given variable relationship. Users can use subclasses of this
    with `hoomd.tune.ManualTuneDefinition` to tune attributes with a functional
    relation.

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

    @abstractmethod
    def reset(self):
        """Reset all solving internals.

        This should put the solver in its initial state as if it has not seen
        any tunables or done any solving yet.
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


class RootSolver(SolverStep):
    """Abstract base class for finding x such that :math:`f(x) = 0`.

    For solving for a non-zero value, :math:`f(x) - y_t = 0` is solved.
    """
    pass


class ScaleSolver(RootSolver):
    r"""Solves equations of f(x) = y using a ratio of y with the target.

    Each time this solver is called it takes updates according to the following
    equation if the correlation is positive:

    .. math::

        x_n = \min{\left(\frac{\gamma + t}{y + \gamma}, s_{max}\right)} \cdot x

    and

    .. math::

        x_n = \min{\left(\frac{y + \gamma}{\gamma + t}, s_{max}\right)} \cdot x

    if the correlation is negative, where :math:`t` is the target and
    :math:`x_n` is the new x.

    The solver will stop updating when :math:`\lvert y - t \rvert \le tol`.

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
                 correlation="positive",
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
            if self.correlation == "positive":
                scale = (self.gamma + target) / (y + self.gamma)
            else:
                scale = (y + self.gamma) / (self.gamma + target)
        else:
            # y was zero. Try a value an order of magnitude smaller
            if self.correlation == "positive":
                scale = 0.1
            else:
                scale = 1.1

        if scale > self.max_scale:
            scale = self.max_scale
        # Ensures we stay within the tunable's domain (i.e. we don't take on
        # values to high or low).
        tunable.x = tunable.clamp_into_domain(scale * x)
        return False

    def reset(self):
        """Reset all solving internals."""
        pass

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("max_scale", "gamma", "correlation", "tol"))


class _GradientHelper:
    """Provides helper methods for solvers that require gradients."""

    @staticmethod
    def _initialize_tuning(tunable):
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


class SecantSolver(RootSolver, _GradientHelper):
    r"""Solves equations of f(x) = y using the secant method.

    The solver updates ``x`` each step via,

    .. math::

        x_n = x - \gamma \cdot (y - t) \cdot \frac{x - x_{o}}{y - y_{old}}

    where :math:`o` represent the old values, :math:`n` the new, and :math:`t`
    the target. Due to the need for a previous value, then first time this is
    called it makes a slight jump higher or lower to start the method.

    The solver will stop updating when :math:`\lvert y - t \rvert \le tol`.

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
        self.tol = tol
        self.reset()

    def solve_one(self, tunable):
        """Solve one step."""
        # start tuning new tunable
        x, y, target = tunable.x, tunable.y, tunable.target
        f_x = y - target
        if tunable not in self._previous_pair:
            self._previous_pair[tunable] = (x, f_x)
            self._initialize_tuning(tunable)
            return False

        # check for convergence
        if abs(f_x) <= self.tol:
            return True

        old_x, old_f_x = self._previous_pair[tunable]
        # Attempt to find the new value of x using the standard secant formula.
        # We use f(x) = y - target since this is the root we are searching for.
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

    def reset(self):
        """Reset all solving internals."""
        self._previous_pair = {}
        self._counters = {}

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("gamma", "tol", "_counters", "_previous_pair"))


class Optimizer(SolverStep):
    """Abstract base class for optimizing :math:`f(x)`."""
    pass


class GradientDescent(Optimizer, _GradientHelper):
    r"""Solves equations of :math:`min_x f(x)` using gradient descent.

    Derivatives are computed using the forward difference.

    The solver updates ``x`` each step via,

    .. math::

        x_n = x_{n-1} - \alpha {\left (1 - \kappa) \nabla f
                                + \kappa \Delta_{n-1} \right)}

    where :math:`\Delta` is the last step size. This gives the optimizer a sense
    of momentum which for noisy (stochastic) optimization can lead to smoother
    optimization. Due to the need for two values to compute a derivative, then
    first time this is called it makes a slight jump higher or lower to start
    the method.

    The solver will stop updating when a maximum is detected (i.e. the step size
    is smaller than ``tol``).

    Args:
        alpha (`hoomd.variant.variant_like`, optional): Either a number between
            0 and 1 used to dampen the rate of change in x or a variant that
            varies not by timestep but by the number of times `solve` has been
            called (i.e. the number of steps taken) (defaults to 0.1). ``alpha``
            scales the corrections to x each iteration.  Larger values of
            ``alpha`` lead to larger changes while a ``alpha`` of 0 leads to no
            change in x at all.
        kappa (`numpy.ndarray`, optional): Real number array that determines how
            much of the previous steps' directions to use (defaults to ``None``
            which does no averaging over past step directions). The array values
            correspond to weight that the :math:`N` last steps are weighted when
            combined with the current step. The current step is weighted by
            :math:`1 - \sum_{i=1}^{N} \kappa_i`.
        tol (`float`, optional): The absolute tolerance for convergence of
            y, (defaults to ``1e-5``).
        maximize (`bool`, optional): Whether or not to maximize function
            (defaults to ``True``).
        max_delta (`float`, optional): The maximum step size to allow (defaults
            to ``None`` which allows a step size of any length).

    Attributes:
        kappa (numpy.ndarray): Real number array that determines how much of the
            previous steps' directions to use. The array values correspond to
            weight that the :math:`N` last steps are weighted when combined with
            the current step. The current step is weighted by
            :math:`1 - \sum_{i=1}^{N} \kappa_i`.
        tol (float): The absolute tolerance for convergence of y.
        maximize (bool): Whether or not to maximize function.
        max_delta (float): The maximum step size to allow.
    """

    _max_allowable_counter = 3

    def __init__(self,
                 alpha: float = 0.1,
                 kappa: typing.Optional[np.ndarray] = None,
                 tol: float = 1e-5,
                 maximize: bool = True,
                 max_delta: typing.Optional[float] = None):
        self.alpha = alpha
        self.kappa = None if kappa is None else kappa
        if self.kappa is not None:
            self._remainder_kappa = 1 - kappa.sum()
        self.tol = tol
        self.maximize = maximize
        self.max_delta = max_delta
        self._tuned = set()
        self.reset()

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
        self._cnt += 1
        return super().solve(tunables)

    @property
    def alpha(self):
        """float: Number between 0 and 1 that dampens of change in x.

        Larger values of ``alpha`` lead to larger changes while a ``alpha``
        of 0 leads to no change in x at all. The property returns the current
        ``alpha`` given the current number of steps.

        The property can be set as in the constructor.
        """
        return self._alpha(self._cnt)

    @alpha.setter
    def alpha(self, new_alpha: hoomd.variant.variant_like):
        if isinstance(new_alpha, float):
            self._alpha = hoomd.variant.Constant(new_alpha)
        elif isinstance(new_alpha, hoomd.variant.Variant):
            self._alpha = new_alpha
        else:
            raise TypeError(
                "Expected either a hoomd.variant.variant_like object.")

    def solve_one(self, tunable):
        """Solve one step."""
        # Already solved.
        if tunable in self._tuned:
            return True

        x, y = tunable.x, tunable.y
        # start tuning new tunable
        if tunable not in self._previous_pair:
            self._previous_pair[tunable] = (x, y)
            self._initialize_tuning(tunable)
            return False

        old_x, old_y = self._previous_pair[tunable]

        # Handle y not changing
        if old_y == y:
            new_x = self._handle_static_y(tunable, x, old_x)
            return False

        grad = (y - old_y) / (x - old_x)
        if self.maximize:
            grad *= -1

        delta = self._get_delta(grad)
        if abs(delta) < self.tol:
            self._tuned.add(tunable)
            return True
        new_x = tunable.clamp_into_domain(x - delta)
        # We need to check if the new x is essentially the same as the previous.
        # If this is the case we should not update the entry in
        # `self._previous_pair` as this would prevent all future tunings.
        if not isclose(new_x, x):
            tunable.x = new_x
            self._previous_pair[tunable] = (x, y)
            if self.kappa is not None:
                self._past_grads[-1] = grad
                self._past_grads = np.roll(self._past_grads, 1)
        return False

    def _get_delta(self, grad: float):
        if self.kappa is None or self._past_grads is None:
            delta = self.alpha * grad
        else:
            previous_contribution = np.sum(self.kappa * self._past_grads[:-1])
            current_contribution = self._remainder_kappa * grad
            delta = self.alpha * (current_contribution + previous_contribution)
        if self.max_delta is None or abs(delta) <= self.max_delta:
            return delta
        return (grad / abs(grad)) * self.max_delta

    def reset(self):
        """Reset all solving internals."""
        self._cnt = 0
        self._previous_pair = {}
        if self.kappa is not None:
            self._past_grads = np.zeros(len(self.kappa) + 1)

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("alpha", "tol", "_previous_pair")) and np.array_equal(
                self.kappa, other.kappa)


class _Repeater:

    def __init__(self, value):
        self._a = value

    def __call__(self):
        return self._a


class GridOptimizer(Optimizer):
    """Optimize by consistently narrowing the range where the extrema is.

    The algorithm is as follows. Given a domain :math:`d = [a, b]`, :math:`d` is
    broken up into ``n_bins`` subsequent bins. For the next ``n_bins`` calls,
    the optimizer tests the function value at each bin center. The next call
    does one of two things. If the number of rounds has reached ``n_rounds`` the
    optimization is done, and the center of the best bin is the solution.
    Otherwise, another round is performed where the bin's extent is the new
    domain.

    Warning:
        Changing a tunables domain during usage of a `GridOptimizer` results
        in incorrect behavior.

    Args:
        n_bins (`int`, optional): The number of bins in the range to test
            (defaults to 5).
        n_rounds (`int`, optional): The number of rounds to perform the
            optimization over (defaults to 1).
        maximize (`bool`, optional): Whether to maximize or minimize the
            function (defaults to ``True``).
    """

    def __init__(self,
                 n_bins: int = 5,
                 n_rounds: int = 1,
                 maximize: bool = True):
        self._n_bins = n_bins
        self._n_rounds = n_rounds
        self._opt = max if maximize else min
        self.reset()

    def solve_one(self, tunable):
        """Perform one step of optimization protocol."""
        if self._solved:
            return True

        # Initialize data for tunable and start on round 1 first bin
        if tunable not in self._bins:
            self._initial_binning(tunable)
            tunable.x = self._get_bin_center(tunable, 0)
            return False

        bin_y = self._bin_y[tunable]
        bin_y.append(tunable.y)
        # Need to increment round or finish optimizing
        if len(bin_y) == self._n_bins:
            index = bin_y.index(self._opt(bin_y))
            boundaries = self._bins[tunable][index:index + 2]
            if self._round[tunable] == self._n_rounds:
                center = sum(boundaries) / 2
                tunable.x = center
                self._final_bins[tunable] = tuple(boundaries)
                self._solved[tunable] = True
                return True
            self._bins[tunable] = np.linspace(*boundaries, self._n_bins + 1)
            bin_y.clear()
            tunable.x = self._get_bin_center(tunable, 0)
            self._round[tunable] += 1
            return False

        # Standard intra-round optimizing.
        tunable.x = self._get_bin_center(tunable, len(bin_y))
        return False

    def reset(self):
        """Reset all solving internals."""
        # bin boundaries
        self._bins = {}
        # The current optimization round
        self._round = defaultdict(_Repeater(1))
        # The y values for the center of the bins
        self._bin_y = defaultdict(list)
        # The final bin for each tunable
        self._final_bins = {}
        # Whether a tunable is solved
        self._solved = defaultdict(_Repeater(False))

    def _initial_binning(self, tunable):
        """Get the initial bin boundaries for a tunable."""
        min_, max_ = tunable.domain
        if max_ is None or min_ is None:
            raise ValueError(
                "GridOptimizer requires max and min x value to tune.")
        self._bins[tunable] = np.linspace(min_, max_, self._n_bins + 1)

    def _get_bin_center(self, tunable, index):
        """Get the bin center for a given tunable and bin index."""
        min_, max_ = tunable.domain
        return sum(self._bins[tunable][index:index + 2]) / 2

    def __eq__(self, other):
        """Test for equality."""
        if not isinstance(other, SolverStep):
            return NotImplemented
        if not isinstance(other, type(self)):
            return False
        return (all(
            getattr(self, attr) == getattr(other, attr)
            for attr in ("_n_bins", "_n_rounds", "_round", "_solved", "_bin_y"))
                and self._bins.keys() == other._bins.keys() and all(
                    np.array_equal(a, other._bins[key])
                    for a, key in self._bins.items()))
