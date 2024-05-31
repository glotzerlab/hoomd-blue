# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Tune Newtonian event chain parameters."""

from hoomd.custom import _InternalAction
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.tune import _InternalCustomTuner
from hoomd.tune.attr_tuner import _TuneDefinition
from hoomd.tune import RootSolver, ScaleSolver, SecantSolver
from hoomd.hpmc.nec.integrate import HPMCNECIntegrator


class _ChainTimeTuneDefinition(_TuneDefinition):
    """Encapsulates getting particles per chain and getting/setting chain time.

    This class should only be used for the _InternalChainTime class to tune
    HPMC-NEC chain time. For this class 'x' is the chain time and 'y' is the
    number of particles per chain.
    """

    def __init__(self, target, domain=None):
        self.integrator = None
        self.previous_hit = None
        self.previous_start = None
        self.previous_particles_per_chain = None
        super().__init__(target, domain)

    def _get_y(self):
        statistics = getattr(self.integrator, "nec_counters")
        chain_hit = statistics.chain_at_collision_count
        chain_start = statistics.chain_start_count

        # We return None when no chains are recorded since we don't want
        # the chain time to be updated. Likewise, when we do not have a
        # previous recorded number of particles in a chain we return None
        # since what happened previous timesteps may not be indicative of
        # the current system.
        #
        # None in the hoomd solver infrastructure means that the value
        # either cannot be computed or would be inaccurate at the current
        # time. It informs the `RootSolver` object to skip tuning this
        # attribute for now.
        if self.previous_start is None or chain_start == 0:
            self.previous_hit = chain_hit
            self.previous_start = chain_start
            return None

        # If no more chains have been recorded return previous
        # number of particles_per_chain per chain.
        elif self.previous_start == chain_start:
            return self.previous_particles_per_chain

        # If we have recorded a previous total larger than the current one
        # then this condition implies a new run call. We should be able to
        # tune here as we have no other indication the system has changed.
        elif ((self.previous_start > chain_start)
              or (self.previous_hit > chain_hit)):
            particles_per_chain = chain_hit / chain_start
        else:
            # yapf incorrectly formats this when written on one line
            delta_prev = chain_hit - self.previous_hit
            delta_start = chain_start - self.previous_start
            particles_per_chain = delta_prev / delta_start

        # We store the previous information because this lets us find the
        # acceptance rate since this has last been called which allows for us to
        # disregard the information before the last tune.
        self.previous_hit = chain_hit
        self.previous_start = chain_start
        self.previous_particles_per_chain = particles_per_chain
        return particles_per_chain

    def _get_x(self):
        return self.integrator.chain_time

    def _set_x(self, value):
        self.integrator.chain_time = value

    def __hash__(self):
        return hash(("chain_time", self._target, self._domain))

    def __eq__(self, other):
        return (self._target == other._target and self._domain == other._domain)


class _InternalChainTime(_InternalAction):
    """Internal class for the ChainTime tuner."""
    _min_chain_time = 1e-7

    def __init__(self, target, solver, max_chain_time=None):

        # A flag for knowing when to update the maximum move sizes
        self._update_chain_time = False

        # A counter when tuned reaches 1 it means that the tuner has reported
        # being tuned one time in a row. However, as the first run of the tuner
        # is likely at timestep 0 which means that the counters are (0, 0) and
        # _ChainTimeTuneDefinition returns y == target for that case, we need
        # two rounds of tuning to be sure that we have converged. Since, in
        # general, solvers do not do much if any work on already tuned tunables,
        # this is not a performance problem.
        self._tuned = 0
        self._is_attached = False

        self._chain_time_def = _ChainTimeTuneDefinition(
            target, (self._min_chain_time, max_chain_time))

        param_dict = ParameterDict(
            target=OnlyTypes(float,
                             postprocess=self._process_chain_time_target),
            solver=RootSolver,
            max_chain_time=OnlyTypes(
                float,
                allow_none=True,
                postprocess=self._process_chain_time_range),
            min_chain_time=OnlyTypes(
                float, postprocess=self._process_chain_time_range))

        self._param_dict.update(param_dict)
        self.target = target
        self.solver = solver
        self.max_chain_time = max_chain_time
        self.min_chain_time = self._min_chain_time

    def _process_chain_time_range(self, target):
        # check range
        if not (0 <= target):
            raise ValueError("Chain time {} must be positive.".format(target))
        self._tuned = 0
        return target

    def _process_chain_time_target(self, target):
        # check range
        if not (0 <= target <= 1000):
            raise ValueError(
                "Value {} should be between 0 and 1000.".format(target))

        self._chain_time_def.target = target
        self._tuned = 0
        return target

    def attach(self, simulation):
        if not isinstance(simulation.operations.integrator, HPMCNECIntegrator):
            raise RuntimeError(
                "ChainTimeTuner can only be used in HPMC-NEC simulations.")
        self._chain_time_def.integrator = simulation.operations.integrator
        self._is_attached = True

    @property
    def _attached(self):
        """bool: Whether or not the tuner is attached to a simulation."""
        return self._is_attached

    @property
    def tuned(self):
        """bool: Whether or not the move sizes are considered tuned.

        A `ChainTime` object is considered tuned if it the solver tolerance has
        been met by all tunables for 2 iterations.
        """
        return self._tuned >= 2

    def detach(self):
        self._chain_time_def.integrator = None
        self._is_attached = False

    def act(self, timestep=None):
        """Tune chain time.

        Args:
            timestep (`int`, optional): Current simulation timestep. Is
                currently ignored.
        """
        if self._is_attached:
            # update maximum chain time
            if self._update_chain_time:
                self._chain_time_def.domain = (self.min_chain_time,
                                               self.max_chain_time)

            tuned = self.solver.solve([self._chain_time_def])
            self._tuned = self._tuned + 1 if tuned else 0


class ChainTime(_InternalCustomTuner):
    """Tunes HPMCNECIntegrator chain time to targeted mean particles per chain.

    Tip:
        For the most common creation of a `ChainTime` tuner see
        `ChainTime.secant_solver` and `ChainTime.scale_solver` respectively.

    Args:
        trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when to
            run the tuner.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.RootSolver`): A solver that tunes chain times to
            reach the specified target.
        max_chain_time (float): The maximum value of chain time to attempt.

    Attributes:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (hoomd.tune.RootSolver): A solver that tunes move sizes to reach
            the specified target.
        max_chain_time (float): The maximum value of chain time to attempt.
    """
    _internal_class = _InternalChainTime

    @classmethod
    def scale_solver(cls,
                     trigger,
                     target,
                     max_chain_time=None,
                     max_scale=2.,
                     gamma=1.,
                     tol=1e-2):
        """Create a `ChainTime` tuner with a `hoomd.tune.ScaleSolver`.

        Args:
            trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when
                to run the tuner.
            target (float): The number of collisions in a chain that is
                desired.
            max_chain_time (float): The maximum value of chain time to attempt.
            max_scale (float): The maximum amount to scale the current
                chain_time value with.
            gamma (float): The value of gamma to pass through to
                `hoomd.tune.ScaleSolver`. Controls the size of corrections to
                the move size (larger values increase stability while increasing
                convergence time).
            tol (float): The absolute tolerance to allow between the current
                acceptance rate and the target before the move sizes are
                considered tuned. The tolerance should not be too much lower
                than the default of 0.01 as acceptance rates can vary
                significantly at typical tuning rates.
        """
        solver = ScaleSolver(max_scale, gamma, 'positive', tol)
        return cls(trigger, target, solver, max_chain_time)

    @classmethod
    def secant_solver(cls,
                      trigger,
                      target,
                      max_chain_time=None,
                      gamma=0.8,
                      tol=1e-2):
        """Create a `ChainTime` tuner with a `hoomd.tune.SecantSolver`.

        This solver can be faster than `hoomd.tune.ScaleSolver`, but depending
        on the system slightly less stable. In general, with the default value
        of gamma this should not be a problem.

        Args:
            trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when
                to run the tuner.
            target (float): The number of collisions in a chain that is desired.
            max_chain_time (float): The maximum value of chain time to attempt,
                defaults to ``None`` which represents no maximum chain time.
            gamma (float): The value of gamma to pass through
                to `hoomd.tune.SecantSolver`. Controls the size of corrections
                to the move size (smaller values increase stability). Should be
                between 0 and 1, defaults to 0.8.
            tol (float): The absolute tolerance to allow between the current
                acceptance rate and the target before the move sizes are
                considered tuned. The tolerance should not be too much lower
                than the default of 0.01 as acceptance rates can vary
                significantly at typical tuning rates.

        Note:
            Increasing ``gamma`` towards 1 does not necessarily speed up
            convergence and can slow it down. In addition, large values of
            ``gamma`` can make the solver unstable especially when tuning
            frequently.
        """
        solver = SecantSolver(gamma, tol)
        return cls(trigger, target, solver, max_chain_time)
