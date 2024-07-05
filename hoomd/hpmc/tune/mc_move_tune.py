# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement generic classes for move acceptance ratio tuning."""
import abc

from hoomd.custom import _InternalAction
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.tune.attr_tuner import _TuneDefinition
from hoomd.tune import RootSolver


class _MCTuneDefinition(_TuneDefinition):
    """Encapsulates getting the acceptance rate and getting/setting MC moves.

    This class should only be used for the _TuneMCMove class to tune HPMC
    move sizes. For this class 'x' is the move size and 'y' is the acceptance
    rate.
    """

    def __init__(self, target, domain=None):
        self.previous_accepted_moves = None
        self.previous_total = None
        self.previous_acceptance_rate = None
        super().__init__(target, domain)

    @abc.abstractmethod
    def get_ratio(self):
        """Get the ratio of accepted to rejected moves."""
        pass

    def _get_y(self):
        ratio = self.get_ratio()
        accepted_moves = ratio[0]
        total_moves = sum(ratio)

        # We return None when no moves are recorded since we don't want
        # the move size to be updated. Likewise, when we do not have a previous
        # recorded acceptance rate we return None since what happened previous
        # timesteps may not be indicative of the current system. None in the
        # hoomd solver infrastructure means that the value either cannot be
        # computed or would be inaccurate at the current time. It informs the
        # `RootSolver` object to skip tuning this attribute for now.
        if self.previous_total is None or total_moves == 0:
            self.previous_accepted_moves = accepted_moves
            self.previous_total = total_moves
            return None

        # Called twice in same step return computed value
        elif (self.previous_total == total_moves
              and self.previous_accepted_moves == accepted_moves):
            return self.previous_acceptance_rate

        # If we have recorded a previous total then this condition implies a new
        # run call. We should be able to tune here as we have no other
        # indication the system has changed.
        elif (self.previous_total >= total_moves
              or self.previous_accepted_moves >= accepted_moves):
            acceptance_rate = accepted_moves / total_moves
        else:
            acceptance_rate = (accepted_moves - self.previous_accepted_moves) \
                               / (total_moves - self.previous_total)

        # We store the previous information becuase this lets us find the
        # acceptance rate since this has last been called which allows for us to
        # disregard the information before the last tune.
        self.previous_accepted_moves = accepted_moves
        self.previous_total = total_moves
        self.previous_acceptance_rate = acceptance_rate
        return acceptance_rate


class _TuneMCMove(_InternalAction):
    """Internal class for the MoveSize tuner."""
    _min_move_size = 1e-7

    def __init__(self, target, solver):
        self._tunables = []
        # A counter when tuned reaches 1 it means that the tuner has reported
        # being tuned one time in a row. However, as the first run of the tuner
        # is likely at timestep 0 which means that the counters are (0, 0) and
        # _MoveSizeTuneDefinition returns y == target for that case, we need two
        # rounds of tuning to be sure that we have converged. Since, in general,
        # solvers do not do much if any work on already tuned tunables, this is
        # not a performance problem.
        self._tuned = 0
        self._is_attached = False

        # This is a bit complicated because we are having to ensure that we keep
        # the list of tunables and the solver updated with the changes to
        # attributes. However, these are simply forwarding a change along.
        param_dict = ParameterDict(target=OnlyTypes(
            float, postprocess=self._target_postprocess),
                                   solver=RootSolver)

        self._param_dict.update(param_dict)
        self.target = target
        self.solver = solver

    def attach(self, simulation):
        self._is_attached = True

    @property
    def _attached(self):
        """bool: Whether or not the tuner is attached to a simulation."""
        return self._is_attached

    @property
    def tuned(self):
        """bool: Whether or not the move sizes are considered tuned.

        An instance is considered tuned if it the solver tolerance has been met
        by all tunables for 2 iterations.
        """
        return self._tuned >= 2

    def detach(self):
        self._is_attached = False

    def act(self, timestep=None):
        """Tune move sizes.

        Args:
            timestep (`int`, optional): Current simulation timestep.
        """
        if self._attached:
            # update maximum move sizes
            tuned = self.solver.solve(self._tunables)
            self._tuned = self._tuned + 1 if tuned else 0

    def _update_tunables_attr(self, attr, value):
        for tunable in self._tunables:
            setattr(tunable, attr, value)

    def _target_postprocess(self, target):
        if not (0 <= target <= 1):
            raise ValueError(f"target {target} should be between 0 and 1.")

        self._update_tunables_attr('target', target)
        self._tuned = 0
        return target
