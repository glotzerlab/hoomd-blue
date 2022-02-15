# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement VolumeMoveSize."""

import hoomd
from hoomd.custom import _InternalAction
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import (OnlyFrom, OnlyTypes, OnlyIf,
                                      to_type_converter)

from hoomd.tune import _InternalCustomTuner
from hoomd.tune.attr_tuner import (_TuneDefinition, SolverStep, ScaleSolver,
                                   SecantSolver)

from hoomd.hpmc.integrate import HPMCIntegrator


class _MoveSizeTuneDefinition(_TuneDefinition):
    """Encapsulates getting the acceptance rate and getting/setting move size.

    This class should only be used for the _InternalMoveSize class to tune HPMC
    move sizes. For this class 'x' is the move size and 'y' is the acceptance
    rate.
    """
    acceptable_attrs = {
        "volume", "aspect", "shear_x", "shear_y", "shear_z", "length_x",
        "length_y", "length_z"
    }

    def __init__(self, boxmc, attr, target, domain=None):
        if attr not in self.acceptable_attrs:
            raise ValueError(f"Only {self.acceptable_attrs} are allowed as "
                             f"tunable attributes not {attr}.")
        self.attr = attr
        self.boxmc = boxmc
        self.previous_accepted_moves = None
        self.previous_total = None
        self.previous_acceptance_rate = None
        super().__init__(target, domain)

    def get_ratio(self):
        if self.attr.startswith("len"):
            attr = "volume"
        elif self.attr.startswith("sh"):
            attr = "shear"
        else:
            attr = self.attr
        return getattr(self.boxmc, attr + "_moves")

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
        # `SolverStep` object to skip tuning this attribute for now.
        if self.previous_total is None or total_moves == 0:
            self.previous_accepted_moves = accepted_moves
            self.previous_total = total_moves
            return None

        # If no more trial moves have been recorded return previous
        # acceptance_rate.
        elif self.previous_total == total_moves:
            return self.previous_acceptance_rate

        # If we have recorded a previous total then this condition implies a new
        # run call. We should be able to tune here as we have no other
        # indication the system has changed.
        elif (self.previous_total > total_moves
              or self.previous_accepted_moves > accepted_moves):
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

    def _get_x(self):
        splits = self.attr.split("_")
        attr = splits[0]
        if len(splits) > 1:
            to_index = {"x": 0, "y": 1, "z": 2}
            return getattr(self.boxmc, attr)["delta"][to_index[splits[1]]]
        return getattr(self.boxmc, attr)["delta"]

    def _set_x(self, value):
        splits = self.attr.split("_")
        attr = splits[0]
        if len(splits) == 1:
            getattr(self.boxmc, self.attr)["delta"] = value
            return
        to_index = {"x": 0, "y": 1, "z": 2}
        new_value = list(getattr(self.boxmc, attr)["delta"])
        new_value[to_index[splits[1]]] = value
        getattr(self.boxmc, attr)["delta"] = new_value

    def __hash__(self):
        return hash((self.attr, self._target, self._domain))

    def __eq__(self, other):
        return (self.attr == other.attr and self._target == other._target
                and self._domain == other._domain)


class _InternalVolumeMoveSize(_InternalAction):
    """Internal class for the box volume tuner."""
    _min_move_size = 1e-7

    def __init__(
        self,
        boxmc,
        moves,
        target,
        solver,
        max_move_size=None,
    ):
        # Flags for knowing when to update classes attributes
        self._update_move_sizes = False
        self._should_update_tunables = False

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
        params = ParameterDict(
            boxmc=hoomd.hpmc.update.BoxMC,
            moves=[
                OnlyFrom(_MoveSizeTuneDefinition.acceptable_attrs,
                         postprocess=self._flag_new_tunables)
            ],
            target=OnlyTypes(float, postprocess=self._target_postprocess),
            solver=OnlyTypes(SolverStep, strict=True),
            max_move_size=OnlyIf(
                to_type_converter({
                    attr: OnlyTypes(float,
                                    allow_none=True,
                                    postprocess=self._flag_move_size_update)
                    for attr in _MoveSizeTuneDefinition.acceptable_attrs
                }),))
        params["boxmc"] = boxmc
        params["moves"] = moves
        params["target"] = target
        params["solver"] = solver
        if max_move_size is None:
            max_move_size = {
                attr: None for attr in _MoveSizeTuneDefinition.acceptable_attrs
            }
        params["max_move_size"] = max_move_size
        self._param_dict.update(params)

        self._update_tunables()

    def attach(self, simulation):
        if not isinstance(simulation.operations.integrator, HPMCIntegrator):
            raise RuntimeError(
                "MoveSizeTuner can only be used in HPMC simulations.")
        self._is_attached = True

    @property
    def _attached(self):
        """bool: Whether or not the tuner is attached to a simulation."""
        return self._is_attached

    @property
    def tuned(self):
        """bool: Whether or not the move sizes are considered tuned.

        A `MoveSize` object is considered tuned if it the solver tolerance has
        been met by all tunables for 2 iterations.
        """
        return self._tuned > 1

    def detach(self):
        self._is_attached = False

    def act(self, timestep=None):
        """Tune move sizes.

        Args:
            timestep (`int`, optional): Current simulation timestep.
        """
        if self._is_attached:
            # update maximum move sizes
            if self._should_update_tunables:
                self._update_tunables()
            if self._update_move_sizes:
                for tunable in self._tunables:
                    max_move_size = self.max_move_size[tunable.attr]
                    tunable.domain = (self._min_move_size, max_move_size)

            if self.tuned:
                return
            tuned = self.solver.solve(self._tunables)
            self._tuned = self._tuned + 1 if tuned else 0

    def _update_tunables(self):
        tunables = self._tunables

        def filter_tunables(tunable):
            return tunable.attr in self.moves

        self._tunables = list(filter(filter_tunables, tunables))
        tune_definitions = {t.attr for t in self._tunables}

        # Add any move size tune definitions that are required by the new
        # specification.
        for move in self.moves:
            if move in tune_definitions:
                continue
            max_move_size = self.max_move_size[move]
            move_definition = _MoveSizeTuneDefinition(
                self.boxmc, move, self.target,
                (self._min_move_size, max_move_size))
            self._tunables.append(move_definition)
        self._should_update_tunables = False
        self._tuned = 0

    def _update_tunables_attr(self, attr, value):
        for tunable in self._tunables:
            setattr(tunable, attr, value)

    def _target_postprocess(self, target):
        if not (0 <= target <= 1):
            raise ValueError(f"target {target} should be between 0 and 1.")

        self._update_tunables_attr('target', target)
        self._tuned = 0
        return target

    def _flag_move_size_update(self, value):
        self._update_move_sizes = True
        return value

    def _flag_new_tunables(self, value):
        self._should_update_tunables = True
        return value
        self._param_dict._cpp_obj = 5


class VolumeMoveSize(_InternalCustomTuner):
    """Tunes HPMCIntegrator move sizes to targeted acceptance rate.

    For most common creation of a `MoveSize` tuner see `MoveSize.secant_solver`
    and `MoveSize.scale_solver` respectively.

    Args:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC` object to
            tune.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'delta'.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.SolverStep`): A solver that tunes move sizes to
            reach the specified target.
        types (list[str]): A list of string particle types to tune the move
            size for, defaults to None which upon attaching will tune all types
            in the system currently.
        max_move_size (float): The maximum volume move size to attempt.

    Attributes:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC` object to
            tune.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'a' and 'd'.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (hoomd.tune.SolverStep): A solver that tunes move sizes to
            reach the specified target.
        types (list[str]): A list of string particle
            types to tune the move size for, defaults to None which upon
            attaching will tune all types in the system currently.
        max_move_size (float): The maximum value of a volume move
            size to attempt.

    Note:
        Limiting the maximum move sizes can lead to the inability to converge to
        the desired acceptance rate. Also, not limiting the move size can lead
        to move sizes that require the use of multiple periodic images to check
        for overlaps, especially in low density systems since the acceptance
        rate tends towards 1. Therefore, it is recommended to pick a moderate
        maximum move size for at least the translational moves to prevent
        requiring checking periodic images.
    """
    _internal_class = _InternalVolumeMoveSize

    @classmethod
    def scale_solver(cls,
                     trigger,
                     boxmc,
                     moves,
                     target,
                     max_move_size=None,
                     max_scale=2.,
                     gamma=1.,
                     tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.ScaleSolver`.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC`
                object to tune.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'a' and 'd'.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (list[str]): A list of string particle types to tune the
                move size for, defaults to None which upon attaching will tune
                all types in the system currently.
            max_move_size (float): The maximum value of a volume
                move size to attempt.
            max_scale (float): Maximum scale factor.
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
        solver = ScaleSolver(max_scale, gamma, 'negative', tol)
        return cls(trigger, boxmc, moves, target, solver, max_move_size)

    @classmethod
    def secant_solver(cls,
                      trigger,
                      boxmc,
                      moves,
                      target,
                      max_move_size=None,
                      gamma=0.8,
                      tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.SecantSolver`.

        This solver can be faster than `hoomd.tune.ScaleSolver`, but depending
        on the system slightly less stable. In general, with the default value
        of gamma this should not be a problem.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC`
                object to tune.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'a' and 'd'.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (list[str]): A list of string
                particle types to tune the move size for, defaults to None which
                upon attaching will tune all types in the system currently.
            max_move_size (float): The maximum value of a volume
                move size to attempt, defaults to ``None`` which represents no
                maximum move size.
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
            convergence and can slow it done. In addition, large values of
            ``gamma`` can make the solver unstable especially when tuning
            frequently.
        """
        solver = SecantSolver(gamma, tol)
        return cls(trigger, boxmc, moves, target, solver, max_move_size)
