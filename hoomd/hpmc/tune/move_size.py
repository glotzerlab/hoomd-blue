# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement MoveSize."""

from hoomd.custom import _InternalAction
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
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
    _attr_acceptance = {'a': 'rotate_moves', 'd': 'translate_moves'}

    def __init__(self, attr, type, target, domain=None):
        if attr not in self._attr_acceptance:
            raise ValueError("Only {} are allowed as tunable "
                             "attributes.".format(self._available_attrs))
        self.attr = attr
        self.type = type
        self.integrator = None
        self.previous_accepted_moves = None
        self.previous_total = None
        self.previous_acceptance_rate = None
        super().__init__(target, domain)

    def _get_y(self):
        ratio = getattr(self.integrator, self._attr_acceptance[self.attr])
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
        return getattr(self.integrator, self.attr)[self.type]

    def _set_x(self, value):
        getattr(self.integrator, self.attr)[self.type] = value

    def __hash__(self):
        return hash((self.attr, self.type, self._target, self._domain))

    def __eq__(self, other):
        return (self.attr == other.attr and self.type == other.type
                and self._target == other._target
                and self._domain == other._domain)


class _InternalMoveSize(_InternalAction):
    """Internal class for the MoveSize tuner."""
    _min_move_size = 1e-7

    def __init__(self,
                 moves,
                 target,
                 solver,
                 types=None,
                 max_translation_move=None,
                 max_rotation_move=None):
        # A flag for knowing when to update the maximum move sizes
        self._update_move_sizes = False

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

        # set up maximum trial move sizes
        t_moves = TypeParameter(
            'max_translation_move', 'particle_type',
            TypeParameterDict(OnlyTypes(float,
                                        postprocess=self._flag_move_size_update,
                                        allow_none=True),
                              len_keys=1))
        r_moves = TypeParameter(
            'max_rotation_move', 'particle_type',
            TypeParameterDict(OnlyTypes(float,
                                        postprocess=self._flag_move_size_update,
                                        allow_none=True),
                              len_keys=1))
        self._typeparam_dict = {
            'max_translation_move': t_moves,
            'max_rotation_move': r_moves
        }

        # This is a bit complicated because we are having to ensure that we keep
        # the list of tunables and the solver updated with the changes to
        # attributes. However, these are simply forwarding a change along.
        param_dict = ParameterDict(
            moves=OnlyIf(to_type_converter([OnlyFrom(['a', 'd'])]),
                         postprocess=self._update_moves),
            types=OnlyIf(to_type_converter([str]),
                         postprocess=self._update_types,
                         allow_none=True),
            target=OnlyTypes(float, postprocess=self._target_postprocess),
            solver=SolverStep)

        self._param_dict.update(param_dict)
        self.target = target
        self.solver = solver
        self.moves = moves
        self.types = types

        self.max_rotation_move.default = max_rotation_move
        self.max_translation_move.default = max_translation_move
        if types is not None:
            self._update_tunables(new_moves=moves, new_types=types)

    def attach(self, simulation):
        if not isinstance(simulation.operations.integrator, HPMCIntegrator):
            raise RuntimeError(
                "MoveSizeTuner can only be used in HPMC simulations.")
        particle_types = simulation.state.particle_types
        if self.types is None:
            self.types = particle_types
        if not all(t in particle_types for t in self.types):
            raise RuntimeError(
                "Invalid particle type found specified types for tuning.")
        self._update_tunables(new_moves=self.moves, new_types=self.types)
        self._update_tunables_attr('integrator',
                                   simulation.operations.integrator)
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
        return self._tuned >= 2

    def detach(self):
        self._update_tunables_attr('integrator', None)
        self._is_attached = False

    def act(self, timestep=None):
        """Tune move sizes.

        Args:
            timestep (`int`, optional): Current simulation timestep.
        """
        if self._is_attached:
            # update maximum move sizes
            if self._update_move_sizes:
                for tunable in self._tunables:
                    if tunable.attr == 'a':
                        max_move_size = self.max_rotation_move[tunable.type]
                    else:
                        max_move_size = self.max_translation_move[tunable.type]
                    tunable.domain = (self._min_move_size, max_move_size)

            tuned = self.solver.solve(self._tunables)
            self._tuned = self._tuned + 1 if tuned else 0

    def _update_tunables(self, *, new_moves=tuple(), new_types=tuple()):
        tunables = self._tunables

        # First filter out any move size tune definitions that don't match
        # the new specification.
        def filter_tunables(tunable):
            return ((new_moves is None or tunable.attr in new_moves)
                    and (new_types is None or tunable.type in new_types))

        self._tunables = list(filter(filter_tunables, tunables))
        tune_definitions = set(self._tunables)

        # Add any move size tune definitions that are required by the new
        # specification.
        for move in new_moves:
            for new_type in new_types:
                if move == 'a':
                    max_move_size = self.max_rotation_move[new_type]
                else:
                    max_move_size = self.max_translation_move[new_type]
                move_definition = _MoveSizeTuneDefinition(
                    move, new_type, self.target,
                    (self._min_move_size, max_move_size))
                if move_definition not in tune_definitions:
                    self._tunables.append(move_definition)

    def _update_tunables_attr(self, attr, value):
        for tunable in self._tunables:
            setattr(tunable, attr, value)

    def _target_postprocess(self, target):
        if not (0 <= target <= 1):
            raise ValueError(f"target {target} should be between 0 and 1.")

        self._update_tunables_attr('target', target)
        self._tuned = 0
        return target

    def _update_moves(self, value):
        self._update_tunables(new_moves=value)
        self._tuned = 0
        return value

    def _update_types(self, value):
        self._update_tunables(new_types=value)
        self._tuned = 0
        return value

    def _flag_move_size_update(self, value):
        self._update_move_sizes = True
        return value


class MoveSize(_InternalCustomTuner):
    """Tunes HPMCIntegrator move sizes to targeted acceptance rate.

    Direct instantiation of this class requires a `hoomd.tune.SolverStep` that
    determines how move sizes are updated. This class also provides class
    methods to create a `MoveSize` tuner with built-in solvers; see
    `MoveSize.secant_solver` and `MoveSize.scale_solver`.

    Args:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        moves (list[str]): A list of types of moves to tune. Available options
            are ``'a'`` and ``'d'``.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.SolverStep`): A solver that tunes move sizes to
            reach the specified target.
        types (list[str]): A list of string particle types to tune the move
            size for, defaults to None which upon attaching will tune all types
            in the system currently.
        max_translation_move (float): The maximum value of a translational move
            size to attempt :math:`[\\mathrm{length}]`.
        max_rotation_move (float): The maximum value of a rotational move size
            to attempt.

    Attributes:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        moves (list[str]): A list of types of moves to tune. Available options
            are ``'a'`` and ``'d'``.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (hoomd.tune.SolverStep): A solver that tunes move sizes to
            reach the specified target.
        types (list[str]): A list of string particle
            types to tune the move size for, defaults to None which upon
            attaching will tune all types in the system currently.
        max_translation_move (float): The maximum value of a translational move
            size to attempt :math:`[\\mathrm{length}]`.
        max_rotation_move (float): The maximum value of a rotational move size
            to attempt.

    Note:
        Limiting the maximum move sizes can lead to the inability to converge to
        the desired acceptance rate. Also, not limiting the move size can lead
        to move sizes that require the use of multiple periodic images to check
        for overlaps, especially in low density systems since the acceptance
        rate tends towards 1. Therefore, it is recommended to pick a moderate
        maximum move size for at least the translational moves to prevent
        requiring checking periodic images.

    Note:
        In systems containing disparate particle shapes and/or sizes, move
        sizes for the different types should be tuned independently so that the
        acceptances rates for the different particles are each near the target
        acceptance ratio. Otherwise, the *global* acceptance ratio, a weighted
        average of the acceptance ratios for each individual particle type, will
        approach the target value, while the per-type acceptance ratios may not
        be close to the target value. This requires setting the ``types``
        attribute to be one type at a time while setting the
        ``ignore_statistics`` flag of the shape property of the HPMC integrator
        for all other types to ``True``.

    """
    _internal_class = _InternalMoveSize

    @classmethod
    def scale_solver(cls,
                     trigger,
                     moves,
                     target,
                     types=None,
                     max_translation_move=None,
                     max_rotation_move=None,
                     max_scale=2.,
                     gamma=1.,
                     tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.ScaleSolver`.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            moves (list[str]): A list of types of moves to tune. Available
                options are ``'a'`` and ``'d'``.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (list[str]): A list of string particle types to tune the
                move size for, defaults to None which upon attaching will tune
                all types in the system currently.
            max_translation_move (float): The maximum value of a translational
                move size to attempt :math:`[\\mathrm{length}]`.
            max_rotation_move (float): The maximum value of a rotational move
                size to attempt.
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
        return cls(trigger, moves, target, solver, types, max_translation_move,
                   max_rotation_move)

    @classmethod
    def secant_solver(cls,
                      trigger,
                      moves,
                      target,
                      types=None,
                      max_translation_move=None,
                      max_rotation_move=None,
                      gamma=0.8,
                      tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.SecantSolver`.

        This solver can be faster than `hoomd.tune.ScaleSolver`, but depending
        on the system slightly less stable. In general, with the default value
        of gamma this should not be a problem.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            moves (list[str]): A list of types of moves to tune. Available
                options are ``'a'`` and ``'d'``.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (list[str]): A list of string
                particle types to tune the move size for, defaults to None which
                upon attaching will tune all types in the system currently.
            max_translation_move (float): The maximum value of a translational
                move size to attempt, defaults to ``None`` which represents no
                maximum move size :math:`[\\mathrm{length}]`.
            max_rotation_move (float): The maximum value of a rotational move
                size to attempt, defaults to ``None`` which represents no
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
            convergence and can slow it down. In addition, large values of
            ``gamma`` can make the solver unstable, especially when tuning
            frequently.
        """
        solver = SecantSolver(gamma, tol)
        return cls(trigger, moves, target, solver, types, max_translation_move,
                   max_rotation_move)
