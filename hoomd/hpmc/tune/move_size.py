from hoomd.custom import _InternalAction
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyFrom, OnlyType, OnlyIf, to_type_converter
from hoomd.tune import _InternalCustomTuner
from hoomd.tune.attr_tuner import (
    _TuneDefinition, Solver, ScaleSolver, SecantSolver)
from hoomd.hpmc.integrate import _HPMCIntegrator


class _MoveSizeTuneDefinition(_TuneDefinition):
    """Encapsulates getting the acceptance rate and getting/setting move size.

    This class should only be used for the _InternalMoveSize class to tune HPMC
    move sizes. For this class 'x' is the move size and 'y' is the acceptance
    rate.
    """
    _attr_acceptance = {
        'a': 'rotate_moves',
        'd': 'translate_moves'
    }

    def __init__(self, attr, type, target, domain=None):
        if attr not in self._attr_acceptance:
            raise ValueError("Only {} are allowed as tunable "
                             "attributes.".format(self._available_attrs))
        self.attr = attr
        self.type = type
        self.integrator = None
        self.previous_ratio = None
        self.previous_total = None
        super().__init__(target, domain)

    def _get_y(self):
        current_ratio = getattr(self.integrator,
                                self._attr_acceptance[self.attr])
        current_total = sum(current_ratio)
        # We return the target when no moves are recorded since we don't want
        # the move size to be updated. This is "hackish", but there is no right
        # answer here since there is no concept of an acceptance_rate with no
        # trial moves. We could error, but I think this make the class easier to
        # use for users.
        if current_total == 0:
            return self._target
        # If no more trial moves have been recorded return previous
        # acceptance_rate. In general, this conditional should not be true, if
        # it is true different solvers may error, but this is the most natural
        # solution, I could think of.
        if (self.previous_total is not None
                and self.previous_total == current_total):
            return self.previous_ratio[0] / self.previous_total

        if self.previous_ratio is None or self.previous_total > current_total:
            acceptance_rate = current_ratio[0] / current_total
        else:
            acceptance_rate = ((current_ratio[0] - self.previous_ratio[0])
                               / (current_total - self.previous_total))
        # We store the previous information becuase this lets us find the
        # acceptance rate since this has last been called which allows for us to
        # disregard the information before the last tune.
        self.previous_ratio = current_ratio
        self.previous_total = current_total
        return acceptance_rate

    def _get_x(self):
        return getattr(self.integrator, self.attr)[self.type]

    def _set_x(self, value):
        getattr(self.integrator, self.attr)[self.type] = value

    def __hash__(self):
        return hash((self.attr, self.type, self._target, self._domain))

    def __eq__(self, other):
        return (self.attr == other.attr and
                self.type == other.type and
                self._target == other._target and
                self._domain == other._domain)


class _InternalMoveSize(_InternalAction):
    """Internal class for the MoveSize tuner."""
    _min_move_size = 1e-7

    def __init__(self, moves, target, solver, types=None, max_move_size=None):
        def target_postprocess(target):
            def is_fraction(value):
                if 0 <= value <= 1:
                    return value
                raise ValueError(
                    "Value {} should be between 0 and 1.".format(value))

            self._update_tunables_attr('target', is_fraction(target))
            self._tuned = 0
            return target

        def max_move_size_postprocess(move_size):
            self._update_tunables_attr(
                'domain', (self._min_move_size, move_size))
            return move_size

        def update_moves(value):
            self._update_tunables(new_moves=value)
            self._tuned = 0
            return value

        def update_types(value):
            self._update_tunables(new_types=value)
            self._tuned = 0
            return value

        self._tunables = []
        # A counter when tuned reaches 1 it means that the tuner has reported
        # being tuned one time in a row. However, as the first run of the tuner
        # is likely at timestep 0 which means that the counters are (0, 0) and
        # _MoveSizeTuneDefinition returns y == target for that case, we need two
        # rounds of tuning to be sure that we have converged. Since, in general,
        # solvers do not do much if any work on already tuned tunables, this is
        # not a performance problem.
        self._tuned = 0
        self._attached = False
        # This is a bit complicated because we are having to ensure that we keep
        # the list of tunables and the solver updated with the changes to
        # attributes. However, these are simply forwarding a change along.
        param_dict = ParameterDict(
            moves=OnlyIf(to_type_converter([OnlyFrom(['a', 'd'])]),
                         postprocess=update_moves),
            types=OnlyIf(to_type_converter([str]),
                         postprocess=update_types,
                         allow_none=True),
            target=OnlyType(float, postprocess=target_postprocess),
            max_move_size=OnlyType(
                float, allow_none=True,
                postprocess=max_move_size_postprocess
            ),
            solver=Solver
        )

        param_dict['moves'] = moves
        param_dict['types'] = types
        param_dict['max_move_size'] = max_move_size
        param_dict['target'] = target
        param_dict['solver'] = solver
        self._param_dict.update(param_dict)
        if types is not None:
            self._update_tunables(new_moves=moves, new_types=types)

    def attach(self, simulation):
        if not isinstance(simulation.operations.integrator, _HPMCIntegrator):
            raise RuntimeError(
                "MoveSizeTuner can only be used in HPMC simulations.")
        particle_types = simulation.state.particle_types
        if self.types is None:
            self.types = particle_types
        if not all(t in particle_types for t in self.types):
            raise RuntimeError(
                "Invalid particle type found specified types for tuning.")
        self._update_tunables(new_moves=self.moves, new_types=self.types)
        self._update_tunables_attr(
            'integrator', simulation.operations.integrator)
        self._attached = True

    @property
    def is_attached(self):
        """bool: Whether or not the tuner is attached to a simulation."""
        return self._attached

    @property
    def tuned(self):
        """bool: Whether or not the move sizes are considered tuned.

        Can be set to False.
        """
        return self._tuned == 2

    @tuned.setter
    def tuned(self, value):
        if not value:
            self._tuned = 0
        else:
            raise ValueError("Cannot attempt to set tuned to True. "
                             "Detach operation instead.")

    def detach(self):
        self._update_tunables_attr('integrator', None)
        self._attached = False

    def act(self, timestep=None):
        """Tune move sizes.

        Args:
            timestep (:obj:`int`, optional): Current simulation timestep. Is
                currently ignored.
        """
        if not self.tuned and self.is_attached:
            tuned = self.solver.solve(self._tunables)
            if tuned:
                self._tuned += 1
            else:
                self._tuned = 0

    def _update_tunables(self, *, new_moves=tuple(), new_types=tuple()):
        tunables = self._tunables
        tune_definitions = set(self._tunables)

        # First filter out any move size tune definitions that don't match
        # the new specification.
        def filter_tunables(tunable):
            return ((new_moves is None or tunable.attr in new_moves) and
                    (new_types is None or tunable.type in new_types))

        self._tunables = list(filter(filter_tunables, tunables))

        # Add any move size tune definitions that are required by the new
        # specification.
        for move in new_moves:
            for new_type in new_types:
                move_definition = _MoveSizeTuneDefinition(
                    move, new_type, self.target,
                    (self._min_move_size, self.max_move_size))
                if move_definition not in tune_definitions:
                    self._tunables.append(move_definition)

    def _update_tunables_attr(self, attr, value):
        for tunable in self._tunables:
            setattr(tunable, attr, value)


class MoveSize(_InternalCustomTuner):
    """Tunes HPMCIntegrator move sizes to targeted acceptance rate.

    For most common creation of a `MoveSize` tuner see `MoveSize.secant_solver`
    and `MoveSize.scale_solver` respectively.

    Args:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'a' and 'd'.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.Solver`): A solver that tunes move sizes to reach
            the specified target.
        types (:obj:`list` [:obj:`str`], optional): A list of string particle
            types to tune the move size for, defaults to None which upon
            attaching will tune all types in the system currently.
        max_move_size (:obj:`float`, optional): The max value of move size to
            attempt.

    Attributes:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'a' and 'd'.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.Solver`): A solver that tunes move sizes to reach
            the specified target.
        types (list[str]): A list of string particle
            types to tune the move size for, defaults to None which upon
            attaching will tune all types in the system currently.
        max_move_size (float): The max value of move size to
            attempt.
    """
    _internal_class = _InternalMoveSize

    @classmethod
    def scale_solver(cls, trigger, moves, target,
                      types=None, max_move_size=None,
                      max_scale=2., gamma=1., tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.ScaleSolver`.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'a' and 'd'.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (:obj:`list` [:obj:`str`], optional): A list of string
                particle types to tune the move size for, defaults to None which
                upon attaching will tune all types in the system currently.
            max_move_size (:obj:`float`, optional): The max value of move size
                to attempt.
            gamma (:obj:`float`, optional): The value of gamma to pass through
                to `hoomd.tune.ScaleSolver`. Controls the size of corrections
                to the move size (larger values increase stability while
                increasing convergence time).
            tol (:obj:`float`, optional): The absolute tolerance to allow
                between the current acceptance rate and the target before the
                move sizes are considered tuned. The tolerance should not be too
                much lower than the default as acceptance rates can be fairly
                variable in typical tuning rates.
        """
        solver = ScaleSolver(max_scale, gamma, 'negative', tol)
        return cls(trigger, moves, target, solver, types, max_move_size)

    @classmethod
    def secant_solver(cls, trigger, moves, target, types=None,
                      max_move_size=None, gamma=0.8, tol=1e-2):
        """Create a `MoveSize` tuner with a `hoomd.tune.SecantSolver`.

        This solver can be faster than `hoomd.tune.ScaleSolver`, but depending
        on the application less stable. In general, with the default value of
        gamma this should not be a problem.

        Args:
            trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to
                run the tuner.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'a' and 'd'.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            types (:obj:`list` [:obj:`str`], optional): A list of string
                particle types to tune the move size for, defaults to None which
                upon attaching will tune all types in the system currently.
            max_move_size (:obj:`float`, optional): The max value of move size
                to attempt, default to no max size.
            gamma (:obj:`float`, optional): The value of gamma to pass through
                to `hoomd.tune.SecantSolver`. Controls the size of corrections
                to the move size (smaller values increase stability). Should be
                between 0 and 1, defaults to 0.8.
            tol (:obj:`float`, optional): The absolute tolerance to allow
                between the current acceptance rate and the target before the
                move sizes are considered tuned, defaults to 1e-2. The tolerance
                should not be too much lower than the default as acceptance
                rates can be fairly variable in typical tuning rates.

        Note:
            Increasing ``gamma`` towards 1 does not necessarily speed up
            convergence and can slow it done. In addition, large values of
            ``gamma`` can make the solver unstable especially when tuning
            frequently.
        """
        solver = SecantSolver(gamma, tol)
        return cls(trigger, moves, target, solver, types, max_move_size)
