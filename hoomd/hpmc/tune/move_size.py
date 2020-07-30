from hoomd.custom import _InternalAction
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyFrom, OnlyType, OnlyIf, to_type_converter
from hoomd.tune import _InternalCustomTuner
from hoomd.tune.attr_tuner import (
    _TuneDefinition, Solver, ScaleSolver, SecantSolver)
from hoomd.hpmc.integrate import _HPMCIntegrator


class _MoveSizeTuneDefinition(_TuneDefinition):
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
    def __init__(self, moves, target, solver, types=None, max_move_size=None):
        def target_postprocess(target):
            def is_fraction(value):
                if 0 <= value <= 1:
                    return value
                raise ValueError(
                    "Value {} should be between 0 and 1.".format(value))

            self._update_tunables_attr('target', is_fraction(target))
            return target

        def max_move_size_postprocess(move_size):
            self._update_tunables_attr( 'domain', (0, move_size))
            return move_size

        def update_moves(value):
            self._update_tunables(new_moves=value)
            return value

        def update_types(value):
            self._update_tunables(new_types=value)
            return value

        self._tunables = []
        self._tuned = False
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
            '_integrator', simulation.operations.integrator)
        self._attached = True

    @property
    def is_attached(self):
        return self._attached

    @property
    def tuned(self):
        return self._tuned

    def detach(self):
        self._update_tunables_attr('_integrator', None)
        self._attached = False

    def act(self, timestep):
        if not self._tuned:
            self._tuned = self.solver.solve(self._tunables)

    def _update_tunables(self, *, new_moves=tuple(), new_types=tuple()):
        tunables = self._tunables
        tune_definitions = set(self._tunables)

        # First filter out any move size tune definitions that don't match
        # the new specification.
        def filter_tunables(tunable):
            return ((new_moves is None or tunable._attr in new_moves)
                    and (new_types is None or tunable._type in new_types))

        self._tunables = list(filter(filter_tunables, tunables))

        # Add any move size tune definitions that are required by the new
        # specification.
        for move in new_moves:
            for new_type in new_types:
                move_definition = _MoveSizeTuneDefinition(
                    move, new_type, self.target, (0, self.max_move_size))
                if move_definition not in tune_definitions:
                    self._tunables.append(move_definition)

    def _update_tunables_attr(self, attr, value):
        for tunable in self._tunables:
            setattr(tunable, attr, value)


class MoveSize(_InternalCustomTuner):
    _internal_class = _InternalMoveSize

    @classmethod
    def scaled_solver(cls, trigger, moves, target,
                      types=None, max_move_size=None,
                      max_scale=2., gamma=1., tol=1e-2):
        solver = ScaleSolver(max_scale, gamma, tol)
        return cls(trigger, moves, target, solver, types, max_move_size)

    @classmethod
    def secant_solver(cls, trigger, moves, target, types=None,
                      max_move_size=None, gamma=0.9, tol=1e-2):
        solver = SecantSolver(gamma, tol)
        return cls(trigger, moves, target, solver, types, max_move_size)
