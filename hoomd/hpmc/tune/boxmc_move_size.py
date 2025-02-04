# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxMCMoveSize."""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import (OnlyFrom, OnlyTypes, OnlyIf,
                                      to_type_converter)

from hoomd.tune import _InternalCustomTuner
from hoomd.tune import ScaleSolver, SecantSolver

from hoomd.hpmc.integrate import HPMCIntegrator
from hoomd.hpmc.tune import mc_move_tune


class _MoveSizeTuneDefinition(mc_move_tune._MCTuneDefinition):
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
        splits = attr.split("_")
        self.attr = splits[0]
        if len(splits) > 1:
            self.index = ["x", "y", "z"].index(splits[1])
        else:
            self.index = -1

        self.boxmc = boxmc
        super().__init__(target, domain)

    def get_ratio(self):
        if self.attr.startswith("l"):
            attr = "volume"
        else:
            attr = self.attr
        return getattr(self.boxmc, attr + "_moves")

    def _get_x(self):
        x = getattr(self.boxmc, self.attr)["delta"]
        if self.index >= 0:
            return x[self.index]
        return x

    def _set_x(self, value):
        if self.index < 0:
            getattr(self.boxmc, self.attr)["delta"] = value
            return
        new_value = list(getattr(self.boxmc, self.attr)["delta"])
        new_value[self.index] = value
        getattr(self.boxmc, self.attr)["delta"] = new_value

    def __hash__(self):
        return hash((self.attr, self._target, self._domain))

    def __eq__(self, other):
        return (self.attr == other.attr and self._target == other._target
                and self._domain == other._domain)


class _InternalBoxMCMoveSize(mc_move_tune._TuneMCMove):
    """Internal class for the box volume tuner."""

    def __init__(
        self,
        boxmc,
        moves,
        target,
        solver,
        max_move_size=None,
    ):
        super().__init__(target, solver)
        # Flags for knowing when to update classes attributes
        self._update_move_sizes = False
        self._should_update_tunables = False

        # This is a bit complicated because we are having to ensure that we keep
        # the list of tunables and the solver updated with the changes to
        # attributes. However, these are simply forwarding a change along.
        params = ParameterDict(
            boxmc=hoomd.hpmc.update.BoxMC,
            moves=[
                OnlyFrom(_MoveSizeTuneDefinition.acceptable_attrs,
                         postprocess=self._flag_new_tunables)
            ],
            max_move_size=OnlyIf(
                to_type_converter({
                    attr:
                        OnlyTypes(float,
                                  allow_none=True,
                                  postprocess=self._flag_move_size_update)
                    for attr in _MoveSizeTuneDefinition.acceptable_attrs
                }),))
        params["boxmc"] = boxmc
        params["moves"] = moves
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
        super().attach(simulation)

    def act(self, timestep=None):
        """Tune move sizes.

        Args:
            timestep (`int`, optional): Current simulation timestep.
        """
        # update maximum move sizes
        if self._should_update_tunables:
            self._update_tunables()
        if self._update_move_sizes:
            for tunable in self._tunables:
                if tunable.index >= 0:
                    direction = ("x", "y", "z")[tunable.index]
                    move_size_attr = f"{tunable.attr}_{direction}"
                else:
                    move_size_attr = tunable.attr
                max_move_size = self.max_move_size[move_size_attr]
                tunable.domain = (self._min_move_size, max_move_size)
        super().act(timestep)

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

    def _flag_move_size_update(self, value):
        self._update_move_sizes = True
        return value

    def _flag_new_tunables(self, value):
        self._should_update_tunables = True
        return value
        self._param_dict._cpp_obj = 5


class BoxMCMoveSize(_InternalCustomTuner):
    """Tunes ``BoxMC`` move sizes to targeted acceptance rate.

    Tip:
        For most common creation of a `BoxMCMoveSize` tuner see
        `BoxMCMoveSize.secant_solver` and `BoxMCMoveSize.scale_solver`
        respectively.

    The tuner will continue tuning despite being ``tuned``. Thus, if simulation
    conditions change the move sizes will continue to change and the tuner will
    no longer be ``tuned``. The changes to the move size are completely
    controlled by the given `hoomd.tune.RootSolver` instance. See the
    doumentation at `hoomd.tune` for more information.

    Warning:
        The tuner should be removed from the simulation once tuned to prevent
        invalid results due to the breaking of balance.

    Args:
        trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when to
            run the tuner.
        boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC` object to
            tune.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'volume', 'aspect', 'shear_{x,y,z}', and 'length_{x,y,z}' where
            brackets denote multiple options. For shear and length moves each
            dimension is tuned independently.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (`hoomd.tune.RootSolver`): A solver that tunes move sizes
            to reach the specified target.
        max_move_size (`dict` [`str`, `float` ], optional): The maximum volume
            move size to attempt for each move time. See the available moves in
            the `moves` attribute documentation. Defaults to no maximum ``None``
            for each move type.

    Attributes:
        trigger (hoomd.trigger.Trigger): ``Trigger`` to determine when to run
            the tuner.
        boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC` object to
            tune.
        moves (list[str]): A list of types of moves to tune. Available options
            are 'volume', 'aspect', 'shear_{x,y,z}', and 'length_{x,y,z}' where
            brackets denote multiple options. For shear and length moves each
            dimension is tuned independently.
        target (float): The acceptance rate for trial moves that is desired. The
            value should be between 0 and 1.
        solver (hoomd.tune.RootSolver): A solver that tunes move sizes to reach
            the specified target.
        max_move_size (float): The maximum volume move size
            to attempt for each move time. See the available moves in the
            `moves` attribute documentation.

    Warning:
        Over-limiting the maximum move sizes can lead to the inability to
        converge to the desired acceptance rate.

    Warning:
        Since each dimension of length and shear moves are tuned independently
        but the acceptance statistics are collected collectively, the reachable
        target acceptance rates is limited by the other dimensions.
    """
    _internal_class = _InternalBoxMCMoveSize
    _wrap_methods = ("tuned",)

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
        """Create a `BoxMCMoveSize` tuner with a `hoomd.tune.ScaleSolver`.

        Args:
            trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when
                to run the tuner.
            boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC`
                object to tune.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'volume', 'aspect', 'shear_{x,y,z}', and
                'length_{x,y,z}' where brackets denote multiple options. For
                shear and length moves each dimension is tuned independently.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
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
        """Create a `BoxMCMoveSize` tuner with a `hoomd.tune.SecantSolver`.

        This solver can be faster than `hoomd.tune.ScaleSolver`, but depending
        on the system slightly less stable. In general, with the default value
        of gamma this should not be a problem.

        Args:
            trigger (hoomd.trigger.trigger_like): ``Trigger`` to determine when
                to run the tuner.
            boxmc (hoomd.hpmc.update.BoxMC): The `hoomd.hpmc.update.BoxMC`
                object to tune.
            moves (list[str]): A list of types of moves to tune. Available
                options are 'volume', 'aspect', 'shear_{x,y,z}', and
                'length_{x,y,z}' where brackets denote multiple options. For
                shear and length moves each dimension is tuned independently.
            target (float): The acceptance rate for trial moves that is desired.
                The value should be between 0 and 1.
            max_move_size (float): The maximum value of a volume
                move size to attempt, defaults to ``None`` which represents no
                maximum move size.
            gamma (float): The value of gamma to pass through to
                `hoomd.tune.SecantSolver`. Controls the size of corrections to
                the move size (smaller values increase stability). Should be
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
        return cls(trigger, boxmc, moves, target, solver, max_move_size)
