# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxResize."""

from hoomd.operation import Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All


class BoxResize(Updater):
    """Resizes the box between an initial and final box.

    When part of a `hoomd.Simulation` ``updater`` list, this object will resize
    the box between the initial and final boxes passed. The behavior is a linear
    interpolation between the initial and final boxes where the minimum of the
    variant is tagged to `box1` and the maximum is tagged to `box2`. All values
    between the minimum and maximum result in a box that is the interpolation of
    the three lengths and tilt factors of the initial and final boxes.

    Note:
        The passed `Variant` must be bounded (i.e. it must have a true minimum
        and maximum) or the behavior of the updater is undefined.

    Note:
        Currently for MPI simulations the rescaling of particles does not work
        properly in HPMC.

    Args:
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        filter (hoomd.filter.ParticleFilter): The subset of particle positions
            to update.

    Attributes:
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        filter (hoomd.filter.ParticleFilter): The subset of particles to
            update.
    """

    def __init__(self, trigger, box1, box2, variant, filter=All()):
        params = ParameterDict(box1=Box,
                               box2=Box,
                               variant=Variant,
                               filter=ParticleFilter)
        params['box1'] = box1
        params['box2'] = box2
        params['variant'] = variant
        params['trigger'] = trigger
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _hoomd.BoxResizeUpdater(
            self._simulation.state._cpp_sys_def, self.box1._cpp_obj,
            self.box2._cpp_obj, self.variant, group)
        super()._attach()

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
            `None` before the first call to `Simulation.run`.
        """
        if self._attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None

    @staticmethod
    def update(state, box, filter=All()):
        """Immediately scale the particle in the system state to the given box.

        Args:
            state (State): System state to scale.
            box (Box): New box.
            filter (hoomd.filter.ParticleFilter): The subset of particles to
                update.
        """
        group = state._get_group(filter)
        updater = _hoomd.BoxResizeUpdater(state._cpp_sys_def,
                                          state.box._cpp_obj, box._cpp_obj,
                                          Constant(1), group)
        updater.update(state._simulation.timestep)
