from hoomd.operation import Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyType, box_preprocessing
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All


class BoxResize(Updater):
    """Resizes the box between an initial and final box.

    When part of a `Simulation` ``updater`` list, this object will resize the
    box between the initial and final boxes passed. The behavior is a linear
    interpolation between the initial and final boxes where the minimum of the
    variant is tagged to `box1` and the maximum is tagged to
    `box2`. All values between the minimum and maximum result in a box
    that is the interpolation of the three lengths and tilt factors of the
    initial and final boxes.

    Note:
        The passed `Variant` must be bounded (i.e. it must have a true minimum
        and maximum) or the behavior of the updater is undefined.

    Note:
        Currently for MPI simulations the rescaling of particles does not work
        properly in HPMC.

    Args:
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        scale_particles (bool): Whether to scale particles to the new box
            dimensions when the box is resized.

    Attributes:
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        scale_particles (bool): Whether to scale particles to the new box
            dimensions when the box is resized.
    """
    def __init__(self, box1, box2,
                 variant, trigger, scale_particles=All()):
        params = ParameterDict(
            box1=OnlyType(Box, preprocess=box_preprocessing),
            box2=OnlyType(Box, preprocess=box_preprocessing),
            variant=Variant,
            scale_particles=ParticleFilter)
        params['box1'] = box1
        params['box2'] = box2
        params['variant'] = variant
        params['trigger'] = trigger
        # params['scale_particles'] = scale_particles
        self._param_dict.update(params)
        self.scale_particles = scale_particles
        super().__init__(trigger)

    def _attach(self):
        group = self._simulation.state._get_group(self.scale_particles)
        self._cpp_obj = _hoomd.BoxResizeUpdater(
            self._simulation.state._cpp_sys_def,
            self.box1,
            self.box2,
            self.variant,
            group
        )
        super()._attach()

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
        """
        if self._attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None

    @staticmethod
    def update(state, box):
        """Immediately scale the particle in the system state to the given box.

        Args:
            state (State): System state to scale.
            box (Box): New box.
        """
        group = state._get_group(All())
        updater = _hoomd.BoxResizeUpdater(state._cpp_sys_def,
                                          state.box,
                                          box,
                                          Constant(1),
                                          group)
        updater.update(state._simulation.timestep)
