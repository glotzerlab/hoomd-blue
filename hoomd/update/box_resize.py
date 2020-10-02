from hoomd.operation import _Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyType, box_preprocessing
from hoomd.variant import Variant, Power, Constant
from hoomd import _hoomd


class BoxResize(_Updater):
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
                 variant, trigger, scale_particles=True):
        params = ParameterDict(
            box1=OnlyType(Box, preprocess=box_preprocessing),
            box2=OnlyType(Box, preprocess=box_preprocessing),
            variant=Variant,
            scale_particles=bool)
        params['box1'] = box1
        params['box2'] = box2
        params['variant'] = variant
        params['trigger'] = trigger
        params['scale_particles'] = scale_particles
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach(self):
        self._cpp_obj = _hoomd.BoxResizeUpdater(
            self._simulation.state._cpp_sys_def,
            self.box1,
            self.box2,
            self.variant)
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
            box (Box): New box.
        """
        updater = _hoomd.BoxResizeUpdater(state._cpp_sys_def,
                                          state.box,
                                          box,
                                          Constant(1))
        updater.update(state._simulation.timestep)

    @classmethod
    def linear_volume(cls, box1, box2,
                      t_start, t_size,
                      trigger, scale_particles=True):
        """Create a `BoxResize` object that will scale volume/area linearly.

        This uses a :class:`hoomd.variant.Power` variant under the hood.

        Args:
            box1 (hoomd.Box): The box associated with *t_start*.
            box2 (hoomd.Box): The box associated with *t_start + t_size*.
            t_start (int): The timestep to start the volume ramp.
            t_size (int): The length of the volume ramp
            trigger (hoomd.trigger.Trigger): The trigger to activate this
                updater.  scale_particles (bool): Whether to scale particles to
                the new box dimensions when the box is resized.
            scale_particles (bool): Whether to scale particles to the new box
                dimensions when the box is resized.

        Returns:
            hoomd.update.BoxResize: An operation that will scale between
            the boxes linearly in volume (area for 2D).
        """
        box1 = box_preprocessing(box1)
        box2 = box_preprocessing(box2)
        min_ = min(box1.volume, box2.volume)
        max_ = max(box1.volume, box2.volume)
        var = Power(min_, max_, 1 / box2.dimensions, t_start, t_size)
        return cls(box1, box2, var, trigger, scale_particles)
