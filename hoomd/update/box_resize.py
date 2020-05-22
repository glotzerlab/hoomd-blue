from hoomd.operation import _Updater
from hoomd.box import Box
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.variant import Variant, Power
from hoomd import _hoomd


def box_preprocessing(box):
    if isinstance(box, Box):
        return box
    else:
        try:
            return Box.from_box(box)
        except Exception:
            raise ValueError(
                "{} is not convertible into a hoomd.Box object. "
                "using hoomd.Box.from_box".format(box))


class BoxResize(_Updater):
    """Resizes the box between an initial and final box.

    When part of a `Simulation` ``updater`` list, this object will resize the
    box between the initial and final boxes passed. The behavior is a linear
    interpolation between the initial and final boxes where the minimum of the
    variant is the initial box and the maximum is the final box. All
    values between the minimum and maximum result in a box that is the
    interpolation of the three lengths and tilt factors of the initial and final
    boxes.

    Note:
        The passed `Variant` must be well behaved (i.e. it must have a
        true minimum and maximum) or the behavior of the updater is undefined.

    Note:
        Currently for MPI simulations the rescaling of particles does not work
        properly.

    Args:
        initial_box (hoomd.Box): The box associated with the minimum of the
            passed variant.
        final_box (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        scale_particles (bool): Whether to scale particles to the new box
            dimensions when the box is resized.
    """
    def __init__(self, initial_box, final_box,
                 variant, trigger, scale_particles=True):
        params = ParameterDict(
            initial_box=OnlyType(Box, preprocess=box_preprocessing),
            final_box=OnlyType(Box, preprocess=box_preprocessing),
            variant=Variant,
            scale_particles=bool)
        params['initial_box'] = initial_box
        params['final_box'] = final_box
        params['variant'] = variant
        params['trigger'] = trigger
        params['scale_particles'] = scale_particles
        self._param_dict.update(params)
        super().__init__(trigger)

    def attach(self, simulation):
        self._cpp_obj = _hoomd.BoxResizeUpdater(simulation.state._cpp_sys_def,
                                                self.initial_box,
                                                self.final_box,
                                                self.variant)
        super().attach(simulation)

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
        """
        if self.is_attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None

    @classmethod
    def linear_volume(cls, initial_box, final_box,
                      t_start, t_ramp,
                      trigger, scale_particles=True):
        """Create a BoxResize object that will scale volume/area linearly.

        This uses a :class:`hoomd.variant.Power` variant under the hood.

        Args:
            initial_box (hoomd.Box): The box associated with *t_start*.
            final_box (hoomd.Box): The box associated with *t_start + t_ramp*.
            t_start (int): The timestep to start the volume ramp.
            t_ramp (int): The length of the volume ramp
            trigger (hoomd.trigger.Trigger): The trigger to activate this
                updater.  scale_particles (bool): Whether to scale particles to
                the new box dimensions when the box is resized.
            scale_particles (bool): Whether to scale particles to the new box
                dimensions when the box is resized.

        Returns:
            box_resize (hoomd.update.BoxResize): Returns a ``BoxResize`` object
            that will scale between the boxes linearly in volume (area for 2D).
        """
        var = Power(initial_box.volume, final_box.volume,
                    1 / final_box.dimensions, t_start, t_size)
        return cls(initial_box, final_box, var, trigger, scale_particles)
