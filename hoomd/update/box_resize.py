from hoomd.operation import _Updater
from hoomd.box import Box
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.variant import Variant
from hoomd.util import variant_preprocessing
from hoomd import _hoomd


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

    Args:
        initial_box (Box): The box associated with the minimum of the
            passed variant.
        final_box (Box): The box associated with the maximum of the
            passed variant.
        variant (Variant): A variant used to interpolate between
            the two boxes.
        trigger (Trigger): The trigger to activate this updater.
        scale_particles (bool): Whether to scale particles to the new box
            dimensions when the box is resized.

    """
    def __init__(self, initial_box, final_box,
                 variant, trigger=1, scale_particles=True):
        params = ParameterDict(
            initial_box=OnlyType(Box), final_box=OnlyType(Box),
            variant=OnlyType(Variant, variant_preprocessing),
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
