from hoomd.operation import _Updater
from hoomd.box import Box
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.variant import Variant
from hoomd.util import variant_preprocessing
from hoomd._hoomd import BoxResizeUpdater


class BoxResize(_Updater):
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
        self._cpp_obj = BoxResizeUpdater(simulation.state._cpp_sys_def,
                                         self.initial_box, self.final_box,
                                         self.variant)
        super().attach(simulation)

    def get_box(self, timestep):
        if self.is_attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None
