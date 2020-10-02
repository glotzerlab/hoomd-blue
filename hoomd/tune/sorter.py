from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyType
from hoomd.operation import Tuner
from hoomd.trigger import Trigger
from hoomd import _hoomd
import hoomd
from math import log2, ceil


def to_power_of_two(value):
    return int(2. ** ceil(log2(value)))


def natural_number(value):
    try:
        if value < 1:
            raise ValueError("Expected positive integer.")
        else:
            return value
    except TypeError:
        raise ValueError("Expected positive integer.")


class ParticleSorter(Tuner):
    def __init__(self, trigger=200, grid=None):
        self._param_dict = ParameterDict(
            trigger=Trigger,
            grid=OnlyType(int,
                          postprocess=lambda x: int(to_power_of_two(x)),
                          preprocess=natural_number,
                          allow_none=True)
        )
        self.trigger = trigger
        self.grid = grid

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_cls = getattr(_hoomd, 'SFCPackTunerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'SFCPackTuner')
        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def, self.trigger)
        super()._attach()
