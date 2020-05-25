from hoomd.operation import _Tuner
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.trigger import Trigger
from hoomd import _hoomd
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


class ParticleSorter(_Tuner):
    def __init__(self, trigger=200, grid=None):
        self._param_dict = ParameterDict(
            trigger=Trigger,
            grid=OnlyType(int,
                          postprocess=lambda x: int(to_power_of_two(x)),
                          preprocess=natural_number,
                          allow_none=True)
        )
        self.trigger = trigger
        self.grid = None

    def attach(self, simulation):
        if simulation.device.mode == 'gpu':
            cpp_cls = getattr(_hoomd, 'SFCPackTunerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'SFCPackTuner')
        self._cpp_obj = cpp_cls(simulation.state._cpp_sys_def, self.trigger)
        super().attach(simulation)
