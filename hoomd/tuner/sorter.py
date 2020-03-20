from hoomd.operation import _Tuner
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType, OnlyTypeValidNone
from hoomd.util import trigger_preprocessing
from hoomd.trigger import Trigger
from hoomd import _hoomd
from math import log2, ceil


def to_power_of_two(value):
    return int(2. ** ceil(log2(value)))


class NaturalNumber(int):
    def __new__(cls, value, *args, **kwargs):
        try:
            if value < 0:
                raise ValueError("Expected positive integer.")
            else:
                return super(cls, cls).__new__(cls, value)
        except TypeError:
            raise ValueError("Expected positive integer.")


class ParticleSorter(_Tuner):
    def __init__(self, trigger=200, grid=None):
        self._param_dict = ParameterDict(
            trigger=OnlyType(Trigger, preprocess=trigger_preprocessing),
            grid=OnlyTypeValidNone(
                NaturalNumber, postprocess=lambda x: int(to_power_of_two(x))))
        self.trigger = trigger
        self.grid = None

    def attach(self, simulation):
        if simulation.device.mode == 'gpu':
            cpp_cls = getattr(_hoomd, 'SFCPackTunerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'SFCPackTuner')
        self._cpp_obj = cpp_cls(simulation.state._cpp_sys_def, self.trigger)
