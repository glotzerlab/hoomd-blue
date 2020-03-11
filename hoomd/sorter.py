from hoomd.operation import _Tuner
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType, OnlyTypeValidNone
from hoomd.util import trigger_preprocessing
from hoomd.trigger import Trigger
from hoomd import _hoomd


class ParticleSorter(_Tuner):
    def __init__(self, trigger=200, grid=None):
        self._param_dict = ParameterDict(
            trigger=OnlyType(Trigger, preprocess=trigger_preprocessing),
            grid=OnlyTypeValidNone(int))
        self.trigger = trigger
        self.grid = None

    def attach(self, simulation):
        if simulation.device.mode == 'gpu':
            cpp_cls = getattr(_hoomd, 'SFCPackTunerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'SFCPackTuner')
        self._cpp_obj = cpp_cls(simulation.state._cpp_sys_def, self.trigger)
