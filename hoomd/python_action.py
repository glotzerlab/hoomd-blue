from hoomd.operation import _TriggeredOperation
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyType
from hoomd.trigger import Trigger
from hoomd.util import trigger_preprocessing
from hoomd import _hoomd


class _PythonAction(_TriggeredOperation):
    def __init__(self, action, trigger=1):
        self._action = action
        param_dict = ParameterDict(
            trigger=OnlyType(Trigger, preprocess=trigger_preprocessing))
        param_dict['trigger'] = trigger
        self._param_dict.update(param_dict)

    def attach(self, simulation):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            simulation.state._cpp_sys_def, self._action)

        super.attach(simulation)
        self._action.attach(simulation)

    def act(self, timestep):
        if self.is_attached:
            getattr(self._cpp_obj, self._cpp_action)(timestep)
        else:
            pass


class _InternalPythonAction(_PythonAction):
    _use_default_setattr = {'_action'}

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            try:
                return getattr(self._action, attr)
            except AttributeError:
                raise AttributeError(
                    "{} object has no attribute {}".format(type(self), attr))

    def _setattr_hook(self, attr, value):
        if attr in dir(self):
            object.__setattr__(self, attr, value)
        else:
            setattr(self._action, attr, value)
