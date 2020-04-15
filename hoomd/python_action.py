from hoomd.operation import _TriggeredOperation
from hoomd.parameterdicts import ParameterDict
from hoomd.custom_action import _CustomAction
from hoomd.typeconverter import OnlyType
from hoomd.trigger import Trigger
from hoomd.util import trigger_preprocessing
from hoomd.logger import LoggerQuantity
from hoomd import _hoomd


class _PythonAction(_TriggeredOperation):
    def __init__(self, action, trigger=1):
        if not issubclass(action, _CustomAction):
            raise ValueError("action must be a subclass of "
                             "hoomd.custom_action._CustomAction.")
        self._action = action
        loggables = list(action.log_quantities)
        if not all(isinstance(l, LoggerQuantity) for l in loggables.values()):
            raise ValueError("Error wrapping {}. All advertised log "
                             "quantities must be of type LoggerQuantity."
                             "".format(action))
        self._export_dict = loggables

        param_dict = ParameterDict(
            trigger=OnlyType(Trigger, preprocess=trigger_preprocessing))
        param_dict['trigger'] = trigger
        self._param_dict.update(param_dict)

    def attach(self, simulation):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            simulation.state._cpp_sys_def, self._action)

        super().attach(simulation)
        self._action.attach(simulation)

    def act(self, timestep):
        if self.is_attached:
            getattr(self._cpp_obj, self._cpp_action)(timestep)
        else:
            pass


class _InternalPythonAction(_PythonAction):
    _use_default_setattr = {'_action'}

    def __init__(self, trigger, *args, **kwargs):
        super().__init__(self._internal_class(*args, **kwargs), trigger)
        self._export_dict = {key: value.update_cls(self.__class__)
                             for key, value in self._export_dict.items()}

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
