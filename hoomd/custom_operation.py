from hoomd.operation import _TriggeredOperation
from hoomd.parameterdicts import ParameterDict
from hoomd.custom_action import CustomAction
from hoomd.typeconverter import OnlyType
from hoomd.trigger import Trigger
from hoomd.util import trigger_preprocessing
from hoomd.logger import LoggerQuantity
from hoomd import _hoomd


class _CustomOperation(_TriggeredOperation):
    """Wrapper for user created ``Action``s.

    A basic wrapper that allows for Python object inheriting from
    :py:class:`hoomd.custom_action.CustomAction` to be attached to a simulation.
    To see how to implement a custom Python ``Action``, look at the
    documentation for :py:class:`hoomd.custom_action.CustomAction`.

    This class also implements a "pass-through" system for attributes.
    Attributes and methods from the passed in `action` will be available
    directly in this class. This does not apply to attributes with these names:
    `trigger` and `_action`.

    Note:
        Due to the pass through no attribute should exist both in
        `hoomd._CustomOperation` and the `hoomd.CustomAction`.

    Note:
        This object should not be instantiated directly.
    """

    _override_setattr = {'_action'}

    @property
    def _cpp_class_name(self):
        """C++ Class to use for attaching."""
        raise NotImplementedError

    def __init__(self, action, trigger=1):
        if not isinstance(action, CustomAction):
            raise ValueError("action must be a subclass of "
                             "hoomd.custom_action.CustomAction.")
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
        if hasattr(self._action, attr):
            setattr(self._action, attr, value)
        else:
            object.__setattr__(self, attr, value)

    def attach(self, simulation):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            simulation.state._cpp_sys_def, self._action)

        super().attach(simulation)
        self._action.attach(simulation)

    def act(self, timestep):
        """Perform the action of the custom action if attached."""
        if self.is_attached:
            self._action.act(timestep)

    @property
    def action(self):
        return self._action


class _InternalCustomOperation(_CustomOperation):
    """Internal class for Python ``Action``s. Offers a streamlined __init__.
    """

    @property
    def _internal_class(self):
        """Internal class to use for the Action of the Operation."""
        raise NotImplementedError

    def __init__(self, trigger, *args, **kwargs):
        super().__init__(self._internal_class(*args, **kwargs), trigger)
        self._export_dict = {key: value.update_cls(self.__class__)
                             for key, value in self._export_dict.items()}

    @property
    def action(self):
        """Prevents the access of action in public API."""
        raise AttributeError
