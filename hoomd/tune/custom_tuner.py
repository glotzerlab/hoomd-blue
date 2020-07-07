from hoomd import _hoomd
from hoomd.custom import (
    _CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import _Tuner


class _TunerProperty:
    @property
    def updater(self):
        return self._action

    @updater.setter
    def updater(self, updater):
        if isinstance(updater, Action):
            self._action = updater
        else:
            raise ValueError(
                "updater must be an instance of hoomd.custom.Action")


class CustomTuner(_CustomOperation, _TunerProperty, _Tuner):
    """Tuner wrapper for `hoomd.custom.Action` objects.

    For usage see `hoomd.custom._CustomOperation`.
    """
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'

    def attach(self, simulation):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            simulation.state._cpp_sys_def, self.trigger, self._action)
        super().attach(simulation)
        self._action.attach(simulation)


class _InternalCustomTuner(
        _InternalCustomOperation, _TunerProperty, _Tuner):
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'
