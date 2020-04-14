from hoomd.python_action import _PythonAction, _InternalPythonAction
from hoomd.custom_action import _CustomAction, _InternalCustomAction


class _CustomUpdater(_CustomAction):
    def update(self, timestep):
        return self.act(timestep)


class _InternalCustomUpdater(_InternalCustomAction):
    def update(self, timestep):
        return self.act(timestep)


class _PythonUpdater(_PythonAction):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
    _cpp_action = 'update'

    @property
    def updater(self):
        return self._action

    @updater.setter
    def updater(self, updater):
        if isinstance(updater, _CustomAction):
            self._action = updater
        else:
            raise ValueError("updater must be an instance of _CustomAction")


class _InternalPythonUpdater(_InternalPythonAction):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
    _cpp_action = 'update'

    @property
    def updater(self):
        return self._action

    @updater.setter
    def updater(self, updater):
        if isinstance(updater, _CustomAction):
            self._action = updater
        else:
            raise ValueError("updater must be an instance of _CustomAction")
