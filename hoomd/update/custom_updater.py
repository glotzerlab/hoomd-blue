from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import _CustomAction, _InternalCustomAction


class _CustomUpdaterAction(_CustomAction):
    def update(self, timestep):
        return self.act(timestep)


class _InternalCustomUpdaterAction(_InternalCustomAction):
    def update(self, timestep):
        return self.act(timestep)


class _CustomUpdater(_CustomOperation):
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


class _InternalCustomUpdater(_InternalCustomOperation):
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
