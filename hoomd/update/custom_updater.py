from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import _CustomAction, _InternalCustomAction


class _UpdateMethod:
    def update(self, timestep):
        return self.act(timestep)


class _CustomUpdaterAction(_CustomAction, _UpdateMethod):
    pass


class _InternalCustomUpdaterAction(_InternalCustomAction, _UpdateMethod):
    pass


class _UpdaterProperty:
    @property
    def updater(self):
        return self._action

    @updater.setter
    def updater(self, updater):
        if isinstance(updater, _CustomAction):
            self._action = updater
        else:
            raise ValueError("updater must be an instance of _CustomAction")


class _CustomUpdater(_CustomOperation, _UpdaterProperty):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'


class _InternalCustomUpdater(_InternalCustomOperation, _UpdaterProperty):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
