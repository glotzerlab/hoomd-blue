from hoomd.custom_operation import _CustomOperation, _InternalCustomOperation
from hoomd.custom_action import CustomAction


class _UpdaterProperty:
    @property
    def updater(self):
        return self._action

    @updater.setter
    def updater(self, updater):
        if isinstance(updater, CustomAction):
            self._action = updater
        else:
            raise ValueError("updater must be an instance of CustomAction")


class CustomUpdater(_CustomOperation, _UpdaterProperty):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'


class _InternalCustomUpdater(_InternalCustomOperation, _UpdaterProperty):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
