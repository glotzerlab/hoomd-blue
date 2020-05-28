from hoomd.custom import (
    _CustomOperation, _InternalCustomOperation, Action)


class _UpdaterProperty:
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


class CustomUpdater(_CustomOperation, _UpdaterProperty):
    """Updater wrapper for `hoomd.custom.Action` objects.

    For usage see `hoomd.custom._CustomOperation`.
    """
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'


class _InternalCustomUpdater(_InternalCustomOperation, _UpdaterProperty):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
