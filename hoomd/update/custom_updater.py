# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomUpdater."""

from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Updater


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


class CustomUpdater(CustomOperation, _UpdaterProperty, Updater):
    """User-defined updater.

    Args:
        action (hoomd.custom.Action): The action to call.
        trigger (hoomd.trigger.Trigger): Select the timesteps to call the
          action.

    `CustomUpdater` is a `hoomd.operation.Updater` that wraps a user-defined
    `hoomd.custom.Action` object so the action can be added to a
    `hoomd.Operations` instance for use with `hoomd.Simulation` objects.

    Updaters modify the system state.

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.tune.CustomTuner`

        `hoomd.write.CustomWriter`
    """
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'


class _InternalCustomUpdater(_InternalCustomOperation, Updater):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
    _operation_func = "update"

    def update(self, timestep):
        return self._action.act(timestep)
