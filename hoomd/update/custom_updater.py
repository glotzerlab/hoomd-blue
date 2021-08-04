# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

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
    `hoomd.custom.Action` object  so it can be added to the `hoomd.Simulation`'s
    `hoomd.Operations` and called during the run.

    Updaters may modify the system state.

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.tune.CustomTuner`

        `hoomd.write.CustomWriter`
    """
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'


class _InternalCustomUpdater(_InternalCustomOperation, _UpdaterProperty,
                             Updater):
    _cpp_list_name = 'updaters'
    _cpp_class_name = 'PythonUpdater'
