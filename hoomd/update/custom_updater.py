# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomUpdater.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    class ExampleAction(hoomd.custom.Action):
        def act(self, timestep):
            pass

    custom_action = ExampleAction()
"""

from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Updater
import warnings


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
        trigger (hoomd.trigger.trigger_like): Select the timesteps to call the
          action.

    `CustomUpdater` is a `hoomd.operation.Updater` that wraps a user-defined
    `hoomd.custom.Action` object so the action can be added to a
    `hoomd.Operations` instance for use with `hoomd.Simulation` objects.

    Updaters modify the system state.

    .. rubric:: Example:

    .. code-block:: python

            custom_updater = hoomd.update.CustomUpdater(
                action=custom_action,
                trigger=hoomd.trigger.Periodic(1000))
            simulation.operations.updaters.append(custom_updater)

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
        """
        .. deprecated:: 4.5.0

            Use `Simulation` to call the operation.
        """
        warnings.warn(
            "`_InternalCustomUpdater.update` is deprecated,"
            "use `Simulation` to call the operation.", FutureWarning)
