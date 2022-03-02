# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomTuner."""

from hoomd import _hoomd
from hoomd.operation import Operation
from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Tuner


class _TunerProperty:

    @property
    def tuner(self):
        return self._action

    @tuner.setter
    def tuner(self, tuner):
        if isinstance(tuner, Action):
            self._action = tuner
        else:
            raise ValueError(
                "updater must be an instance of hoomd.custom.Action")


class CustomTuner(CustomOperation, _TunerProperty, Tuner):
    """User-defined tuner.

    Args:
        action (hoomd.custom.Action): The action to call.
        trigger (hoomd.trigger.Trigger): Select the timesteps to call the
          action.

    `CustomTuner` is a `hoomd.operation.Tuner` that wraps a user-defined
    `hoomd.custom.Action` object so the action can be added to a
    `hoomd.Operations` instance for use with `hoomd.Simulation` objects.

    Tuners modify the parameters of other operations to improve performance.
    Tuners may read the system state, but not modify it.

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.update.CustomUpdater`

        `hoomd.write.CustomWriter`
    """
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'

    def _attach(self):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            self._simulation.state._cpp_sys_def, self.trigger, self._action)
        self._action.attach(self._simulation)
        Operation._attach(self)


class _InternalCustomTuner(_InternalCustomOperation, Tuner):
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'
    _operation_func = "tune"

    def _attach(self):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            self._simulation.state._cpp_sys_def, self.trigger, self._action)
        self._action.attach(self._simulation)
        Operation._attach(self)

    def tune(self, timestep):
        return self._action.act(timestep)
