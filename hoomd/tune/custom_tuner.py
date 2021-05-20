# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

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
    """Tuner wrapper for `hoomd.custom.Action` objects.

    For usage see `hoomd.custom.CustomOperation`.
    """
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'

    def _attach(self):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            self._simulation.state._cpp_sys_def, self.trigger, self._action)
        self._action.attach(self._simulation)
        Operation._attach(self)


class _InternalCustomTuner(_InternalCustomOperation, _TunerProperty, Tuner):
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'

    def _attach(self):
        self._cpp_obj = getattr(_hoomd, self._cpp_class_name)(
            self._simulation.state._cpp_sys_def, self.trigger, self._action)
        self._action.attach(self._simulation)
        Operation._attach(self)
