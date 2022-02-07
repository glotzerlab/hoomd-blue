# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomWriter."""

from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Writer


class _WriterProperty:

    @property
    def writer(self):
        return self._action

    @writer.setter
    def writer(self, analyzer):
        if isinstance(analyzer, Action):
            self._action = analyzer
        else:
            raise ValueError(
                "analyzer must be an instance of hoomd.custom.Action")


class CustomWriter(CustomOperation, _WriterProperty, Writer):
    """User-defined writer.

    Args:
        action (hoomd.custom.Action): The action to call.
        trigger (hoomd.trigger.Trigger): Select the timesteps to call the
          action.

    `CustomWriter` is a `hoomd.operation.Writer` that wraps a user-defined
    `hoomd.custom.Action` object so the action can be added to a
    `hoomd.Operations` instance for use with `hoomd.Simulation` objects.

    Writers may read the system state and generate output files or print to
    output streams. Writers should not modify the system state.

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.update.CustomUpdater`

        `hoomd.tune.CustomTuner`
    """
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomWriter(_InternalCustomOperation, Writer):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
    _operation_func = "write"

    def write(self, timestep):
        return self._action.act(timestep)
