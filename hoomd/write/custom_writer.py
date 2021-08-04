# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement CustomWriter."""

from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Writer


class _WriterProperty:

    @property
    def analyzer(self):
        return self._action

    @analyzer.setter
    def analyzer(self, analyzer):
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
    `hoomd.custom.Action` object  so it can be added to the `hoomd.Simulation`'s
    `hoomd.Operations` and called during the run.

    Writers may read the system state and generate output files or print to
    output streams. Writers should not modify the system state.

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.update.CustomUpdater`

        `hoomd.tune.CustomTuner`
    """
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'


class _InternalCustomWriter(_InternalCustomOperation, _WriterProperty, Writer):
    _cpp_list_name = 'analyzers'
    _cpp_class_name = 'PythonAnalyzer'
