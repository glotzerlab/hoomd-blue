# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement CustomTuner.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    class ExampleAction(hoomd.custom.Action):
        def act(self, timestep):
            pass

    custom_action = ExampleAction()
"""

from hoomd.custom import (CustomOperation, _InternalCustomOperation, Action)
from hoomd.operation import Tuner
import warnings


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
        trigger (hoomd.trigger.trigger_like): Select the timesteps to call the
          action.

    `CustomTuner` is a `hoomd.operation.Tuner` that wraps a user-defined
    `hoomd.custom.Action` object so the action can be added to a
    `hoomd.Operations` instance for use with `hoomd.Simulation` objects.

    Tuners modify the parameters of other operations to improve performance.
    Tuners may read the system state, but not modify it.

    .. rubric:: Example:

    .. code-block:: python

            custom_tuner = hoomd.tune.CustomTuner(
                action=custom_action,
                trigger=hoomd.trigger.Periodic(1000))
            simulation.operations.tuners.append(custom_tuner)

    See Also:
        The base class `hoomd.custom.CustomOperation`.

        `hoomd.update.CustomUpdater`

        `hoomd.write.CustomWriter`
    """
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'


class _InternalCustomTuner(_InternalCustomOperation, Tuner):
    _cpp_list_name = 'tuners'
    _cpp_class_name = 'PythonTuner'
    _operation_func = "tune"

    def tune(self, timestep):
        return self._action.act(timestep)
        """
        .. deprecated:: 4.5.0

           Use `Simulation` to call the operation.
        """
        warnings.warn(
            "`_InternalCustomTuner.tune` is deprecated,"
            "use `Simulation` to call the operation.", FutureWarning)
