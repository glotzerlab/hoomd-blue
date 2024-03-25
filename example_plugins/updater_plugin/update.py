# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Example Updater."""

# Import the C++ module.
from hoomd.updater_plugin import _updater_plugin

# Import the hoomd Python package.
import hoomd
from hoomd import operation


class ExampleUpdater(operation.Updater):
    """Example updater."""

    def __init__(self, trigger: hoomd.trigger.Trigger):
        # initialize base class
        super().__init__(trigger)

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _updater_plugin.ExampleUpdater(
                self._simulation.state._cpp_sys_def, self.trigger)
        else:
            self._cpp_obj = _updater_plugin.ExampleUpdaterGPU(
                self._simulation.state._cpp_sys_def, self.trigger)
