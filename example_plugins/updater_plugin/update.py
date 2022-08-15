# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Example Updater."""

# Import the C++ module.
from hoomd.updater_plugin import _updater_plugin

# Import the hoomd Python package.
import hoomd


class ExampleUpdater():
    """Example updater."""

    def __init__(self, period=1):
        # initialize base class
        # hoomd.update._updater.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_updater = _updater_plugin.ExampleUpdater(
                hoomd.context.current.system_definition)
        else:
            self.cpp_updater = _updater_plugin.ExampleUpdaterGPU(
                hoomd.context.current.system_definition)

        self.setupUpdater(period)
