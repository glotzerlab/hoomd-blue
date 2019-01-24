# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# this simple python interface just activates the c++ ExampleUpdater from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (example_plugin) but with an underscore
# in front
from hoomd.example_plugin import _example_plugin

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from
# hoomd_script
import hoomd

## Zeroes all particle velocities
#
# Every \a period time steps, particle velocities are modified so that they are all zero
#
class example(hoomd.update._updater):
    ## Initialize the velocity zeroer
    #
    # \param period Velocities will be zeroed every \a period time steps
    #
    # \b Examples:
    # \code
    # example_plugin.update.example()
    # zeroer = example_plugin.update.example(period=10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, period=1):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.update._updater.__init__(self);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _example_plugin.ExampleUpdater(hoomd.context.current.system_definition);
        else:
            self.cpp_updater = _example_plugin.ExampleUpdaterGPU(hoomd.context.current.system_definition);

        self.setupUpdater(period);
