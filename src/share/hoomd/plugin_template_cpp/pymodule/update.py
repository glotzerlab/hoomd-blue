# -*- coding: iso-8859-1 -*-

# this simple python interface just actiavates the c++ ExampleUpdater from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
import _plugin_template

# Next, since we are extending an updater, we need to bring in the base class updater and some other parts from 
# hoomd_script
from hoomd_script.update import _updater
from hoomd_script import util
from hoomd_script import globals
import hoomd

## Zeroes all particle velocities
#
# Every \a period time steps, particle velocities are modified so that they are all zero
#
class example(_updater):
    ## Initialize the velocity zeroer
    #
    # \param period Velocities will be zeroed every \a period time steps
    # 
    # \b Examples:
    # \code
    # plugin_template.update.example()
    # zeroer = plugin_template.update.example(period=10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, period=1):
        util.print_status_line();
    
        # initialize base class
        _updater.__init__(self);
        
        # initialize the reflected c++ class
        if globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
            self.cpp_integrator = _plugin_template.ExampleUpdater(globals.system_definition);
        elif globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
            self.cpp_integrator = _plugin_template.ExampleUpdaterGPU(globals.system_definition);
        else:
            print >> sys.stderr, "\n***Error! Invalid execution mode\n";
            raise RuntimeError("Error creating example updater");
        self.setupUpdater(period);
