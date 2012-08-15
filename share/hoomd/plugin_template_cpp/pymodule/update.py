# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# this simple python interface just actiavates the c++ ExampleUpdater from cppmodule
# Check out any of the python code in lib/hoomd-python-module/hoomd_script for moreexamples

# First, we need to import the C++ module. It has the same name as this module (plugin_template) but with an underscore
# in front
from hoomd_plugins.plugin_template import _plugin_template

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
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_updater = _plugin_template.ExampleUpdater(globals.system_definition);
        else:
            self.cpp_updater = _plugin_template.ExampleUpdaterGPU(globals.system_definition);

        self.setupUpdater(period);

