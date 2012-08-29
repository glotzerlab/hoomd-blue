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

# Maintainer: jglaser / All Developers are free to add commands for new features

# \package hoomd_script.comm
# \brief Commands to support MPI communication
#
import hoomd;
from hoomd_script import init;
from hoomd_script import data;
from hoomd_script import util;
from hoomd_script import globals;

import sys;

## Setup up domain decomposition
# \brief Initialize the domain decomposition of the global simulation domain
# \internal
def init_domain_decomposition(mpi_options):
        if not init.is_initialized():
            globals.msg.error("Possible internal error! Cannot create domain decomposition before initialization.\n")
            raise RuntimeError('Error setting up domain decomposition');

        if mpi_options is not None:
            if type(mpi_options) != type(dict()):
                globals.msg.error("MPI options parameters specified incorrectly. See documentation.\n")
                raise RuntimeError('Error setting up domain decomposition');
        else:
            mpi_options = dict()

        # default values for arguents
        nx = ny = nz = 0
        linear = False

        if 'nx' in mpi_options:
            nx = mpi_options['nx']
        if 'ny' in mpi_options:
            ny = mpi_options['ny']
        if 'nz' in mpi_options:
            nz = mpi_options['nz']
        if 'linear' in mpi_options:
            linear = mpi_options['linear']

        if linear is True:
            # set up linear decomposition
            nz = globals.exec_conf.getNRanks()
  
        # exit early if we are only running on one processor
        if globals.exec_conf.getNRanks() == 1:
            return

        # take a snapshot of the global system
        pdata = globals.system_definition.getParticleData()
        nglobal = pdata.getNGlobal();
        snap = hoomd.SnapshotParticleData(nglobal)
        pdata.takeSnapshot(snap)

        # initialize domain decomposition
        cpp_decomposition = hoomd.DomainDecomposition(globals.exec_conf, pdata.getGlobalBox().getL(), nx, ny, nz);

        # create the c++ mirror Communicator
        if not globals.exec_conf.isCUDAEnabled():
            cpp_communicator = hoomd.Communicator(globals.system_definition, cpp_decomposition)
        else:
            cpp_communicator = hoomd.CommunicatorGPU(globals.system_definition, cpp_decomposition)

        # set Communicator in C++ System
        globals.system.setCommunicator(cpp_communicator)

        # set Communicator in SystemDefinition
        pdata.setDomainDecomposition(cpp_decomposition)

        # initialize domains from global snapshot
        pdata.initializeFromSnapshot(snap)
