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
import init;
import data;
import util;
import globals;

import sys;

## \internal
# Check if MPI is available inside HOOMD
#
# \returns true if the HOOMD library has been compiled with MPI support
#
def check_mpi():
    return hoomd.is_MPI_available()

## \internal
# \brief Check if the python bindings for Boost.MPI is available, if so, import them
#
# Importing the Boost.MPI python bindings initializes the MPI environment.
# The module is imported upon calling this function, allowing the user to initialize MPI
# at a late instant,  e.g. after GPU initialization. This initialization order is
# required for some CUDA-MPI implementations to work correctly.
#
def check_boost_mpi():
    global mpi
    try:
        # Try standard location
        import mpi
    except ImportError:
        try:
            # According to the Boost.MPI documentation, this should be used
            import boost.mpi as mpi
        except ImportError:
            # Boost.MPI is not available, throw an exception
            globals.msg.warning("Could not load Boost.MPI python bindings. Disabling MPI support.")
            return False

    return True

## Setup up domain decomposition
# \brief Initialize the domain decomposition of the global simulation domain
# \internal
def init_domain_decomposition(mpi_arguments):
        if not init.is_initialized():
            globals.msg.error("Possible internal error! Cannot create MPI partition before initialization.")
            raise RuntimeError('Error setting up MPI partition');

        if type(mpi_arguments) != type(dict()):
            globals.msg.error("MPI partition parameters specified incorrectly.")
            raise RuntimeError('Error setting up MPI partition');
    
        # default values for arguents
        root = 0
        nx = ny = nz = 0
        linear = False

        if 'root' in mpi_arguments:
            root = mpi_arguments['root']
        if 'nx' in mpi_arguments:
            nx = mpi_arguments['nx']
        if 'ny' in mpi_arguments:
            ny = mpi_arguments['ny']
        if 'nz' in mpi_arguments:
            nz = mpi_arguments['nz']
        if 'linear' in mpi_arguments:
            linear = mpi_arguments['linear']

        if not (root >= 0 and root < mpi.world.size):
            globals.msg.warning("Invalid root processor rank (%d). Proceeding with rank 0 as root." % (root))
            root = 0

        if linear is True:
            # set up linear decomposition
            nz = mpi.world.size
   
        # take a snapshot of the global system
        pdata = globals.system_definition.getParticleData()
        nglobal = pdata.getNGlobal();
        snap = hoomd.SnapshotParticleData(nglobal)
        pdata.takeSnapshot(snap)

        # initialize domain decomposition
        cpp_decomposition = hoomd.DomainDecomposition(globals.exec_conf, mpi.world, pdata.getGlobalBox().getL(), root, nx, ny, nz);

        # create the c++ mirror Communicator
        if not globals.exec_conf.isCUDAEnabled():
            cpp_communicator = hoomd.Communicator(globals.system_definition, mpi.world, cpp_decomposition)
        else:
            cpp_communicator = hoomd.CommunicatorGPU(globals.system_definition, mpi.world, cpp_decomposition)

        # set Communicator in C++ System
        globals.system.setCommunicator(cpp_communicator)

        # set Communicator in SystemDefinition
        pdata.setDomainDecomposition(cpp_decomposition)

        # initialize domains from global snapshot
        pdata.initializeFromSnapshot(snap)
