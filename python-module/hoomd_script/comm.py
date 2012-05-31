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
    if not hoomd.is_MPI_available():
        # MPI is not available, throw an exception
        print >> sys.stderr, "\n***Error! MPI support is not available in this HOOMD version\n";
        raise RuntimeError('Error initializing MPI');

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
            print >> sys.stderr, "\n***Error! Could not load Boost.MPI python bindings.\n"
            raise RuntimeError('Error initializing MPI');

## Setup up domain decomposition
# \brief Initialize the domain decomposition of the global simulation domain
#
class mpi_partition:
    ## Specify the domain decomposition of the simulation domain
    def __init__(self, mpi_comm=None, root=0, nx=0, ny=0, nz=0, linear=False,n_replicas=1):
        util.print_status_line()
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot create MPI partition before initialization.\n"
            raise RuntimeError('Error setting up MPI partition');

        # Check if HOOMD has been compiled with MPI support
        check_mpi()

        comm_world = mpi_comm

        if comm_world is None:
            # Create an MPI environment
            # check if boost MPI is available
            check_boost_mpi()
            comm_world = mpi.world
      
        self.num_replicas = n_replicas;

        # Check if we can split the communicator into replicas
        if (comm_world.size % self.num_replicas) != 0:
            print >> sys.stderr, "\n***Error! The total number of MPI ranks (%d) has to be a multiple of " \
                                 "\n          the number of replicas (%d).\n" % (comm_world.size, self.num_replicas)
            raise RuntimeError('Error splitting MPI communicator')

        # construct intra-replica communicator
        self.replica_rank = int(comm_world.rank*self.num_replicas/comm_world.size )
        self.local_comm = comm_world.split(self.replica_rank)

        self.root = root
        # construct global communicator
        if not (self.root >= 0 and self.root < comm_world.size/self.num_replicas):
            print >> sys.stderr, "\n***Warning! The root processor rank supplied (%d) is not between 0 and the"\
                                 "\n            number %d of processors available for this replica."\
                                 "\n            Proceeding with root=0.\n" \
                                 % (self.root, comm_world.size/self.num_replicas)
            self.root = 0

        # I'm commenting out this line of code because it results in a syntax error in python 2.4
        # replica_comm seems to be unused at the moment
        # self.replica_comm = comm_world.split(0 if ((comm_world.rank - self.root) % self.num_replicas == 0) else 1)
        if (comm_world.rank - self.root) % self.num_replicas == 1:
            # Only set global communicator on the root nodes
            self.replica_comm = None;

        # Initialize this replica

        if linear is True:
            # set up linear decomposition
            nz = self.local_comm.size
   
        # take a snapshot of the global system
        pdata = globals.system_definition.getParticleData()
        nglobal = pdata.getNGlobal();
        snap = hoomd.SnapshotParticleData(nglobal)
        pdata.takeSnapshot(snap)

        # initialize domain decomposition
        self.cpp_decomposition = hoomd.DomainDecomposition(self.local_comm, pdata.getGlobalBox().getL(), self.root, nx, ny, nz);

        # create the c++ mirror Communicator
        if not globals.exec_conf.isCUDAEnabled():
            self.communicator = hoomd.Communicator(globals.system_definition, self.local_comm, self.cpp_decomposition)
        else:
            self.communicator = hoomd.CommunicatorGPU(globals.system_definition, self.local_comm, self.cpp_decomposition)

        # set Communicator in C++ System
        globals.system.setCommunicator(self.communicator)

        # set Communicator in SystemDefinition
        pdata.setDomainDecomposition(self.cpp_decomposition)

        # initialize domains from global snapshot
        pdata.initializeFromSnapshot(snap)

        # store this object in the global variables
        globals.mpi_partition = self

    ## Returns true if this is the root processor
    def isRoot(self):
        return (self.root==self.local_comm.rank)
