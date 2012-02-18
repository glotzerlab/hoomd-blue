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

## Boost MPI communicator
mpi_comm = None;

## Setup up domain decomposition
# \brief Initialize the domain decomposition of the global simulation domain
#
class mpi_partition:
    ## Specify the domain decomposition of the simulation domain
    def __init__(self, mpi_comm_in=None, root=0):
        util.print_status_line()
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot create MPI partition before intialization.\n"
            raise RuntimeError('Error setting up MPI partition');

        # Check if HOOMD has been compiled with MPI support
        check_boost_mpi()

        global mpi_comm
        mpi_comm = mpi_comm_in

        if mpi_comm is None:
            # Create an MPI environment
            # check if boost MPI is available
            check_boost_mpi()
            mpi_comm = mpi.world

        self.cpp_mpi_init = hoomd.MPIInitializer(globals.system_definition, mpi_comm, root);

        # Get ranks of neighboring processors
        self.neighbor_ranks = hoomd.std_vector_uint();
        self.is_at_boundary = hoomd.std_vector_bool();

        for dir in range(6):
            self.neighbor_ranks.append(self.cpp_mpi_init.getNeighborRank(dir));
            self.is_at_boundary.append(self.cpp_mpi_init.isAtBoundary(dir));

        # Get dimensions of domain decomposition
        self.dim = hoomd.make_uint3(self.cpp_mpi_init.getDimension(0), \
                                   self.cpp_mpi_init.getDimension(1), \
                                   self.cpp_mpi_init.getDimension(2));

        # create the c++ mirror Communicator
        if not globals.exec_conf.isCUDAEnabled():
            globals.communicator = hoomd.Communicator(globals.system_definition, mpi_comm, self.neighbor_ranks, \
                                                     self.is_at_boundary, self.dim, self.cpp_mpi_init.getGlobalBox());
        else:
            globals.communicator = hoomd.CommunicatorGPU(globals.system_definition, mpi_comm, self.neighbor_ranks, \
                                                     self.is_at_boundary, self.dim, self.cpp_mpi_init.getGlobalBox());

        # set Communicator in C++ System
        globals.system.setCommunicator(globals.communicator)

        # set Communicator in ParticleData
        globals.system_definition.getParticleData().setMPICommunicator(mpi_comm)

        # store this object in the global variables
        globals.mpi_partition = self

        # distribute particle data on processors
        self.scatter(root);

    ## Scatter particle data
    # \brief Distributes particle data from processor with rank root on the other processors
    #
    # \param root Rank of processor that contains the complete particle data
    def scatter(self, root=0):
        self.cpp_mpi_init.scatter(root)
