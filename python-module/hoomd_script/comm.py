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

## \package hoomd_script.comm
# \brief Commands to support MPI communication

import hoomd;
from hoomd_script import init;
from hoomd_script import data;
from hoomd_script import util;
from hoomd_script import globals;

import sys;

## Get the number of ranks
# \returns the number of MPI ranks in this partition
# \note Returns 1 in non-mpi builds
def get_num_ranks():
    if HOOMD.is_MPI_available():
        if init.is_initialized():
            return globals.exec_conf.getNRanks();
        else:
            if globals.options.nrank is not None:
                return globals.options.nrank;
            else:
                return hoomd.ExecutionConfiguration.getNRanksGlobal()
    else:
        return 1;

## Return the current rank
# If HOOMD is already initialized, it returns the actual MPI rank.
# If HOOMD is not yet initialized, it guesses the rank from environment
# variables.
# \note Always returns 0 in non-mpi builds
def get_rank():
    if hoomd.is_MPI_available():
        if init.is_initialized():
            return globals.exec_conf.getRank()
        else:
            if globals.options.nrank is not None:
                # recompute local rank
                return int(hoomd.EecutionConfiguration.getRankGlobal() % globals.options.nrank)
            else:
                return hoomd.ExecutionConfiguration.getRankGlobal()
    else:
        return 0;

## Return the current partition
# If HOOMD is already initialized, it returns the actual partition.
# If HOOMD is not yet initialized, it guesses the partition id from environment
# variables.
# \note Always returns 0 in non-mpi builds
def get_partition():
    if hoomd.is_MPI_available():
        if init.is_initialized():
            return globals.exec_conf.getPartition()
        else:
            if globals.options.nrank is not None:
                # re-compute partition number
                return int(hoomd.ExecutionConfiguration.getRankGlobal()/globals.options.nrank)
            else:
                return 0
    else:
        return 0;
