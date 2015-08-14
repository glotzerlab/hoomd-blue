# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
# the University of Michigan All rights reserved.

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
from hoomd_script import util;
from hoomd_script import globals;
import hoomd_script;

import sys;

## Get the number of ranks
# \returns the number of MPI ranks in this partition
# context.initialize() must be called before get_num_ranks()
# \note Returns 1 in non-mpi builds
def get_num_ranks():
    hoomd_script.context._verify_init();
    if hoomd.is_MPI_available():
        return globals.exec_conf.getNRanks();
    else:
        return 1;

## Return the current rank
# context.initialize() must be called before get_rank()
# \note Always returns 0 in non-mpi builds
def get_rank():
    hoomd_script.context._verify_init();

    if hoomd.is_MPI_available():
        return globals.exec_conf.getRank()
    else:
        return 0;

## Return the current partition
# context.initialize() must be called before get_partition()
# \note Always returns 0 in non-mpi builds
def get_partition():
    hoomd_script.context._verify_init();

    if hoomd.is_MPI_available():
        return globals.exec_conf.getPartition()
    else:
        return 0;

## Perform a MPI barrier synchronization inside a partition
# \note does nothing in in non-MPI builds
def barrier_all():
    if hoomd.is_MPI_available():
        hoomd.mpi_barrier_world();

## Perform a MPI barrier synchronization inside a partition
# context.initialize() must be called before barrier()
# \note does nothing in in non-MPI builds
def barrier():
    hoomd_script.context._verify_init();

    if hoomd.is_MPI_available():
        globals.exec_conf.barrier()

## Balances the domain decomposition
# \param fx Fractional widths of the nx-1 cuts in x
# \param fy Fractional widths of the ny-1 cuts in y
# \param fz Fractional widths of the nz-1 cuts in z
#
# A standard domain decomposition divides the simulation box into equal volumes along the Cartesian axes while minimizing
# the surface area between domains. Although this works well for systems where particles are uniformly distributed and
# there is equal computational load for each domain, it can lead to significant load imbalance between processors
# in simulations with density gradients, such as a vapor-liquid interface, or with significant particle clustering. The
# simulation time then becomes limited by the slowest processor. It may then be advantageous in certain systems to
# create domains of unequal volume, for example, by increasing the volume of less dense regions of the simulation box
# in order to balance the number of particles.
#
# The %balance command allows the user to shift the Cartesian cut planes of the decomposition to adjust the volume
# of each domain. The fractional width of the first \f$n_i - 1\f$ domains is specified along each dimension, where
# \f$n_i\f$ is the number of ranks desired along dimension \f$i\f$. If no cut planes are specified, then a uniform
# spacing is assumed. The number of domains in the uniform dimensions is either optimized by the regular
# decomposition spacing, or by the command line options. If the desired decomposition is not commensurate with the
# number of ranks available (for example, a 3x3x3 decomposition when only 8 ranks are available), then a default uniform
# spacing is chosen. For the best control, the user should specify all cut planes.
#
# \b Examples:
# \code
# comm.balance(x=0.4)
# comm.balance(y=0.8, z=[0.2,0.3])
# \endcode
#
# \note This command must be invoked *before* the system is initialized because particles are decomposed at this time.
# \note The domain size cannot be chosen arbitrarily small. There are restrictions placed on the decomposition by the
#       ghost layer width set by the %pair potentials. An error will be raised if the ghost layer width exceeds half
#       the shortest domain size.
class balance():
    def __init__(self,x=[], y=[], z=[]):
        util.print_status_line()

        # check that system is not initialized
        if init.is_initialized():
            raise RuntimeError("Cannot balance decomposition after system is initialized. Call before init.*")
    
        # check that there are ranks available for decomposition
        if globals.exec_conf.getNRanks() == 1:
            globals.msg.warning("Only 1 rank in system, ignoring decomposition to use optimized code pathways.")
            globals.decomposition = None
        elif x == [] and y == [] and z == []:
            globals.msg.warning("Uniform decomposition requested in all dimensions, ignoring balance command")
            globals.decomposition = None
        else:
            # recast single floats as lists that can be iterated, this is the only single input we should expect
            if isinstance(x, float):
                self.x = [x]
            else:
                self.x = x

            if isinstance(y, float):
                self.y = [y]
            else:
                self.y = y

            if isinstance(z, float):
                self.z = [z]
            else:
                self.z = z

            # if a rank dimension is specified and no fractions are set, then fix this uniform spacing
            if globals.options.nx is not None and len(self.x) == 0:
                nx = globals.options.nx
                self.x = [1.0/nx]*(nx-1)
            if globals.options.ny is not None and len(self.y) == 0:
                ny = globals.options.ny
                self.y = [1.0/ny]*(ny-1)
            if globals.options.nz is not None and len(self.z) == 0:
                nz = globals.options.nz
                self.z = [1.0/nz]*(nz-1)

            # set the global decomposition to this class
            globals.decomposition = self

    ## \internal
    # \brief Delayed construction of the C++ object for this balanced decomposition
    # \param box Global simulation box for decomposition
    def _make_cpp_decomposition(self, box):
        try:
            fxs = hoomd.std_vector_scalar()
            fys = hoomd.std_vector_scalar()
            fzs = hoomd.std_vector_scalar()

            sum_x = sum_y = sum_z = 0.0
            tol = 1.0e-5
            for i in self.x:
                if i <= -tol or i >= 1.0 - tol:
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fxs.append(i)
                sum_x += i
            if sum_x >= 1.0 - tol or sum_x <= -tol:
                raise RuntimeError("Sum of decomposition in x must lie between 0.0 and 1.0")
            elif len(self.x) > 0:
                fxs.append(1.0 - sum_x) # balance out the rest of the decomposition

            for i in self.y:
                if i <= -tol or i >= 1.0 - tol:
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fys.append(i)
                sum_y += i
            if sum_y >= 1.0 - tol or sum_y <= -tol:
                raise RuntimeError("Sum of decomposition in y must lie between 0.0 and 1.0")
            elif len(self.y) > 0:
                fys.append(1.0 - sum_y) # balance out the rest of the decomposition

            for i in self.z:
                if i <= -tol or i >= 1.0 - tol:
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fzs.append(i)
                sum_z += i
            if sum_z >= 1.0 - tol or sum_z <= -tol:
                raise RuntimeError("Sum of decomposition in z must lie between 0.0 and 1.0")
            elif len(self.z) > 0:
                fzs.append(1.0 - sum_z) # balance out the rest of the decomposition

            return hoomd.BalancedDomainDecomposition(globals.exec_conf, box.getL(), fxs, fys, fzs)

        except TypeError,te:
            globals.msg.error("Fractional cuts must be iterable (list, tuple, etc.)")
            raise te
