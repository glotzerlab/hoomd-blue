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

## Balances the domain %decomposition
#
# A single domain %decomposition is defined for the simulation.
# A standard domain %decomposition divides the simulation box into equal volumes along the Cartesian axes while minimizing
# the surface area between domains. This works well for systems where particles are uniformly distributed and
# there is equal computational load for each domain, and is the default behavior in HOOMD-blue. If no %decomposition is
# specified for an MPI run, a uniform %decomposition is automatically constructed on initialization.
#
# In simulations with density gradients, such as a vapor-liquid interface, there can be a considerable imbalance of
# particles between different ranks. The simulation time then becomes limited by the slowest processor. It may then be
# advantageous in certain systems to create domains of unequal volume, for example, by increasing the volume of less
# dense regions of the simulation box in order to balance the number of particles.
#
# The %decomposition command allows the user to control the geometry and positions of the %decomposition.
# The fractional width of the first \f$n_i - 1\f$ domains is specified along each dimension, where
# \f$n_i\f$ is the number of ranks desired along dimension \f$i\f$. If no cut planes are specified, then a uniform
# spacing is assumed. The number of domains with uniform spacing can also be specified. If the desired %decomposition
# is not commensurate with the number of ranks available (for example, a 3x3x3 decomposition when only 8 ranks are
# available), then a default uniform spacing is chosen. For the best control, the user should specify the number of
# ranks in each dimension even if uniform spacing is desired.
#
# decomposition can only be called *before* the system is initialized, at which point the particles are decomposed.
# An error is raised if the system is already initialized.
#
# The %decomposition can be adjusted dynamically if the best static decomposition is not known, or the system
# composition is changing dynamically. For this associated command, see update.balance().
#
# \warning The %decomposition command will override specified command line options.
#
class decomposition():
    ## Create a balanced domain decomposition
    # \param x First nx-1 fractional domain widths (if \a nx is None)
    # \param y First ny-1 fractional domain widths (if \a ny is None)
    # \param z First nz-1 fractional domain widths (if \a nz is None)
    # \param nx Number of processors to uniformly space in x dimension (if \a x is None)
    # \param ny Number of processors to uniformly space in y dimension (if \a y is None)
    # \param nz Number of processors to uniformly space in z dimension (if \a z is None)
    #
    # Priority is always given to specified arguments over the command line arguments. If one of these is not set but
    # a command line option is, then the command line option is used. Otherwise, a default %decomposition is chosen.
    #
    # \b Examples:
    # \code
    # comm.decomposition(x=0.4, ny=2, nz=2)
    # comm.decomposition(nx=2, y=0.8, z=[0.2,0.3])
    # \endcode
    #
    # \warning This command must be invoked *before* the system is initialized because particles are decomposed at this time.
    #
    # \note The domain size cannot be chosen arbitrarily small. There are restrictions placed on the %decomposition by the
    #       ghost layer width set by the %pair potentials. An error will be raised at run time if the ghost layer width
    #       exceeds half the shortest domain size.
    #
    # \warning Both fractional widths and the number of processors cannot be set simultaneously, and an error will be
    #          raised if both are set.
    def __init__(self, x=None, y=None, z=None, nx=None, ny=None, nz=None):
        util.print_status_line()

        # check that system is not initialized
        if globals.system is not None:
            globals.msg.error("comm.decomposition: cannot modify decomposition after system is initialized. Call before init.*\n")
            raise RuntimeError("Cannot create decomposition after system is initialized. Call before init.*")

        # check that the context has been initialized though
        if globals.exec_conf is None:
            globals.msg.error("comm.decomposition: call context.initialize() before decomposition can be set\n")
            raise RuntimeError("Cannot initialize decomposition without context.initialize() first")

        # check that there are ranks available for decomposition
        if get_num_ranks() == 1:
            globals.msg.warning("Only 1 rank in system, ignoring decomposition to use optimized code pathways.\n")
            return
        else:
            self.x = []
            self.y = []
            self.z = []
            self.nx = 0
            self.ny = 0
            self.nz = 0
            self.uniform_x = True
            self.uniform_y = True
            self.uniform_z = True

            util._disable_status_lines = True
            self.set_params(x,y,z,nx,ny,nz)
            util._disable_status_lines = False

            # do a one time update of the cuts to the global values if a global is set
            if not self.x and self.nx == 0 and globals.options.nx is not None:
                self.nx = globals.options.nx
                self.uniform_x = True
            if not self.y and self.ny == 0 and globals.options.ny is not None:
                self.ny = globals.options.ny
                self.uniform_y = True
            if not self.z and self.nz == 0:
                if globals.options.linear is True:
                    self.nz = globals.exec_conf.getNRanks()
                    self.uniform_z = True
                elif globals.options.nz is not None:
                    self.nz = globals.options.nz
                    self.uniform_z = True

            # set the global decomposition to this class
            if globals.decomposition is not None:
                globals.msg.warning("comm.decomposition: overriding currently defined domain decomposition\n")

            globals.decomposition = self

    ## Set parameters for the decomposition before initialization.
    # \param x First nx-1 fractional domain widths (if \a nx is None)
    # \param y First ny-1 fractional domain widths (if \a ny is None)
    # \param z First nz-1 fractional domain widths (if \a nz is None)
    # \param nx Number of processors to uniformly space in x dimension (if \a x is None)
    # \param ny Number of processors to uniformly space in y dimension (if \a y is None)
    # \param nz Number of processors to uniformly space in z dimension (if \a z is None)
    #
    # \b Examples:
    # \code
    # decomposition.set_params(x=[0.2])
    # decomposition.set_params(nx=1, y=[0.3,0.4], nz=2)
    # \endcode
    def set_params(self,x=None,y=None,z=None,nx=None,ny=None,nz=None):
        util.print_status_line()

        if (x is not None and nx is not None) or (y is not None and ny is not None) or (z is not None and nz is not None):
            globals.msg.error("comm.decomposition: cannot set fractions and number of processors simultaneously\n")
            raise RuntimeError("Cannot set fractions and number of processors simultaneously")

        # if x is set, use it. otherwise, if nx is set, compute x and set it
        if x is not None:
            # recast single floats as lists that can be iterated, this is the only single input we should expect
            if isinstance(x, float):
                self.x = [x]
            else:
                self.x = x
            self.uniform_x = False
        elif nx is not None:
            self.nx = nx
            self.uniform_x = True

        # do the same in y
        if y is not None:
            if isinstance(y, float):
                self.y = [y]
            else:
                self.y = y
            self.uniform_y = False
        elif ny is not None:
            self.ny = ny
            self.uniform_y = True

        # do the same in z (but also use the linear command line option if it is present, which supersedes nz)
        if z is not None:
            if isinstance(z, float):
                self.z = [z]
            else:
                self.z = z
            self.uniform_z = False
        elif nz is not None:
            self.nz = nz
            self.uniform_z = True

    ## \internal
    # \brief Delayed construction of the C++ object for this balanced decomposition
    # \param box Global simulation box for decomposition
    def _make_cpp_decomposition(self, box):
        # if the box is uniform in all directions, just use these values
        if self.uniform_x and self.uniform_y and self.uniform_z:
            self.cpp_dd = hoomd.DomainDecomposition(globals.exec_conf, box.getL(), self.nx, self.ny, self.nz, not globals.options.onelevel)
            return self.cpp_dd

        # otherwise, make the fractional decomposition
        try:
            fxs = hoomd.std_vector_scalar()
            fys = hoomd.std_vector_scalar()
            fzs = hoomd.std_vector_scalar()

            # if uniform, correct the fractions to be uniform as well
            if self.uniform_x and self.nx > 0:
                self.x = [1.0/self.nx]*(self.nx-1)
            if self.uniform_y and self.ny > 0:
                self.y = [1.0/self.ny]*(self.ny-1)
            if self.uniform_z and self.nz > 0:
                self.z = [1.0/self.nz]*(self.nz-1)

            sum_x = sum_y = sum_z = 0.0
            tol = 1.0e-5
            for i in self.x:
                if i <= -tol or i >= 1.0 - tol:
                    globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fxs.append(i)
                sum_x += i
            if sum_x >= 1.0 - tol or sum_x <= -tol:
                globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in x must lie between 0.0 and 1.0")

            for i in self.y:
                if i <= -tol or i >= 1.0 - tol:
                    globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fys.append(i)
                sum_y += i
            if sum_y >= 1.0 - tol or sum_y <= -tol:
                globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in y must lie between 0.0 and 1.0")

            for i in self.z:
                if i <= -tol or i >= 1.0 - tol:
                    globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                    raise RuntimeError("Fractional decomposition must be between 0.0 and 1.0")
                fzs.append(i)
                sum_z += i
            if sum_z >= 1.0 - tol or sum_z <= -tol:
                globals.msg.error("comm.decomposition: fraction must be between 0.0 and 1.0\n")
                raise RuntimeError("Sum of decomposition in z must lie between 0.0 and 1.0")

            self.cpp_dd = hoomd.DomainDecomposition(globals.exec_conf, box.getL(), fxs, fys, fzs)
            return self.cpp_dd

        except TypeError as te:
            globals.msg.error("Fractional cuts must be iterable (list, tuple, etc.)\n")
            raise te
