# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

# Maintainer: joaander / All Developers are free to add commands for new features

from hoomd import _hoomd;
import hoomd;
import sys;

## \package hoomd.update
# \brief Commands that modify the system state in some way
#
# When an updater is specified, it acts on the particle system each time step to change
# it in some way. See the documentation of specific updaters to find out what they do.

## \internal
# \brief Base class for updaters
#
# An updater in hoomd.reflects an Updater in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd.# writers. 1) The instance of the c++ updater itself is tracked and added to the
# System 2) methods are provided for disabling the updater and changing the
# period which the system calls it
class _updater(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the updater
    #
    # Initializes the cpp_updater to None.
    # Assigns a name to the updater in updater_name;
    def __init__(self):
        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create updater before initialization\n");
            raise RuntimeError('Error creating updater');

        self.cpp_updater = None;

        # increment the id counter
        id = _updater.cur_id;
        _updater.cur_id += 1;

        self.updater_name = "updater%d" % (id);
        self.enabled = True;

        # Store a reference in global simulation variables
        hoomd.context.current.updaters.append(self)

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    #
    # \brief Helper function to setup updater period
    #
    # \param period An integer or callable function period
    # \param phase Phase parameter
    #
    # If an integer is specified, then that is set as the period for the analyzer.
    # If a callable is passed in as a period, then a default period of 1000 is set
    # to the integer period and the variable period is enabled
    #
    def setupUpdater(self, period, phase=-1):
        self.phase = phase;

        if type(period) == type(1.0):
            period = int(period);

        if type(period) == type(1):
            hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, period, phase);
        elif type(period) == type(lambda n: n*2):
            hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, 1000, -1);
            hoomd.context.current.system.setUpdaterPeriodVariable(self.updater_name, period);
        else:
            hoomd.context.msg.error("I don't know what to do with a period of type " + str(type(period)) + "expecting an int or a function\n");
            raise RuntimeError('Error creating updater');

    ## \var enabled
    # \internal
    # \brief True if the updater is enabled

    ## \var cpp_updater
    # \internal
    # \brief Stores the C++ side Updater managed by this class

    ## \var updater_name
    # \internal
    # \brief The Updater's name as it is assigned to the System

    ## \var prev_period
    # \internal
    # \brief Saved period retrieved when an updater is disabled: used to set the period when re-enabled

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_updater is None:
            hoomd.context.msg.error('Bug in hoomd. cpp_updater not set, please report\n');
            raise RuntimeError();

    ## Disables the updater
    #
    # \b Examples:
    # \code
    # updater.disable()
    # \endcode
    #
    # Executing the disable command will remove the updater from the system.
    # Any run() command executed after disabling an updater will not use that
    # updater during the simulation. A disabled updater can be re-enabled
    # with enable()
    #
    # To use this command, you must have saved the updater in a variable, as
    # shown in this example:
    # \code
    # updater = update.some_updater()
    # # ... later in the script
    # updater.disable()
    # \endcode
    def disable(self):
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable an updater that is already disabled");
            return;

        self.prev_period = hoomd.context.current.system.getUpdaterPeriod(self.updater_name);
        hoomd.context.current.system.removeUpdater(self.updater_name);
        self.enabled = False;

    ## Enables the updater
    #
    # \b Examples:
    # \code
    # updater.enable()
    # \endcode
    #
    # See disable() for a detailed description.
    def enable(self):
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable an updater that is already enabled");
            return;

        hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, self.prev_period, self.phase);
        self.enabled = True;

    ## Changes the period between updater executions
    #
    # \param period New period to set
    #
    # \b Examples:
    # \code
    # updater.set_period(100);
    # updater.set_period(1);
    # \endcode
    #
    # While the simulation is \ref run() "running", the action of each updater
    # is executed every \a period time steps. Changing the period does not change the phase set when the analyzer
    # was first created.
    #
    # To use this command, you must have saved the updater in a variable, as
    # shown in this example:
    # \code
    # updater = update.some_updater()
    # # ... later in the script
    # updater.set_period(10)
    # \endcode
    def set_period(self, period):
        hoomd.util.print_status_line();

        if type(period) == type(1.0):
            period = int(period);

        if type(period) == type(1):
            if self.enabled:
                hoomd.context.current.system.setUpdaterPeriod(self.updater_name, period, self.phase);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            hoomd.context.msg.warning("A period cannot be changed to a variable one");
        else:
            hoomd.context.msg.warning("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function");

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled

        return data

#
# **************************************************************************

## Sorts particles in memory to improve cache coherency
#
# Every \a period time steps, particles are reordered in memory based on
# a Hilbert curve. This operation is very efficient, and the reordered particles
# significantly improve performance of all other algorithmic steps in HOOMD.
#
# The reordering is accomplished by placing particles in spatial bins. A Hilbert curve is generated that traverses
# these bins and particles are reordered in memory in the same order in which
# they fall on the curve. The grid dimension used over the course of the simulation is held constant, and the default
# is chosen to be as fine as possible without utilizing too much memory. The dimension can be changed with set_params(),
# just be aware that the value chosen will be rounded up to the next power of 2 and that the amount of memory usage for
# 3D simulations grows very quickly:
# - \a grid=128 uses 8 MB
# - \a grid=256 uses 64 MB
# - \a grid=512 uses 512 MB
# - \a grid=1024 uses 4096 MB
#
# 2D simulations do not use any additional memory and default to \a grid=4096
#
# Because all simulations benefit from this process, a sorter is created by
# default. If you have reason to disable it or modify parameters, you
# can use the built-in variable \c sorter to do so after initialization. The
# following code example disables the sorter. The hoomd.init.create_random command
# is just an example; sorter can be modified after any command that initializes
# the system.
# \code
# hoomd.init.create_random(N=1000, phi_p=0.2)
# sorter.disable()
# \endcode
class sort(_updater):
    ## Initialize the sorter
    #
    # Users should not initialize the sorter directly. One in created for you
    # when any initialization command from init is run.
    # The created sorter can be accessed via the built-in variable \c sorter.
    #
    # By default, the sorter is created with a \a grid of 256 (4096 in 2D) and
    # an update period of 300 time steps (100 if running on the CPU).
    # The period can be changed with set_period() and the grid width can be
    # changed with set_params()
    def __init__(self):
        # initialize base class
        _updater.__init__(self);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _hoomd.SFCPackUpdater(hoomd.context.current.system_definition);
        else:
            self.cpp_updater = _hoomd.SFCPackUpdaterGPU(hoomd.context.current.system_definition);

        default_period = 300;
        # change default period to 100 on the CPU
        if not hoomd.context.exec_conf.isCUDAEnabled():
            default_period = 100;

        self.setupUpdater(default_period);

    ## Change sorter parameters
    #
    # \param grid New grid dimension (if set)
    #
    # \b Examples:
    # \code
    # sorter.set_params(grid=128)
    # \endcode
    def set_params(self, grid=None):
        hoomd.util.print_status_line();
        self.check_initialization();

        if grid is not None:
            self.cpp_updater.setGrid(grid);

## Rescales the system box size
#
# Every \a period time steps, the system box dimensions is updated to values given by
# the user (in a variant). As an option, the particles can either be left in place
# as the box is changed or their positions can be scaled with the box.
#
# \MPI_SUPPORTED
class box_resize(_updater):
    ## Initialize box size resize updater
    #
    # \param L (if set) box length in the x,y, and z directions as a function of time (in distance units)
    # \param Lx (if set) box length in the x direction as a function of time (in distance units)
    # \param Ly (if set) box length in the y direction as a function of time (in distance units)
    # \param Lz (if set) box length in the z direction as a function of time (in distance units)
    # \param xy (if set) X-Y tilt factor as a function of time (dimensionless)
    # \param xz (if set) X-Z tilt factor as a function of time (dimensionless)
    # \param yz (if set) Y-Z tilt factor as a function of time (dimensionless)
    # \param period The box size will be updated every \a period time steps
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    #
    # \a L, Lx, \a Ly, \a Lz, \a xy, \a xz, \a yz can either be set to a constant number or a variant may be provided.
    # if any of the box parameters are not specified, they are set to maintain the same value in the current box.
    #
    # Use \a L as a shorthand to specify Lx, Ly, and Lz to the same value.
    #
    # By default, particle positions are rescaled with the box. To change this behavior,
    # use set_params().
    #
    # If, under rescaling, tilt factors get too large, the simulation may slow down due to too many ghost atoms
    # being communicated. update.box.resize does NOT reset the box to orthorhombic shape if this occurs (and does not
    # move the next periodic image into the primary cell).
    #
    # \b Examples:
    # \code
    # update.box_resize(L = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]))
    # box_resize = update.box_resize(L = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # update.box_resize(Lx = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]),
    #                   Ly = hoomd.variant.linear_interp([(0, 20), (1e6, 60)]),
    #                   Lz = hoomd.variant.linear_interp([(0, 10), (1e6, 80)]))
    # update.box_resize(Lx = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]), Ly = 10, Lz = 10)
    #
    # # Shear the box in the xy plane using Lees-Edwards boundary conditions
    # update.box_resize(xy = hoomd.variant.linear_interp([(0,0), (1e6, 1)]))
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    #
    # If \a period is set to None, then the given box lengths are applied immediately and periodic updates
    # are not performed.
    #
    def __init__(self, Lx = None, Ly = None, Lz = None, xy = None, xz = None, yz = None, period = 1, L = None, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        self.metadata_fields = ['period']

        if L is not None:
            Lx = L;
            Ly = L;
            Lz = L;

        if Lx is None and Ly is None and Lz is None and xy is None and xz is None and yz is None:
            hoomd.context.msg.warning("update.box_resize: Ignoring request to setup updater without parameters\n")
            return


        box = hoomd.context.current.system_definition.getParticleData().getGlobalBox();
        # setup arguments
        if Lx is None:
            Lx = box.getL().x;
        if Ly is None:
            Ly = box.getL().y;
        if Lz is None:
            Lz = box.getL().z;

        if xy is None:
            xy = box.getTiltFactorXY();
        if xz is None:
            xz = box.getTiltFactorXZ();
        if yz is None:
            yz = box.getTiltFactorYZ();

        Lx = hoomd.variant._setup_variant_input(Lx);
        Ly = hoomd.variant._setup_variant_input(Ly);
        Lz = hoomd.variant._setup_variant_input(Lz);

        xy = hoomd.variant._setup_variant_input(xy);
        xz = hoomd.variant._setup_variant_input(xz);
        yz = hoomd.variant._setup_variant_input(yz);

        # store metadata
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.xy = xy
        self.xz = xz
        self.yz = yz
        self.metadata_fields = ['Lx','Ly','Lz','xy','xz','yz']

        # create the c++ mirror class
        self.cpp_updater = _hoomd.BoxResizeUpdater(hoomd.context.current.system_definition, Lx.cpp_variant, Ly.cpp_variant, Lz.cpp_variant,
                                                  xy.cpp_variant, xz.cpp_variant, yz.cpp_variant);
        if period is None:
            self.cpp_updater.update(hoomd.context.current.system.getCurrentTimeStep());
        else:
            self.setupUpdater(period, phase);

    ## Change box_resize parameters
    #
    # \param scale_particles Set to True to scale particles with the box. Set to False
    #        to have particles remain in place when the box is scaled.
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # box_resize = update.box_resize(Lx = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # \endcode
    #
    # \b Examples:
    # \code
    # box_resize.set_params(scale_particles = False)
    # box_resize.set_params(scale_particles = True)
    # \endcode
    def set_params(self, scale_particles=None):
        hoomd.util.print_status_line();
        self.check_initialization();

        if scale_particles is not None:
            self.cpp_updater.setParams(scale_particles);

## Adjusts the boundaries of a domain %decomposition on a regular 3D grid.
#
# Every \a period steps, the boundaries of the processor domains are adjusted to distribute the particle load close
# to evenly between them. The load imbalance is defined as the number of particles owned by a rank divided by the
# average number of particles per rank if the particles had a uniform distribution:
# \f[
# I = \frac{N(i)}{N / P}
# \f]
# where \f$ N(i) \f$ is the number of particles on processor \f$i\f$, \f$N\f$ is the total number of particles, and
# \f$P\f$ is the number of ranks.
#
# In order to adjust the load imbalance, the sizes are rescaled by the inverse of the imbalance factor. To reduce
# oscillations and communication overhead, a domain cannot move more than 5% of its current size in a single
# rebalancing step, and the edge of a domain cannot move more than half the distance to its neighbors.
#
# Simulations with interfaces (so that there is a particle density gradient) or clustering should benefit from load
# balancing. The potential speedup is roughly \f$I-1.0\f$, so that if the largest imbalance is 1.4, then the user
# can expect a roughly 40% speedup in the simulation. This is of course an estimate that assumes that all algorithms
# are roughly linear in \f$N\f$, all GPUs are fully occupied, and the simulation is limited by the speed of the slowest
# processor. It also assumes that all particles roughly equal. If you have a simulation where, for example, some particles
# have significantly more pair force neighbors than others, this estimate of the load imbalance may not produce the
# optimal results.
#
# A load balancing adjustment is only performed when the maximum load imbalance exceeds a \a tolerance. The ideal load
# balance is 1.0, so setting \a tolerance less than 1.0 will force an adjustment every \a period. The load balancer
# can attempt multiple iterations of balancing every \a period, and up to \a maxiter attempts can be made. The optimal
# values of \a period and \a maxiter will depend on your simulation.
#
# Load balancing can be performed independently and sequentially for each dimension of the simulation box. A small
# performance increase may be obtained by disabling load balancing along dimensions that are known to be homogeneous.
# For example, if there is a planar vapor-liquid interface normal to the \f$z\f$ axis, then it may be advantageous to
# disable balancing along \f$x\f$ and \f$y\f$.
#
# In systems that are well-behaved, there is minimal overhead of balancing with a small \a period. However, if the
# system is not capable of being balanced (for example, due to the density distribution or minimum domain size), having
# a small \a period and high \a maxiter may lead to a large performance loss. In such systems, it is currently best to
# either balance infrequently or to balance once in a short test run and then set the decomposition statically in a
# separate initialization.
#
# Balancing is ignored if there is no domain decomposition available (MPI is not built or is running on a single rank).
#
# \MPI_SUPPORTED
class balance(_updater):
    ## Create a load balancer
    #
    # \param x If true, balance in x dimension
    # \param y If true, balance in y dimension
    # \param z If true, balance in z dimension
    # \param tolerance Load imbalance tolerance (if <= 1.0, balance every step)
    # \param maxiter Maximum number of iterations to attempt in a single step
    # \param period Balancing will be attempted every \a period time steps
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    #
    def __init__(self, x=True, y=True, z=True, tolerance=1.02, maxiter=1, period=1000, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # balancing cannot be done without mpi
        if not _hoomd.is_MPI_available():
            hoomd.context.msg.warning("Ignoring balance command, not supported in current configuration.\n")
            return

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _hoomd.LoadBalancer(hoomd.context.current.system_definition, hoomd.context.current.decomposition.cpp_dd);
        else:
            self.cpp_updater = _hoomd.LoadBalancerGPU(hoomd.context.current.system_definition, hoomd.context.current.decomposition.cpp_dd);

        self.setupUpdater(period,phase)

        # stash arguments to metadata
        self.metadata_fields = ['tolerance','maxiter','period','phase']
        self.period = period
        self.phase = phase

        # configure the parameters
        hoomd.util.quiet_status()
        self.set_params(x,y,z,tolerance, maxiter)
        hoomd.util.unquiet_status()

    ## Change load balancing parameters
    #
    # \param x If true, balance in x dimension
    # \param y If true, balance in y dimension
    # \param z If true, balance in z dimension
    # \param tolerance Load imbalance tolerance (if <= 1.0, always rebalance)
    # \param maxiter Maximum number of iterations to attempt in a single step
    #
    # \b Examples:
    # \code
    # balance.set_params(x=True, y=False)
    # balance.set_params(tolerance=0.02, maxiter=5)
    # \endcode
    def set_params(self, x=None, y=None, z=None, tolerance=None, maxiter=None):
        hoomd.util.print_status_line()
        self.check_initialization()

        if x is not None:
            self.cpp_updater.enableDimension(0, x)
        if y is not None:
            self.cpp_updater.enableDimension(1, y)
        if z is not None:
            self.cpp_updater.enableDimension(2, z)
        if tolerance is not None:
            self.tolerance = tolerance
            self.cpp_updater.setTolerance(self.tolerance)
        if maxiter is not None:
            self.maxiter = maxiter
            self.cpp_updater.setMaxIterations(self.maxiter)

# Global current id counter to assign updaters unique names
_updater.cur_id = 0;
