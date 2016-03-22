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

import hoomd;
from hoomd_script import compute;
from hoomd_script import util;
from hoomd_script import variant;
import sys;
from hoomd_script import init;
from hoomd_script import meta;
import hoomd_script

## \package hoomd_script.update
# \brief Commands that modify the system state in some way
#
# When an updater is specified, it acts on the particle system each time step to change
# it in some way. See the documentation of specific updaters to find out what they do.

## \internal
# \brief Base class for updaters
#
# An updater in hoomd_script reflects an Updater in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ updater itself is tracked and added to the
# System 2) methods are provided for disabling the updater and changing the
# period which the system calls it
class _updater(meta._metadata):
    ## \internal
    # \brief Constructs the updater
    #
    # Initializes the cpp_updater to None.
    # Assigns a name to the updater in updater_name;
    def __init__(self):
        # check if initialization has occurred
        if not init.is_initialized():
            hoomd_script.context.msg.error("Cannot create updater before initialization\n");
            raise RuntimeError('Error creating updater');

        self.cpp_updater = None;

        # increment the id counter
        id = _updater.cur_id;
        _updater.cur_id += 1;

        self.updater_name = "updater%d" % (id);
        self.enabled = True;

        # Store a reference in global simulation variables
        hoomd_script.context.current.updaters.append(self)

        # base class constructor
        meta._metadata.__init__(self)

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
            hoomd_script.context.current.system.addUpdater(self.cpp_updater, self.updater_name, period, phase);
        elif type(period) == type(lambda n: n*2):
            hoomd_script.context.current.system.addUpdater(self.cpp_updater, self.updater_name, 1000, -1);
            hoomd_script.context.current.system.setUpdaterPeriodVariable(self.updater_name, period);
        else:
            hoomd_script.context.msg.error("I don't know what to do with a period of type " + str(type(period)) + "expecting an int or a function\n");
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
            hoomd_script.context.msg.error('Bug in hoomd_script: cpp_updater not set, please report\n');
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
        util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd_script.context.msg.warning("Ignoring command to disable an updater that is already disabled");
            return;

        self.prev_period = hoomd_script.context.current.system.getUpdaterPeriod(self.updater_name);
        hoomd_script.context.current.system.removeUpdater(self.updater_name);
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
        util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd_script.context.msg.warning("Ignoring command to enable an updater that is already enabled");
            return;

        hoomd_script.context.current.system.addUpdater(self.cpp_updater, self.updater_name, self.prev_period, self.phase);
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
        util.print_status_line();

        if type(period) == type(1.0):
            period = int(period);

        if type(period) == type(1):
            if self.enabled:
                hoomd_script.context.current.system.setUpdaterPeriod(self.updater_name, period, self.phase);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            hoomd_script.context.msg.warning("A period cannot be changed to a variable one");
        else:
            hoomd_script.context.msg.warning("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function");

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = meta._metadata.get_metadata(self)
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
# following code example disables the sorter. The init.create_random command
# is just an example; sorter can be modified after any command that initializes
# the system.
# \code
# init.create_random(N=1000, phi_p=0.2)
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
        if not hoomd_script.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = hoomd.SFCPackUpdater(hoomd_script.context.current.system_definition);
        else:
            self.cpp_updater = hoomd.SFCPackUpdaterGPU(hoomd_script.context.current.system_definition);

        default_period = 300;
        # change default period to 100 on the CPU
        if not hoomd_script.context.exec_conf.isCUDAEnabled():
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
        util.print_status_line();
        self.check_initialization();

        if grid is not None:
            self.cpp_updater.setGrid(grid);


## Rescales particle velocities
#
# Every \a period time steps, particle velocities are rescaled by equal factors
# so that they are consistent with a given temperature in the equipartition theorem
# \f$\langle 1/2 m v^2 \rangle = k_B T \f$.
#
# update.rescale_temp is best coupled with the \ref integrate.nve "NVE" integrator.
# \MPI_SUPPORTED
class rescale_temp(_updater):
    ## Initialize the rescaler
    #
    # \param T Temperature set point (in energy units)
    # \param period Velocities will be rescaled every \a period time steps
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # \b Examples:
    # \code
    # update.rescale_temp(T=1.2)
    # rescaler = update.rescale_temp(T=0.5)
    # update.rescale_temp(period=100, T=1.03)
    # update.rescale_temp(period=100, T=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, T, period=1, phase=-1):
        util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # setup the variant inputs
        T = variant._setup_variant_input(T);

        # create the compute thermo
        thermo = compute._get_unique_thermo(group=hoomd_script.context.current.group_all);

        # create the c++ mirror class
        self.cpp_updater = hoomd.TempRescaleUpdater(hoomd_script.context.current.system_definition, thermo.cpp_compute, T.cpp_variant);
        self.setupUpdater(period, phase);

        # store metadta
        self.T = T
        self.period = period
        self.metadata_fields = ['T','period']

    ## Change rescale_temp parameters
    #
    # \param T New temperature set point (in energy units)
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # rescaler = update.rescale_temp(T=0.5)
    # \endcode
    #
    # \b Examples:
    # \code
    # rescaler.set_params(T=2.0)
    # \endcode
    def set_params(self, T=None):
        util.print_status_line();
        self.check_initialization();

        if T is not None:
            T = variant._setup_variant_input(T);
            self.cpp_updater.setT(T.cpp_variant);
            self.T = T

## Zeroes system momentum
#
# Every \a period time steps, particle velocities are modified such that the total linear
# momentum of the system is set to zero.
#
# update.zero_momentum is intended to be used when the \ref integrate.nve "NVE" integrator has the
# \a limit option specified, where Newton's third law is broken and systems could gain momentum.
# Of course, it can be used in any script.
#
# \MPI_SUPPORTED
class zero_momentum(_updater):
    ## Initialize the momentum zeroer
    #
    # \param period Momentum will be zeroed every \a period time steps
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    #
    # \b Examples:
    # \code
    # update.zero_momentum()
    # zeroer= update.zero_momentum(period=10)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, period=1, phase=-1):
        util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # create the c++ mirror class
        self.cpp_updater = hoomd.ZeroMomentumUpdater(hoomd_script.context.current.system_definition);
        self.setupUpdater(period, phase);

        # store metadata
        self.period = period
        self.metadata_fields = ['period']

## Enforces 2D simulation
#
# Every time step, particle velocities and accelerations are modified so that their z components are 0: forcing
# 2D simulations when other calculations may cause particles to drift out of the plane.
#
# Using enforce2d is only allowed when the system is specified as having only 2 dimensions. This specification can
# be made in the xml file read by init.read_xml() or set dynamically via the particle data access routines. Setting
# the number of dimensions to 2 also changes the degrees of freedom calculation for temperature calculations and forces
# the neighbor list to only find 2D neighbors. Doing so requires that a small, but non-zero, value be set for the z
# dimension of the simulation box.
#
# \MPI_SUPPORTED
class enforce2d(_updater):
    ## Initialize the 2D enforcement
    #
    # \b Examples:
    # \code
    # update.enforce2d()
    # \endcode
    #
    def __init__(self):
        util.print_status_line();
        period = 1;

        # initialize base class
        _updater.__init__(self);

        # create the c++ mirror class
        if not hoomd_script.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = hoomd.Enforce2DUpdater(hoomd_script.context.current.system_definition);
        else:
            self.cpp_updater = hoomd.Enforce2DUpdaterGPU(hoomd_script.context.current.system_definition);
        self.setupUpdater(period);

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
    # update.box_resize(L = variant.linear_interp([(0, 20), (1e6, 50)]))
    # box_resize = update.box_resize(L = variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]),
    #                   Ly = variant.linear_interp([(0, 20), (1e6, 60)]),
    #                   Lz = variant.linear_interp([(0, 10), (1e6, 80)]))
    # update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), Ly = 10, Lz = 10)
    #
    # # Shear the box in the xy plane using Lees-Edwards boundary conditions
    # update.box_resize(xy = variant.linear_interp([(0,0), (1e6, 1)]))
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    #
    # If \a period is set to None, then the given box lengths are applied immediately and periodic updates
    # are not performed.
    #
    def __init__(self, Lx = None, Ly = None, Lz = None, xy = None, xz = None, yz = None, period = 1, L = None, phase=-1):
        util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        self.metadata_fields = ['period']

        if L is not None:
            Lx = L;
            Ly = L;
            Lz = L;

        if Lx is None and Ly is None and Lz is None and xy is None and xz is None and yz is None:
            hoomd_script.context.msg.warning("update.box_resize: Ignoring request to setup updater without parameters\n")
            return


        box = hoomd_script.context.current.system_definition.getParticleData().getGlobalBox();
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

        Lx = variant._setup_variant_input(Lx);
        Ly = variant._setup_variant_input(Ly);
        Lz = variant._setup_variant_input(Lz);

        xy = variant._setup_variant_input(xy);
        xz = variant._setup_variant_input(xz);
        yz = variant._setup_variant_input(yz);

        # store metadata
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.xy = xy
        self.xz = xz
        self.yz = yz
        self.metadata_fields = ['Lx','Ly','Lz','xy','xz','yz']

        # create the c++ mirror class
        self.cpp_updater = hoomd.BoxResizeUpdater(hoomd_script.context.current.system_definition, Lx.cpp_variant, Ly.cpp_variant, Lz.cpp_variant,
                                                  xy.cpp_variant, xz.cpp_variant, yz.cpp_variant);
        if period is None:
            self.cpp_updater.update(hoomd_script.context.current.system.getCurrentTimeStep());
        else:
            self.setupUpdater(period, phase);

    ## Change box_resize parameters
    #
    # \param scale_particles Set to True to scale particles with the box. Set to False
    #        to have particles remain in place when the box is scaled.
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # box_resize = update.box_resize(Lx = variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
    # \endcode
    #
    # \b Examples:
    # \code
    # box_resize.set_params(scale_particles = False)
    # box_resize.set_params(scale_particles = True)
    # \endcode
    def set_params(self, scale_particles=None):
        util.print_status_line();
        self.check_initialization();

        if scale_particles is not None:
            self.cpp_updater.setParams(scale_particles);

## Constrain particles to the surface of a ellipsoid
#
# The command update.constraint_ellipsoid specifies that all particles are constrained
# to the surface of an ellipsoid. Each time step particles are projected onto the surface of the ellipsoid.
# Method from: http://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf
# Note: For the algorithm to work, we must have \f$rx >= rz,~ry >= rz,~rz > 0\f$.
# Also note: this method does not properly conserve virial coefficients.
# Also note: random thermal forces from the integrator are applied in 3D not 2D, therefore they aren't fully accurate.
# Suggested use is therefore only for T=0.
# \MPI_NOT_SUPPORTED
class constraint_ellipsoid(_updater):
    ## Specify the %ellipsoid updater
    #
    # \param group Group for which the update will be set
    # \param P (x,y,z) tuple indicating the position of the center of the ellipsoid (in distance units).
    # \param rx radius of an ellipsoid in the X direction (in distance units).
    # \param ry radius of an ellipsoid in the Y direction (in distance units).
    # \param rz radius of an ellipsoid in the Z direction (in distance units).
    # \param r radius of a sphere (in distance units), such that r=rx=ry=rz.
    #
    # \b Examples:
    # \code
    # update.constraint_ellipsoid(P=(-1,5,0), r=9)
    # update.constraint_ellipsoid(rx=7, ry=5, rz=3)
    # \endcode
    def __init__(self, group, r=None, rx=None, ry=None, rz=None, P=hoomd.make_scalar3(0,0,0)):
        util.print_status_line();
        period = 1;

        # Error out in MPI simulations
        if (hoomd.is_MPI_available()):
            if context.current.system_definition.getParticleData().getDomainDecomposition():
                context.msg.error("constrain.ellipsoid is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error initializing updater.")

        # Error out if no radii are set
        if (r is None and rx is None and ry is None and rz is None):
            context.msg.error("no radii were defined in update.constraint_ellipsoid.\n\n")
            raise RuntimeError("Error initializing updater.")

        # initialize the base class
        _updater.__init__(self);

        # Set parameters
        P = hoomd.make_scalar3(P[0], P[1], P[2]);
        if (r is not None): rx = ry = rz = r

        # create the c++ mirror class
        if not context.exec_conf.isCUDAEnabled():
            self.cpp_updater = hoomd.ConstraintEllipsoid(context.current.system_definition, group.cpp_group, P, rx, ry, rz);
        else:
            self.cpp_updater = hoomd.ConstraintEllipsoidGPU(context.current.system_definition, group.cpp_group, P, rx, ry, rz);

        self.setupUpdater(period);

        # store metadata
        self.group = group
        self.P = P
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.metadata_fields = ['group','P', 'rx', 'ry', 'rz']

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
        util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # balancing cannot be done without mpi
        if not hoomd.is_MPI_available():
            hoomd_script.context.msg.warning("Ignoring balance command, not supported in current configuration.\n")
            return

        # create the c++ mirror class
        if not hoomd_script.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = hoomd.LoadBalancer(hoomd_script.context.current.system_definition, hoomd_script.context.current.decomposition.cpp_dd);
        else:
            self.cpp_updater = hoomd.LoadBalancerGPU(hoomd_script.context.current.system_definition, hoomd_script.context.current.decomposition.cpp_dd);

        self.setupUpdater(period,phase)

        # stash arguments to metadata
        self.metadata_fields = ['tolerance','maxiter','period','phase']
        self.period = period
        self.phase = phase

        # configure the parameters
        util.quiet_status()
        self.set_params(x,y,z,tolerance, maxiter)
        util.unquiet_status()

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
        util.print_status_line()
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
