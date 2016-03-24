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

from hoomd import _hoomd
from hoomd.md import _md
import hoomd;
from hoomd.update import _updater
import sys;

## \package hoomd.update
# \brief Commands that modify the system state in some way
#
# When an updater is specified, it acts on the particle system each time step to change
# it in some way. See the documentation of specific updaters to find out what they do.

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
    # update.rescale_temp(period=100, T=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, T, period=1, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # setup the variant inputs
        T = hoomd.variant._setup_variant_input(T);

        # create the compute thermo
        thermo = hoomd.compute._get_unique_thermo(group=hoomd.context.current.group_all);

        # create the c++ mirror class
        self.cpp_updater = _md.TempRescaleUpdater(hoomd.context.current.system_definition, thermo.cpp_compute, T.cpp_variant);
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
        hoomd.util.print_status_line();
        self.check_initialization();

        if T is not None:
            T = hoomd.variant._setup_variant_input(T);
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
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        # create the c++ mirror class
        self.cpp_updater = _md.ZeroMomentumUpdater(hoomd.context.current.system_definition);
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
# be made in the xml file read by hoomd.init.read_xml() or set dynamically via the particle data access routines. Setting
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
        hoomd.util.print_status_line();
        period = 1;

        # initialize base class
        _updater.__init__(self);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _md.Enforce2DUpdater(hoomd.context.current.system_definition);
        else:
            self.cpp_updater = _md.Enforce2DUpdaterGPU(hoomd.context.current.system_definition);
        self.setupUpdater(period);

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
    def __init__(self, group, r=None, rx=None, ry=None, rz=None, P=_hoomd.make_scalar3(0,0,0)):
        hoomd.util.print_status_line();
        period = 1;

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
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
        P = _hoomd.make_scalar3(P[0], P[1], P[2]);
        if (r is not None): rx = ry = rz = r

        # create the c++ mirror class
        if not context.exec_conf.isCUDAEnabled():
            self.cpp_updater = _hoomd.ConstraintEllipsoid(context.current.system_definition, group.cpp_group, P, rx, ry, rz);
        else:
            self.cpp_updater = _hoomd.ConstraintEllipsoidGPU(context.current.system_definition, group.cpp_group, P, rx, ry, rz);

        self.setupUpdater(period);

        # store metadata
        self.group = group
        self.P = P
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.metadata_fields = ['group','P', 'rx', 'ry', 'rz']
