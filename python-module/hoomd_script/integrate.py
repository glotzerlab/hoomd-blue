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

# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.integrate
# \brief Commands that integrate the equations of motion
#
# To integrate the system forward in time, an integration mode must be set. Only one integration mode can be active at
# a time, and the last \c integrate.mode_* command before the run() command is the one that will take effect. It is 
# possible to set one mode, run() for a certain number of steps and then switch to another mode before the next run()
# command.
#
# The most commonly used mode is integrate.mode_standard . It specifies a standard mode where, at each time
# step, all of the specified forces are evaluated and used in moving the system forward to the next step.
# integrate.mode_standard doesn't integrate any particles by itself, one or more compatible integration methods must
# be specified before the run() command. Like commands that specify forces, integration methods are \b persistent and
# remain set until they are disabled (this differs greatly from HOOMD-blue behavior in all versions prior to 0.9.0).
# The benefit and reason for this change is that now multiple integration methods can be specified on different particle
# groups, allowing portions of the system to be fixed, integrated at a different temperature, etc...
#
# To clarify, the following series of commands will run for 1000 time steps in the NVT ensemble and then switch to
# NVE for another 1000 steps.
#
# \code
# all = group.all()
# integrate.mode_standard(dt=0.005)
# nvt = integrate.nvt(group=all, T=1.2, tau=0.5)
# run(1000)
# nvt.disable()
# integrate.nve(group=all)
# run(1000)
# \endcode
#
# For more detailed information on the interaction of integration methods and integration modes, see
# integrate.mode_standard.
#
# Some integrators provide parameters that can be changed between runs.
# In order to access the integrator to change it, it needs to be saved
# in a variable. For example:
# \code
# integrator = integrate.nvt(group=all, T=1.2, tau=0.5)
# run(100)
# integrator.set_params(T=1.0)
# run(100)
# \endcode
# This code snippet runs the first 100 time steps with T=1.2 and the next 100 with T=1.0

import hoomd;
from hoomd_script import globals;
from hoomd_script import compute;
import sys;
from hoomd_script import util;
from hoomd_script import variant;
from hoomd_script import init;
from hoomd_script import comm;

## \internal
# \brief Base class for integrators
#
# An integrator in hoomd_script reflects an Integrator in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ integrator itself is tracked 2) All
# forces created so far in the simulation are updated in the cpp_integrator
# whenever run() is called.
class _integrator:
    ## \internal
    # \brief Constructs the integrator
    #
    # This doesn't really do much bet set some member variables to None
    def __init__(self):
        # check if initialization has occured
        if not init.is_initialized():
            globals.msg.error("Cannot create integrator before initialization\n");
            raise RuntimeError('Error creating integrator');
        
        # by default, integrators do not support methods
        self.cpp_integrator = None;
        self.supports_methods = False;
        
        # save ourselves in the global variable
        globals.integrator = self;
        
    ## \var cpp_integrator
    # \internal 
    # \brief Stores the C++ side Integrator managed by this class
    
    ## \var supports_methods
    # \internal
    # \brief True if this integrator supports integration methods
    # \note If hoomd ever needs to support multiple TYPES of methods, we could just change this to a string naming the
    # type that is supported and add a type string to each of the integration_methods.

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_integrator is None:
            globals.msg.error('Bug in hoomd_script: cpp_integrator not set, please report\n');
            raise RuntimeError();

    ## \internal
    # \brief Updates the integrators in the reflected c++ class
    def update_forces(self):
        self.check_initialization();
        
        # set the forces
        self.cpp_integrator.removeForceComputes();
        for f in globals.forces:
            if f.cpp_force is None:
                globals.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');
            
            if f.log or f.enabled:    
                f.update_coeffs();
                
            if f.enabled:
                self.cpp_integrator.addForceCompute(f.cpp_force);
        
        # set the constraint forces
        for f in globals.constraint_forces:
            if f.cpp_force is None:
                globals.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');

            if f.enabled:
                self.cpp_integrator.addForceConstraint(f.cpp_force);
            
    
    ## \internal
    # \brief Updates the integration methods in the reflected c++ class
    def update_methods(self):
        self.check_initialization();
            
        # if we support methods, add them all to the list
        if self.supports_methods:
            self.cpp_integrator.removeAllIntegrationMethods();
            
            if len(globals.integration_methods) == 0:
                globals.msg.error('This integrator requires that one or more integration methods be specified.\n');
                raise RuntimeError('Error initializing integrator methods');
            
            for m in globals.integration_methods:
                self.cpp_integrator.addIntegrationMethod(m.cpp_method);

        else:
            if len(globals.integration_methods) > 0:
                globals.msg.error("This integrator does not support the use of integration methods,\n");
                globals.msg.error("but some have been specified in the script. Remove them or use\n");
                globals.msg.error("a different integrator.\n");
                raise RuntimeError('Error initializing integrator methods');

    ## \internal
    # \brief Counts the number of degrees of freedom and updates each compute.thermo specified
    def update_thermos(self):
        self.check_initialization();
        
        for t in globals.thermos:
            ndof = self.cpp_integrator.getNDOF(t.group.cpp_group);
            t.cpp_compute.setNDOF(ndof);

## \internal
# \brief Base class for integration methods
#
# An integration_method in hoomd_script reflects an IntegrationMethod in c++. It is responsible for all high-level
# management that happens behind the scenes for hoomd_script writers. 1) The instance of the c++ integration method
# itself is tracked and added to the integrator and 2) methods are provided for disabling the integration method from
# being active for the next run()
#
# The design of integration_method exactly mirrors that of _force for consistency
class _integration_method:
    ## \internal
    # \brief Constructs the integration_method
    #
    # Initializes the cpp_method to None.
    def __init__(self):
        # check if initialization has occured
        if not init.is_initialized():
            globals.msg.error("Cannot create an integration method before initialization\n");
            raise RuntimeError('Error creating integration method');
        
        self.cpp_method = None;
        
        self.enabled = True;
        globals.integration_methods.append(self);

    ## \var enabled
    # \internal
    # \brief True if the integration method is enabled

    ## \var cpp_method
    # \internal
    # \brief Stores the C++ side IntegrationMethod managed by this class

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_method is None:
            globals.msg.error('Bug in hoomd_script: cpp_method not set, please report\n');
            raise RuntimeError();

    ## Disables the integration method
    #
    # \b Examples:
    # \code
    # method.disable()
    # \endcode
    #
    # Executing the disable command will remove the integration method from the simulation.Any run() command executed
    # after disabling an integration method will not apply the integration method to the particles during the
    # simulation. A disabled integration method can be re-enabled with enable()
    #
    # To use this command, you must have saved the force in a variable, as 
    # shown in this example:
    # \code
    # method = integrate.some_method()
    # # ... later in the script
    # method.disable()
    # \endcode
    def disable(self):
        util.print_status_line();
        self.check_initialization()
        
        # check if we are already disabled
        if not self.enabled:
            globals.msg.warning("Ignoring command to disable an integration method that is already disabled");
            return;
        
        self.enabled = False;
        globals.integration_methods.remove(self);

    ## Enables the integration method
    #
    # \b Examples:
    # \code
    # method.enable()
    # \endcode
    #
    # See disable() for a detailed description.
    def enable(self):
        util.print_status_line();
        self.check_initialization();
        
        # check if we are already disabled
        if self.enabled:
            globals.msg.warning("Ignoring command to enable an integration method that is already enabled");
            return;
        
        self.enabled = True;
        globals.integration_methods.append(self);

## Enables a variety of standard integration methods
#
# integrate.mode_standard performs a standard time step integration technique to move the system forward. At each time
# step, all of the specified forces are evaluated and used in moving the system forward to the next step.
#
# By itself, integrate.mode_standard does nothing. You must specify one or more integration methods to apply to the
# system. Each integration method can be applied to only a specific group of particles enabling advanced simulation
# techniques.
#
# The following commands can be used to specify the integration methods used by integrate.mode_standard.
# - integrate.bdnvt
# - integrate.bdnvt_rigid
# - integrate.nve
# - integrate.nve_rigid
# - integrate.nvt
# - integrate.nvt_rigid
# - integrate.npt
# - integrate.nph
#
# There can only be one integration mode active at a time. If there are more than one integrate.mode_* commands in
# a hoomd script, only the most recent before a given run() will take effect.
class mode_standard(_integrator):
    ## Specifies the standard integration mode
    # \param dt Each time step of the simulation run() will advance the real time of the system forward by \a dt (in time units)
    #
    # \b Examples:
    # \code
    # integrate.mode_standard(dt=0.005)
    # integrator_mode = integrate.mode_standard(dt=0.001)
    # \endcode
    def __init__(self, dt):
        util.print_status_line();
        
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        self.cpp_integrator = hoomd.IntegratorTwoStep(globals.system_definition, dt);
        self.supports_methods = True;
       
        globals.system.setIntegrator(self.cpp_integrator);
    
    ## Changes parameters of an existing integration mode
    # \param dt New time step delta (if set) (in time units)
    #
    # To change the parameters of an existing integration mode, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator_mode = integrate.mode_standard(dt=5e-3)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator_mode.set_params(dt=0.007)
    # \endcode
    def set_params(self, dt=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if dt is not None:
            self.cpp_integrator.setDeltaT(dt);

## NVT Integration via the Nos&eacute;-Hoover thermostat
#
# integrate.nvt performs constant volume, constant temperature simulations using the Nos&eacute;-Hoover thermostat.
# Equation 13 in ref. \cite Bond1999 is used to integrate the equations of motion.
#
# integrate.nvt is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
#
# integrate.nvt uses the proper number of degrees of freedom to compute the temperature of the system in both
# 2 and 3 dimensional systems, as long as the number of dimensions is set before the integrate.nvt command
# is specified.
#
class nvt(_integration_method):
    ## Specifies the NVT integration method
    # \param group Group of particles on which to apply this method.
    # \param T Temperature set point for the Nos&eacute;-Hoover thermostat. (in energy units)
    # \param tau Coupling constant for the Nos&eacute;-Hoover thermostat. (in time units)
    #
    # \f$ \tau \f$ is related to the Nos&eacute; mass \f$ Q \f$ by 
    # \f[ \tau = \sqrt{\frac{Q}{g k_B T_0}} \f] where \f$ g \f$ is the number of degrees of freedom,
    # and \f$ k_B T_0 \f$ is the set point (\a T above).
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    #
    # \b Examples:
    # \code
    # all = group.all()
    # integrate.nvt(group=all, T=1.0, tau=0.5)
    # integrator = integrate.nvt(group=all, tau=1.0, T=0.65)
    # typeA = group.type('A')
    # integrator = integrate.nvt(group=typeA, tau=1.0, T=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    def __init__(self, group, T, tau):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        
        # create the compute thermo
        thermo = compute._get_unique_thermo(group=group);

        # setup suffix
        suffix = '_' + group.name;
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNVT(globals.system_definition, group.cpp_group, thermo.cpp_compute, tau, T.cpp_variant, suffix);
        else:
            self.cpp_method = hoomd.TwoStepNVTGPU(globals.system_definition, group.cpp_group, thermo.cpp_compute, tau, T.cpp_variant, suffix);
        
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    # \param tau New coupling constant (if set) (in time units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nvt(group=all, tau=1.0, T=0.65)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(tau=0.6)
    # integrator.set_params(tau=0.7, T=2.0)
    # \endcode
    def set_params(self, T=None, tau=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        if tau is not None:
            self.cpp_method.setTau(tau);

## NPT Integration
#
# integrate.npt performs constant pressure, constant temperature simulations, allowing for a fully deformable
# simulation box
#
# The integration method is based on the rigorous Martyna-Tobias-Klein equations of motion for NPT.
# For optimal stability, the update equations leave the phase-space meeasure invariant and are manifestly
# time-reversible.
#
# By default, integrate.npt performs integration in a cubic box under hydrostatic pressure by simultaneously
# rescaling the lengths \a Lx, \a Ly and \a Lz of the simulation box.
#
# integrate.npt can also perform more advanced integration modes. The integration mode
# is specified by a set of \a couplings and by specifying the box degrees of freedom that are put under
# barostat control.
#
# \a Couplings define which diagonal elements of the pressure tensor \f$ P_{\alpha,\beta} \f$
# should be averaged over, so that the corresponding box lengths are rescaled by the same amount.
#
# Valid \a couplings are:<br>
# - \b none (all box lengths are updated independently)
# - \b xy (\a Lx and \a Ly are coupled)
# - \b xz (\a Lx and \a Lz are coupled)
# - \b yz (\a Ly and \a Lz are coupled)
# - \b xyz (\a Lx and \a Ly and \a Lz are coupled)
#
# The default coupling is \b xyz, i.e. the ratios between all box lengths stay constant.
#
# <em>Degrees of freedom</em> of the box specify which lengths and tilt factors of the box should be updated,
# and how particle coordinates and velocities should be rescaled.
# 
# Valid keywords for degrees of freedom are:
# - \b x (the box length Lx is updated)
# - \b y (the box length Ly is updated)
# - \b z (the box length Lz is updated)
# - \b xy (the tilt factor xy is updated)
# - \b xz (the tilt factor xz is updated)
# - \b yz (the tilt factor yz is updated)
# - \b all (all elements are updated, equivalent to \b x, \b y, \b z, \b xy, \b xz, and \b yz together)
#
# Any of the six keywords can be combined together. By default, the \b x, \b y, and \b z degrees of freedom
# are updated.
#
# For example:
# - Specifying \b xyz copulings and \b x, \b y, and \b z degrees of freedom amounts to \a cubic symmetry (default)
# - Specifying \b xy couplings and \b x, \b y, and \b z degrees of freedom amounts to \a tetragonal symmetry.
# - Specifing no couplings and \b all degrees of freedom amounts to a fully deformable \a triclinic unit cell
#
# integrate.npt is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
#
# integrate.npt uses the proper number of degrees of freedom to compute the temperature and pressure of the system in
# both 2 and 3 dimensional systems, as long as the number of dimensions is set before the integrate.npt command
# is specified.
#
# For the MTK equations of motion, see:
# \cite Martyna1994
# \cite Tuckerman2006
# \cite Yu2010
# Glaser et. al (2013), to be published
class npt(_integration_method):
    ## Specifies the NPT integrator
    # \param group Group of particles on which to apply this method.
    # \param T Temperature set point for the thermostat, not needed if \b nph=True (in energy units)
    # \param P Pressure set point for the barostat (in pressure units)
    # \param tau Coupling constant for the thermostat, not needed if \nph=True (in time units)
    # \param tauP Coupling constant for the barostat (in time units)
    # \param couple Couplings of diagonal elements of the stress tensor, can be \b "none", \b "xy", \b "xz",\b "yz", or \b "xyz" (default)
    # \param x if \b True, rescale \a Lx and x component of particle coordinates and velocities 
    # \param y if \b True, rescale \a Ly and y component of particle coordinates and velocities
    # \param z if \b True, rescale \a Lz and z component of particle coordinates and velocities
    # \param xy if \b True, rescale \a xy tilt factor and x and y components of particle coordinates and velocities
    # \param xz if \b True, rescale \a xz tilt factor and x and z components of particle coordinates and velocities
    # \param yz if \b True, rescale \a yz tilt factor and y and z components of particle coordinates and velocities
    # \param all if \b True, rescale all lengths and tilt factors and components of particle coordinates and velocities
    # \param nph if \b True, integrate without a thermostat, i.e. in the NPH ensemble
    #
    # Both \a T and \a P can be variant types, allowing for temperature/pressure ramps in simulation runs.
    #
    # \f$ \tau \f$ is related to the Nos&eacute; mass \f$ Q \f$ by 
    # \f[ \tau = \sqrt{\frac{Q}{g k_B T_0}} \f] where \f$ g \f$ is the number of degrees of freedom,
    # and \f$ k_B T_0 \f$ is the set point (\a T above).
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    #
    # \b Examples:
    # \code
    # integrate.npt(group=all, T=1.0, tau=0.5, tauP=1.0, P=2.0)
    # integrator = integrate.npt(tau=1.0, dt=5e-3, T=0.65, tauP = 1.2, P=2.0)
    # # orthorhombic symmetry
    # integrator = integrate.npt(tau=1.0, dt=5e-3, T=0.65, tauP = 1.2, P=2.0, couple="none")
    # # tetragonal symmetry
    # integrator = integrate.npt(tau=1.0, dt=5e-3, T=0.65, tauP = 1.2, P=2.0, couple="xy")
    # # triclinic symmetry
    # integrator = integrate.npt(tau=1.0, dt=5e-3, T=0.65, tauP = 1.2, P=2.0, couple="none", all=True)
    # \endcode
    def __init__(self, group, P, tauP, couple="xyz", x=True, y=True, z=True, xy=False, xz=False, yz=False, all=False, nph=False, T=None, tau=None):
        util.print_status_line();
       
        # check the input
        if (T is None or tau is None):
            if nph is False:
                globals.msg.error("integrate.npt: Need temperature T and thermostat time scale tau.\n");
                raise RuntimeError("Error setting up NPT integration.");
            else:
                # use dummy values
                T=1.0
                tau=1.0

        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        P = variant._setup_variant_input(P);
        
        # create the compute thermo
        thermo_group = compute._get_unique_thermo(group=group);
        thermo_all = compute._get_unique_thermo(group=globals.group_all);

        # need to know if we are running 2D simulations
        twod = (globals.system_definition.getNDimensions() == 2);
        if twod:
            globals.msg.notice(2, "When running in 2D, z couplings and degrees of freedom are silently ignored.\n");

        # initialize the reflected c++ class
        if twod:
            # silently ignore any couplings that involve z
            if couple == "none":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xy":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_xy
            elif couple == "xz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "yz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xyz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_xy
            else:
                globals.msg.error("Invalid coupling mode\n");
                raise RuntimeError("Error setting up NPT integration.");
        else:
            if couple == "none":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xy":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_xy
            elif couple == "xz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_xz
            elif couple == "yz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_yz
            elif couple == "xyz":
                cpp_couple = hoomd.TwoStepNPTMTK.couplingMode.couple_xyz
            else:
                globals.msg.error("Invalid coupling mode\n");
                raise RuntimeError("Error setting up NPT integration.");

        # set degrees of freedom flags
        # silently ignore z related degrees of freedom when running in 2d
        flags = 0;
        if x or all:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_x
        if y or all:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_y
        if (z or all) and not twod:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_z
        if xy or all:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_xy
        if (xz or all) and not twod:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_xz
        if (yz or all) and not twod:
            flags |= hoomd.TwoStepNPTMTK.baroFlags.baro_yz

        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNPTMTK(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, tau, tauP, T.cpp_variant, P.cpp_variant, cpp_couple, flags, nph);
        else:
            self.cpp_method = hoomd.TwoStepNPTMTKGPU(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, tau, tauP, T.cpp_variant, P.cpp_variant, cpp_couple, flags, nph);

        self.cpp_method.validateGroup()

    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    # \param tau New coupling constant (if set) (in time units)
    # \param P New pressure (if set) (in pressure units)
    # \param tauP New barostat coupling constant (if set) (in time units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.npt(tau=1.0, T=0.65)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(tau=0.6)
    # integrator.set_params(dt=3e-3, T=2.0, P=1.0)
    # \endcode
    def set_params(self, T=None, tau=None, P=None, tauP=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        if tau is not None:
            self.cpp_method.setTau(tau);
        if P is not None:
            # setup the variant inputs
            P = variant._setup_variant_input(P);
            self.cpp_method.setP(P.cpp_variant);
        if tauP is not None:
            self.cpp_method.setTauP(tauP);

## NPH Integration via Andersen barostat or Parrinello-Rahman barostat restricted to anisotropic length fluctuations
#
# integrate.nph performs constant pressure (NPH) simulations using a Martyna-Tobias-Klein barostat, an
# explicitly reversible and measure-preserving integration scheme. It allows for fully deformable simulation
# cells and uses the same underying integrator as integrate.npt (with \b nph=True).
#
# The available options are identical to those of integrate.npt, Except that *T* is fixed at 1.0. For further
# information, refer to its documentation.
#
# \note A time scale \b tau_p for the relaxation of the barostat is required. This is defined as the
#       relaxation time the barostat would have at a temperature \f$ T_0 =1 \f$, and is related to the
#       internally used Barostat mass \f$W\f$ via \f $W=d N T_0 \tau_p^2 \f $, where \f$ d \f$ is the
#       dimensionsality.
#       
# integrate.nph is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
#
# \sa integrate.npt
class nph(npt):
    ## Specifies the NPH integrator
    # \param params Parameters used for the underlying integrate.npt (for documentation, see there)
    #
    # \b Examples:
    # \code
    # # Triclinic unit cell
    # nph=integrate.nph(group=all, P=2.0, tau_p=1.0, couple="none", all=True)
    # # Cubic unit cell
    # nph = integrate.nph(group=all, P=2.0, tau_p=1.0)
    # \endcode
    def __init__(self, **params):
        util.print_status_line();

        # initialize base class
        util._disable_status_lines = True;
        npt.__init__(self, nph=True, T=1.0, **params);
        util._disable_status_lines = False;

## NVE Integration via Velocity-Verlet
#
# integrate.nve performs constant volume, constant energy simulations using the standard
# Velocity-Verlet method. For poor initial conditions that include overlapping atoms, a 
# limit can be specified to the movement a particle is allowed to make in one time step. 
# After a few thousand time steps with the limit set, the system should be in a safe state 
# to continue with unconstrained integration.
#
# Another use-case for integrate.nve is to fix the velocity of a certain group of particles. This can be achieved by
# setting the velocity of those particles in the initial condition and setting the \a zero_force option to True
# for that group. A True value for \a zero_force causes integrate.nve to ignore any net force on each particle and
# integrate them forward in time with a constant velocity.
#
# \note With an active limit, Newton's third law is effectively \b not obeyed and the system 
# can gain linear momentum. Activate the update.zero_momentum updater during the limited nve
# run to prevent this.
#
# integrate.nve is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class nve(_integration_method):
    ## Specifies the NVE integration method
    # \param group Group of particles on which to apply this method.
    # \param limit (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
    # \param zero_force When set to true, particles in the \a group are integrated forward in time with constant
    #                   velocity and any net force on them is ignored.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    #
    # \b Examples:
    # \code
    # all = group.all()
    # integrate.nve(group=all)
    # integrator = integrate.nve(group=all)
    # typeA = group.type('A')
    # integrate.nve(group=typeA, limit=0.01)
    # integrate.nve(group=typeA, zero_force=True)
    # \endcode
    def __init__(self, group, limit=None, zero_force=False):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # create the compute thermo
        compute._get_unique_thermo(group=group);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNVE(globals.system_definition, group.cpp_group, False);
        else:
            self.cpp_method = hoomd.TwoStepNVEGPU(globals.system_definition, group.cpp_group);
        
        # set the limit
        if limit is not None:
            self.cpp_method.setLimit(limit);
        
        self.cpp_method.setZeroForce(zero_force);
        
        self.cpp_method.validateGroup()
        
    ## Changes parameters of an existing integrator
    # \param limit (if set) New limit value to set. Removes the limit if limit is False
    # \param zero_force (if set) New value for the zero force option
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nve(group=all)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(limit=0.01)
    # integrator.set_params(limit=False)
    # \endcode
    def set_params(self, limit=None, zero_force=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if limit is not None:
            if limit == False:
                self.cpp_method.removeLimit();
            else:
                self.cpp_method.setLimit(limit);
        
        if zero_force is not None:
            self.cpp_method.setZeroForce(zero_force);

## NVT integration via Brownian dynamics
#
# integrate.bdnvt performs constant volume, fixed average temperature simulation based on a 
# NVE simulation with added damping and stochastic heat bath forces.
#
# The total added %force \f$ \vec{F}\f$ is
# \f[ \vec{F} = -\gamma \cdot \vec{v} + \vec{F}_{\mathrm{rand}} \f]
# where \f$ \vec{v} \f$ is the particle's velocity and \f$ \vec{F}_{\mathrm{rand}} \f$
# is a random force with magnitude chosen via the fluctuation-dissipation theorem
# to be consistent with the specified drag (\a gamma) and temperature (\a T).
# 
# For poor initial conditions that include overlapping atoms, a 
# limit can be specified to the movement a particle is allowed to make in one time step. 
# After a few thousand time steps with the limit set, the system should be in a safe state 
# to continue with unconstrained integration.
#
# \note With an active limit, Newton's third law is effectively \b not obeyed and the system 
# can gain linear momentum. Activate the update.zero_momentum updater during the limited bdnvt
# run to prevent this.
#
# integrate.bdnvt is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
#
# integrate.bdnvt uses the proper number of degrees of freedom to compute the temperature of the system in both
# 2 and 3 dimensional systems, as long as the number of dimensions is set before the integrate.bdnvt command
# is specified.
#
class bdnvt(_integration_method):
    ## Specifies the BD NVT integrator
    # \param group Group of particles on which to apply this method.
    # \param T Temperature of the simulation (in energy units)
    # \param seed Random seed to use for the run. Simulations that are identical, except for the seed, will follow 
    # different trajectories.
    # \param gamma_diam If True, then then gamma for each particle will be assigned to its diameter. If False (the
    #                   default), gammas are assigned per particle type via set_gamma().
    # \param limit (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
    # \param tally (optional) If true, the energy exchange between the bd thermal reservoir and the particles is
    #                         tracked. Total energy conservation can then be monitored by adding
    #                         \b bdnvt_reservoir_energy_<i>groupname</i> to the logged quantities.
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # Internally, a compute.thermo is automatically specified and associated with \a group.
    #
    # \warning If starting from a restart binary file, the energy of the reservoir will be reset to zero.  
    # \b Examples:
    # \code
    # all = group.all();
    # integrate.bdnvt(group=all, T=1.0, seed=5)
    # integrator = integrate.bdnvt(group=all, T=1.0, seed=100)
    # integrate.bdnvt(group=all, T=1.0, limit=0.01, gamma_diam=1, tally=True)
    # typeA = group.type('A');
    # integrate.bdnvt(group=typeA, T=variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    # \endcode
    def __init__(self, group, T, seed=0, gamma_diam=False, limit=None, tally=False):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        
        # create the compute thermo
        compute._get_unique_thermo(group=group);
        
        # setup suffix
        suffix = '_' + group.name;
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepBDNVT(globals.system_definition, group.cpp_group, T.cpp_variant, seed, gamma_diam, suffix);
        else:
            self.cpp_method = hoomd.TwoStepBDNVTGPU(globals.system_definition, group.cpp_group, T.cpp_variant, seed, gamma_diam, suffix);
       
        self.cpp_method.setTally(tally);
        
        # set the limit
        if limit is not None:
            self.cpp_method.setLimit(limit);
    
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    # \param tally (optional) If true, the energy exchange between the bd thermal reservoir and the particles is
    #                         tracked. Total energy conservation can then be monitored by adding
    #                         \b bdnvt_reservoir_energy_<i>groupname</i> to the logged quantities.
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.bdnvt(group=all, T=1.0)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(T=2.0)
    # integrator.set_params(tally=False)
    # \endcode
    def set_params(self, T=None, tally=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        
        if tally is not None:
            self.cpp_method.setTally(tally);

    ## Sets gamma parameter for a particle type
    # \param a Particle type
    # \param gamma \f$ \gamma \f$ for particle type \a (in units of force/velocity)
    #
    # set_gamma() sets the coefficient \f$ \gamma \f$ for a single particle type, identified
    # by name.
    #
    # The gamma parameter determines how strongly a particular particle is coupled to 
    # the stochastic bath.  The higher the gamma, the more strongly coupled: see 
    # integrate.bdnvt.
    #
    # If gamma is not set for any particle type, it will automatically default to  1.0.
    # It is not an error to specify gammas for particle types that do not exist in the simulation.
    # This can be useful in defining a single simulation script for many different types of particles 
    # even when some simulations only include a subset.
    #
    # \b Examples:
    # \code
    # bd.set_gamma('A', gamma=2.0)
    # \endcode
    #
    def set_gamma(self, a, gamma):
        util.print_status_line();
        self.check_initialization();
        
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma);
        
                
## NVE Integration for rigid bodies
#
# integrate.nve_rigid performs constant volume, constant energy simulations on rigid bodies
# The integration scheme is implemented from \cite Miller2002 .
#
# Reference \cite Nguyen2011 describes the rigid body implementation details in HOOMD-blue. Cite it
# if you utilize rigid body functionality in your work.
#
# integrate.nve_rigid \b only operates on particles that belong to rigid bodies.
#
# integrate.nve_rigid is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class nve_rigid(_integration_method):
    ## Specifies the NVE integration method for rigid bodies 
    # \param group Group of particles on which to apply this method.
    #
    # \b Examples:
    # \code
    # rigid = group.rigid()
    # integrate.nve_rigid(group=rigid)
    # \endcode
    def __init__(self, group):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNVERigid(globals.system_definition, group.cpp_group);
        else:
            self.cpp_method = hoomd.TwoStepNVERigidGPU(globals.system_definition, group.cpp_group);

        self.cpp_method.validateGroup()

## NVT Integration for rigid bodies
#
# integrate.nvt_rigid performs constant volume, constant temperature simulations of the rigid bodies in the system.
# The integration scheme is implemented from \cite Miller2002 and \cite Kamberaj2005 .
#
# Reference \cite Nguyen2011 describes the rigid body implementation details in HOOMD-blue. Cite it
# if you utilize rigid body functionality in your work.
#
# integrate.nvt_rigid \b only operates on particles that belong to rigid bodies.
#
# integrate.nvt_rigid is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class nvt_rigid(_integration_method):
    ## Specifies the NVT integration method for rigid bodies
    # \param group Group of particles on which to apply this method.
    # \param T Temperature set point for the thermostat (in energy units)
    # \param tau Time constant for the thermostat (in time units)
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # \b Examples:
    # \code
    # rigid = group.rigid()
    # integrate.nvt_rigid(group=all, T=1.0, tau=10.0)
    # integrator = integrate.nvt_rigid(group=all, tau=5.0, T=1.0)
    # \endcode
    def __init__(self, group, T, tau):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        
        # create the compute thermo
        thermo = compute._get_unique_thermo(group=group);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNVTRigid(globals.system_definition, group.cpp_group, thermo.cpp_compute, T.cpp_variant);
        else:
            self.cpp_method = hoomd.TwoStepNVTRigidGPU(globals.system_definition, group.cpp_group, thermo.cpp_compute, T.cpp_variant);
    
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    # \param tau New coupling constant (if set) (in time units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nvt_rigid(group=rigid, T=0.65)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(T=2.0, tau=10.0)
    # \endcode
    def set_params(self, T=None, tau=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        if tau is not None:
            self.cpp_method.setTau(tau);  
              
## NVT integration via Brownian dynamics for rigid bodies
#
# integrate.bdnvt_rigid performs constant volume, constant temperature simulation based on a
# NVE simulation with added damping and stochastic heat bath forces. The NVE integration scheme
# is implemented from ref \cite Miller2002.
#
# Reference \cite Nguyen2011 describes the rigid body implementation details in HOOMD-blue. Cite it
# if you utilize rigid body functionality in your work.
#
# The total added %force \f$ \vec{F}\f$ <strong>applied to each constiuent particle</strong> is
# \f[ \vec{F} = -\gamma \cdot \vec{v} + \vec{F}_{\mathrm{rand}} \f]
# where \f$ \vec{v} \f$ is the particle's velocity and \f$ \vec{F}_{\mathrm{rand}} \f$
# is a random force with magnitude chosen via the fluctuation-dissipation theorem
# to be consistent with the specified drag (\a gamma) and temperature (\a T).
#
# integrate.bdnvt_rigid \b only operates on particles that belong to rigid bodies.
#
# integrate.bdnvt_rigid is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class bdnvt_rigid(_integration_method):
    ## Specifies the BD NVT integrator for rigid bodies
    # \param group Group of particles on which to apply this method.
    # \param T Temperature of the simulation \a T (in energy units)
    # \param seed Random seed to use for the run. Simulations that are identical, except for the seed, will follow 
    # different trajectories.
    # \param gamma_diam If True, then then gamma for each particle will be assigned to its diameter. If False (the
    #                   default), gammas are assigned per particle type via set_gamma().
    #
    # \a T can be a variant type, allowing for temperature ramps in simulation runs.
    #
    # \b Examples:
    # \code
    # rigid = group.rigid();
    # integrate.bdnvt_rigid(group=rigid, T=1.0, seed=5)
    # integrator = integrate.bdnvt_rigid(group=all, T=1.0, seed=100)
    # integrate.bdnvt_rigid(group=all, T=1.0, gamma_diam=True)
    # \endcode
    def __init__(self, group, T, seed=0, gamma_diam=False):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepBDNVTRigid(globals.system_definition, group.cpp_group, T.cpp_variant, seed, gamma_diam);
        else:
            self.cpp_method = hoomd.TwoStepBDNVTRigidGPU(globals.system_definition, group.cpp_group, T.cpp_variant, seed, gamma_diam);
        
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.bdnvt_rigid(group=rigid, T=1.0)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(T=2.0)
    # \endcode
    def set_params(self, T=None):
        util.print_status_line();
        self.check_initialization();
                
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);

    ## Sets gamma parameter for a particle type
    # \param a Particle type
    # \param gamma \f$ \gamma \f$ for particle type (see below for examples) (in units of force/velocity)
    #
    # set_gamma() sets the coefficient \f$ \gamma \f$ for a single particle type, identified
    # by name.
    #
    # The gamma parameter determines how strongly a particular particle is coupled to 
    # the stochastic bath.  The higher the gamma, the more strongly coupled: see 
    # integrate.bdnvt_rigid.
    #
    # If gamma is not set for any particle type, it will automatically default to  1.0.
    # It is not an error to specify gammas for particle types that do not exist in the simulation.
    # This can be useful in defining a single simulation script for many different types of particles 
    # even when some simulations only include a subset.
    #
    # \b Examples:
    # \code
    # bd.set_gamma('A', gamma=2.0)
    # \endcode
    #
    def set_gamma(self, a, gamma):
        util.print_status_line();
        self.check_initialization();
        
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma);

## NPT Integration for rigid bodies
#
# integrate.npt_rigid performs constant pressure, constant temperature simulations of the rigid bodies in the system.
# The integration scheme is implemented from \cite Miller2002 and \cite Kamberaj2005 .
#
# Reference \cite Nguyen2011 describes the rigid body implementation details in HOOMD-blue. Cite it
# if you utilize rigid body functionality in your work.
#
# integrate.npt_rigid \b only operates on particles that belong to rigid bodies.
#
# integrate.npt_rigid is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class npt_rigid(_integration_method):
    ## Specifies the NVT integration method for rigid bodies
    # \param group Group of particles on which to apply this method.
    # \param tau Time constant for the thermostat (in time units)
    # \param tauP Time constatnt for the barostat (in time units)
    # \param T Temperature set point for the thermostat (in energy units)
    # \param P Pressure set point for the barostat (in pressure units)
    #
    # \a T (and P) can be a variant type, allowing for temperature (and pressure) ramps in simulation runs.
    #
    # \b Examples:
    # \code
    # rigid = group.rigid()
    # integrate.npt_rigid(group=all, T=1.0, tau=10.0, P=1.0, tauP=1.0)
    # 
    # \endcode
    def __init__(self, group, T, tau, P, tauP):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        T = variant._setup_variant_input(T);
        P = variant._setup_variant_input(P);
        
         # create the compute thermo
        thermo_group = compute._get_unique_thermo(group=group);
        thermo_all = compute._get_unique_thermo(group=globals.group_all);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNPTRigid(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, thermo_all.cpp_compute, tau, tauP, T.cpp_variant, P.cpp_variant);
        else:
            self.cpp_method = hoomd.TwoStepNPTRigidGPU(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, thermo_all.cpp_compute, tau, tauP, T.cpp_variant, P.cpp_variant);
    
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param T New temperature (if set) (in energy units)
    # \param tau New coupling constant (if set) (in time units)
    # \param P New pressure (if set) (in pressure units)
    # \param tauP New coupling constant (if set) (in time units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.npt_rigid(group=rigid, T=0.65, P=1.0)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(T=2.0, tau=10.0, P=1.5)
    # \endcode
    def set_params(self, T=None, tau=None, P=None, tauP=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if T is not None:
            # setup the variant inputs
            T = variant._setup_variant_input(T);
            self.cpp_method.setT(T.cpp_variant);
        if tau is not None:
            self.cpp_method.setTau(tau); 
        if P is not None:
            # setup the variant inputs
            P = variant._setup_variant_input(P);
            self.cpp_method.setP(P.cpp_variant);
        if tauP is not None:
            self.cpp_method.setTauP(tauP); 

## NPH Integration for rigid bodies
#
# integrate.nph_rigid performs constant pressure, constant enthalpy simulations of the rigid bodies in the system.
# The integration scheme is implemented from \cite Miller2002 and \cite Kamberaj2005 .
#
# Reference \cite Nguyen2011 describes the rigid body implementation details in HOOMD-blue. Cite it
# if you utilize rigid body functionality in your work.
#
# integrate.nph_rigid \b only operates on particles that belong to rigid bodies.
#
# integrate.nph_rigid is an integration method. It must be used in concert with an integration mode. It can be used while
# the following modes are active:
# - integrate.mode_standard
class nph_rigid(_integration_method):
    ## Specifies the NPH integration method for rigid bodies
    # \param group Group of particles on which to apply this method.
    # \param tauP Time constatnt for the barostat (in time units)
    # \param P Pressure set point for the barostat (in pressure units)
    #
    # \a P can be a variant type, allowing for pressure ramping in simulation runs.
    #
    # \b Examples:
    # \code
    # rigid = group.rigid()
    # integrate.nph_rigid(group=all, P=1.0, tauP=1.0)
    # 
    # \endcode
    def __init__(self, group, P, tauP):
        util.print_status_line();
        
        # initialize base class
        _integration_method.__init__(self);
        
        # setup the variant inputs
        P = variant._setup_variant_input(P);
        
         # create the compute thermo
        thermo_group = compute._get_unique_thermo(group=group);
        thermo_all = compute._get_unique_thermo(group=globals.group_all);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepNPHRigid(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, thermo_all.cpp_compute, tauP, P.cpp_variant);
        else:
            self.cpp_method = hoomd.TwoStepNPHRigidGPU(globals.system_definition, group.cpp_group, thermo_group.cpp_compute, thermo_all.cpp_compute, tauP, P.cpp_variant);
    
        self.cpp_method.validateGroup()
    
    ## Changes parameters of an existing integrator
    # \param P New pressure (if set) (in pressure units)
    # \param tauP New coupling constant (if set) (in time units)
    #
    # To change the parameters of an existing integrator, you must save it in a variable when it is
    # specified, like so:
    # \code
    # integrator = integrate.nph_rigid(group=rigid, P=1.0)
    # \endcode
    #
    # \b Examples:
    # \code
    # integrator.set_params(P=1.5, tauP=1.0)
    # \endcode
    def set_params(self, P=None, tauP=None):
        util.print_status_line();
        self.check_initialization();
        
        # change the parameters
        if P is not None:
            # setup the variant inputs
            P = variant._setup_variant_input(P);
            self.cpp_method.setP(P.cpp_variant);
        if tauP is not None:
            self.cpp_method.setTauP(tauP); 

## Energy Minimizer (FIRE)
#
# integrate.mode_minimize_fire uses the Fast Inertial Relaxation Engine (FIRE) algorithm to minimize the energy
# for a group of particles while keeping all other particles fixed.  This method is published in
# Bitzek, et al, PRL, 2006.
#
# At each time step,\f$\Delta t \f$, the algorithm uses the NVE Integrator to generate a x, v, and F, and then adjusts
# v according to \f[ \vec{v} = (1-\alpha)\vec{v} + \alpha \hat{F}|\vec{v}|  \f] where \f$ \alpha \f$ and \f$\Delta t \f$
# are dynamically adaptive quantities.  While a current search has been lowering the energy of system for more than 
# \f$N_{min}\f$ steps, \f$ \alpha \f$  is decreased by \f$ \alpha \rightarrow \alpha f_{alpha} \f$ and
# \f$\Delta t \f$ is increased by \f$ \Delta t \rightarrow max(\Delta t * f_{inc}, \Delta t_{max}) \f$.
# If the energy of the system increases (or stays the same), the velocity of the particles is set to 0,
# \f$ \alpha \rightarrow \alpha_{start}\f$ and
# \f$ \Delta t \rightarrow \Delta t * f_{dec} \f$.  Convergence is determined by both the force per particle or the 
# change in energy per particle dropping below \a ftol and \a Etol, respectively or,
# 
# \f[ \frac{\sum |F|}{N*\sqrt{N_{dof}}} <ftol \;\; and \;\; \Delta \frac{\sum |E|}{N} < Etol  \f]
# where N is the number of particles the minimization is acting over (i.e. the group size).  Either of the two criterion can be effectively turned off by setting the tolerance to a large number.
#
# If the minimization is acted over a subset of all the particles in the system, the "other" particles will be kept
# frozen but will still interact with the particles being moved.
#
# \b Example:
# \code
# fire=integrate.mode_minimize_fire( group=group.all(), dt=0.05, ftol=1e-2, Etol=1e-7)
# while not(fire.has_converged()):
#    xml = dump.xml(filename="dump",period=1)
#    run(100)
# \endcode
#
# \note As a default setting, the algorithm will start with a \f$ \Delta t = \frac{1}{10} \Delta t_{max} \f$ and
# attempts at least 10 search steps.  In practice, it was found that this prevents the simulation from making too
# aggressive a first step, but also from quitting before having found a good search direction. The minimum number of
# attempts can be set by the user. 
#
# \warning All other integration methods must be disabled before using the FIRE energy minimizer.
class mode_minimize_fire(_integrator):
    ## Specifies the FIRE energy minimizer.
    # \param group Particle group to be applied FIRE 
    # \param group Group of particles on which to apply this method.
    # \param dt This is the maximum step size the minimizer is permitted to use.  Consider the stability of the system when setting. (in time units)
    # \param Nmin Number of steps energy change is negative before allowing \f$ \alpha \f$ and \f$ \Delta t \f$ to adapt. 
    #   - <i>optional</i>: defaults to 5
    # \param finc Factor to increase \f$ \Delta t \f$ by 
    #   - <i>optional</i>: defaults to 1.1
    # \param fdec Factor to decrease \f$ \Delta t \f$ by 
    #   - <i>optional</i>: defaults to 0.5
    # \param alpha_start Initial (and maximum) \f$ \alpha \f$ 
    #   - <i>optional</i>: defaults to 0.1
    # \param falpha Factor to decrease \f$ \alpha t \f$ by 
    #   - <i>optional</i>: defaults to 0.99
    # \param ftol force convergence criteria (in force units)
    #   - <i>optional</i>: defaults to 1e-1
    # \param Etol energy convergence criteria (in energy units)
    #   - <i>optional</i>: defaults to 1e-5
    # \param min_steps A minimum number of attempts before convergence criteria are considered 
    #   - <i>optional</i>: defaults to 10
    def __init__(self, group, dt, Nmin=None, finc=None, fdec=None, alpha_start=None, falpha=None, ftol = None, Etol= None, min_steps=None):
        util.print_status_line();
        
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_integrator = hoomd.FIREEnergyMinimizer(globals.system_definition, group.cpp_group, dt);
        else:
            self.cpp_integrator = hoomd.FIREEnergyMinimizerGPU(globals.system_definition, group.cpp_group, dt);

        self.supports_methods = False;
        
        globals.system.setIntegrator(self.cpp_integrator);        
        
        # change the set parameters if not None
        if not(Nmin is None):
            self.cpp_integrator.setNmin(Nmin);
        if not(finc is None):
            self.cpp_integrator.setFinc(finc);
        if not(fdec is None):
            self.cpp_integrator.setFdec(fdec);
        if not(alpha_start is None):
            self.cpp_integrator.setAlphaStart(alpha_start);
        if not(falpha is None):
            self.cpp_integrator.setFalpha(falpha);
        if not(ftol is None):
            self.cpp_integrator.setFtol(ftol);
        if not(Etol is None):
            self.cpp_integrator.setEtol(Etol); 
        if not(min_steps is None):
            self.cpp_integrator.setMinSteps(min_steps);               
            
    ## Asks if Energy Minimizer has converged
    #
    def has_converged(self):
        self.check_initialization();
        return self.cpp_integrator.hasConverged()
 
## Energy Minimizer RIGID (FIRE)
#
# integrate.energyminimizer_FIRE uses the Fast Inertial Relaxation Engine (FIRE) algorithm to minimize the energy
# for a group of particles while keeping all other particles fixed.
#
# For the time being, energy minimization will be handled separately for rigid and non-rigid bodies
#
class mode_minimize_rigid_fire(_integrator):
    ## Specifies the FIRE energy minimizer.
    #
    # \param group Group of particles in rigid bodies
    # \param dt This is the maximum timestep the minimizer is permitted to use.  Consider the stability of the system when setting. (in time units)
    # \param Nmin Number of steps energy change is negative before allowing \f$ \alpha \f$ and \f$ \Delta t \f$ to adapt. 
    #   - <i>optional</i>: defaults to 5
    # \param finc Factor to increase \f$ \Delta t \f$ by 
    #   - <i>optional</i>: defaults to 1.1
    # \param fdec Factor to decrease \f$ \Delta t \f$ by 
    #   - <i>optional</i>: defaults to 0.5
    # \param alpha_start Initial (and maximum) \f$ \alpha \f$ 
    #   - <i>optional</i>: defaults to 0.1
    # \param falpha Factor to decrease \f$ \alpha t \f$ by 
    #   - <i>optional</i>: defaults to 0.99
    # \param ftol force convergence criteria 
    #   - <i>optional</i>: defaults to 1e-1
    # \param wtol torque convergence criteria 
    #   - <i>optional</i>: defaults to 1e-1    
    # \param Etol energy convergence criteria 
    #   - <i>optional</i>: defaults to 1e-5
    def __init__(self, group, dt, Nmin=None, finc=None, fdec=None, alpha_start=None, falpha=None, ftol = None, wtol=None, Etol= None):
        util.print_status_line();
        
        # initialize base class
        _integrator.__init__(self);
        
        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_integrator = hoomd.FIREEnergyMinimizerRigid(globals.system_definition, group.cpp_group, dt);
        else:
            self.cpp_integrator = hoomd.FIREEnergyMinimizerRigidGPU(globals.system_definition, group.cpp_group, dt);

        self.supports_methods = False;
        
        globals.system.setIntegrator(self.cpp_integrator);        
        
        # change the set parameters if not None
        if not(Nmin is None):
            self.cpp_integrator.setNmin(Nmin);
        if not(finc is None):
            self.cpp_integrator.setFinc(finc);
        if not(fdec is None):
            self.cpp_integrator.setFdec(fdec);
        if not(alpha_start is None):
            self.cpp_integrator.setAlphaStart(alpha_start);
        if not(falpha is None):
            self.cpp_integrator.setFalpha(falpha);
        if not(ftol is None):
            self.cpp_integrator.setFtol(ftol);
        if not(wtol is None):
            self.cpp_integrator.setWtol(wtol);            
        if not(Etol is None):
            self.cpp_integrator.setEtol(Etol);
            
    ## Asks if Energy Minimizer has converged
    #
    def has_converged(self):
        self.check_initialization();
        return self.cpp_integrator.hasConverged()

## Applies the Berendsen thermostat.
#
# integrate.berendsen rescales the velocities of all particles on each time step. The rescaling is performed so that
# the difference in the current temperature from the set point decays exponentially \cite Berendsen1984
# \f[
#     \frac{dT_\mathrm{cur}}{dt} = \frac{T - T_\mathrm{cur}}{\tau}
# \f]
#
class berendsen(_integration_method):
    ## Initialize the Berendsen thermostat.
    # \param group Group to which the Berendsen thermostat will be applied.
    # \param T Temperature of thermostat. (in energy units)
    # \param tau Time constant of thermostat. (in time units)
    #
    def __init__(self, group, T, tau):
        util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        T = variant._setup_variant_input(T);

        # create the compute thermo
        thermo = compute._get_unique_thermo(group = group);

        # initialize the reflected c++ class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_method = hoomd.TwoStepBerendsen(globals.system_definition,
                                                     group.cpp_group,
                                                     thermo.cpp_compute,
                                                     tau,
                                                     T.cpp_variant);
        else:
            self.cpp_method = hoomd.TwoStepBerendsenGPU(globals.system_definition,
                                                        group.cpp_group,
                                                        thermo.cpp_compute,
                                                        tau,
                                                        T.cpp_variant);

