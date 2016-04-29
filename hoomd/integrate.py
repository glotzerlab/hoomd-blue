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

## \package hoomd.integrate
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

from hoomd import _hoomd;
import hoomd;
import copy;
import sys;

## \internal
# \brief Base class for integrators
#
# An integrator in hoomd_script reflects an Integrator in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ integrator itself is tracked 2) All
# forces created so far in the simulation are updated in the cpp_integrator
# whenever run() is called.
class _integrator(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the integrator
    #
    # This doesn't really do much bet set some member variables to None
    def __init__(self):
        # check if initialization has occured
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create integrator before initialization\n");
            raise RuntimeError('Error creating integrator');

        # by default, integrators do not support methods
        self.cpp_integrator = None;
        self.supports_methods = False;

        # save ourselves in the global variable
        hoomd.context.current.integrator = self;

        # base class constructor
        hoomd.meta._metadata.__init__(self)

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
            hoomd.context.msg.error('Bug in hoomd_script: cpp_integrator not set, please report\n');
            raise RuntimeError();

    ## \internal
    # \brief Updates the integrators in the reflected c++ class
    def update_forces(self):
        self.check_initialization();

        # set the forces
        self.cpp_integrator.removeForceComputes();
        for f in hoomd.context.current.forces:
            if f.cpp_force is None:
                hoomd.context.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');

            if f.log or f.enabled:
                f.update_coeffs();

            if f.enabled:
                self.cpp_integrator.addForceCompute(f.cpp_force);

        # set the constraint forces
        for f in hoomd.context.current.constraint_forces:
            if f.cpp_force is None:
                hoomd.context.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');

            if f.enabled:
                self.cpp_integrator.addForceConstraint(f.cpp_force);

                # register any composite body forces
                if f.composite:
                    self.cpp_integrator.addForceComposite(f.cpp_force);

                f.update_coeffs();


    ## \internal
    # \brief Updates the integration methods in the reflected c++ class
    def update_methods(self):
        self.check_initialization();

        # if we support methods, add them all to the list
        if self.supports_methods:
            self.cpp_integrator.removeAllIntegrationMethods();

            if len(hoomd.context.current.integration_methods) == 0:
                hoomd.context.msg.error('This integrator requires that one or more integration methods be specified.\n');
                raise RuntimeError('Error initializing integrator methods');

            for m in hoomd.context.current.integration_methods:
                self.cpp_integrator.addIntegrationMethod(m.cpp_method);

        else:
            if len(hoomd.context.current.integration_methods) > 0:
                hoomd.context.msg.error("This integrator does not support the use of integration methods,\n");
                hoomd.context.msg.error("but some have been specified in the script. Remove them or use\n");
                hoomd.context.msg.error("a different integrator.\n");
                raise RuntimeError('Error initializing integrator methods');

    ## \internal
    # \brief Counts the number of degrees of freedom and updates each hoomd.compute.thermo specified
    def update_thermos(self):
        self.check_initialization();

        for t in hoomd.context.current.thermos:
            ndof = self.cpp_integrator.getNDOF(t.group.cpp_group);
            t.cpp_compute.setNDOF(ndof);

            ndof_rot = self.cpp_integrator.getRotationalNDOF(t.group.cpp_group);
            t.cpp_compute.setRotationalNDOF(ndof_rot);


## \internal
# \brief Base class for integration methods
#
# An integration_method in hoomd_script reflects an IntegrationMethod in c++. It is responsible for all high-level
# management that happens behind the scenes for hoomd_script writers. 1) The instance of the c++ integration method
# itself is tracked and added to the integrator and 2) methods are provided for disabling the integration method from
# being active for the next run()
#
# The design of integration_method exactly mirrors that of _force for consistency
class _integration_method(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the integration_method
    #
    # Initializes the cpp_method to None.
    def __init__(self):
        # check if initialization has occured
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create an integration method before initialization\n");
            raise RuntimeError('Error creating integration method');

        self.cpp_method = None;

        self.enabled = True;
        hoomd.context.current.integration_methods.append(self);

        # base class constructor
        hoomd.meta._metadata.__init__(self)

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
            hoomd.context.msg.error('Bug in hoomd_script: cpp_method not set, please report\n');
            raise RuntimeError();

    def disable(self):
        R""" Disables the integration method.

        Examples::

            method.disable()

        Executing the disable command will remove the integration method from the simulation.
        Any :py:func:`hoomd.run()` command executed after disabling an integration method will
        not apply the integration method to the particles during the
        simulation. A disabled integration method can be re-enabled with :py:meth:`enable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization()

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable an integration method that is already disabled");
            return;

        self.enabled = False;
        hoomd.context.current.integration_methods.remove(self);

    def enable(self):
        R""" Enables the integration method.

        Examples::

            method.enable()

        See Also:
            :py:meth:`disable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable an integration method that is already enabled");
            return;

        self.enabled = True;
        hoomd.context.integration_methods.append(self);

    ## \internal
    # \brief Override get_metadata() to add 'enabled' field
    def get_metadata(self):
        data = meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        return data
