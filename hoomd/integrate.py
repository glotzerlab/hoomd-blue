# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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
# An integrator in hoomd reflects an Integrator in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ integrator itself is tracked 2) All
# forces created so far in the simulation are updated in the cpp_integrator
# whenever run() is called.
class _integrator(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the integrator
    #
    # This doesn't really do much bet set some member variables to None
    def __init__(self):
        # check if initialization has occurred
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
            hoomd.context.msg.error('Bug in hoomd.integrate: cpp_integrator not set, please report\n');
            raise RuntimeError();

    ## \internal
    # \brief Updates the integrators in the reflected c++ class
    def update_forces(self):
        self.check_initialization();

        # set the forces
        self.cpp_integrator.removeForceComputes();
        for f in hoomd.context.current.forces:
            if f.cpp_force is None:
                hoomd.context.msg.error('Bug in hoomd.integrate: cpp_force not set, please report\n');
                raise RuntimeError('Error updating forces');

            if f.log or f.enabled:
                f.update_coeffs();

            if f.enabled:
                self.cpp_integrator.addForceCompute(f.cpp_force);

        # set the constraint forces
        for f in hoomd.context.current.constraint_forces:
            if f.cpp_force is None:
                hoomd.context.msg.error('Bug in hoomd.integrate: cpp_force not set, please report\n');
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

    @classmethod
    def _gsd_state_name(cls):
        raise NotImplementedError("GSD Schema is not implemented for {}".format(cls.__name__));

    def _connect_gsd(self, gsd):
        # This is an internal method, and should not be called directly. See gsd.dump_state() instead
        if isinstance(gsd, hoomd.dump.gsd) and hasattr(self.cpp_integrator, "connectGSDStateSignal"):
            self.cpp_integrator.connectGSDStateSignal(gsd.cpp_analyzer, self._gsd_state_name());
        else:
            raise NotImplementedError("GSD Schema is not implemented for {}".format(self.__class__.__name__));

    def _connect_gsd_shape_spec(self, gsd):
        # This is an internal method, and should not be called directly. See gsd.dump_shape() instead
        if isinstance(gsd, hoomd.dump.gsd) and hasattr(self.cpp_integrator, "connectGSDShapeSpec"):
            self.cpp_integrator.connectGSDShapeSpec(gsd.cpp_analyzer);
        else:
            raise NotImplementedError("GSD Schema is not implemented for {}".format(self.__class__.__name__));

    def restore_state(self):
        """ Restore the state information from the file used to initialize the simulations
        """
        hoomd.util.print_status_line();
        if isinstance(hoomd.context.current.state_reader, _hoomd.GSDReader) and hasattr(self.cpp_integrator, "restoreStateGSD"):
            self.cpp_integrator.restoreStateGSD(hoomd.context.current.state_reader, self._gsd_state_name());
        else:
            if hoomd.context.current.state_reader is None:
                hoomd.context.msg.error("Can only restore after the state reader has been initialized.\n");
            else:
                hoomd.context.msg.error("Restoring state from {reader_name} is not currently supported for {name}\n".format(reader_name=hoomd.context.current.state_reader.__name__, name=self.__class__.__name__));
            raise RuntimeError("Can not restore state information!");

## \internal
# \brief Base class for integration methods
#
# An integration_method in hoomd.integrate reflects an IntegrationMethod in c++. It is responsible for all high-level
# management that happens behind the scenes for hoomd.integrate writers. 1) The instance of the c++ integration method
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
        # check if initialization has occurred
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
            hoomd.context.msg.error('Bug in hoomd.integrate: cpp_method not set, please report\n');
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
        hoomd.context.current.integration_methods.append(self);

    ## \internal
    # \brief Override get_metadata() to add 'enabled' field
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        return data
