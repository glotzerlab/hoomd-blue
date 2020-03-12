# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Modify the system state periodically.

When an updater is specified, it acts on the particle system every *period* steps to change
it in some way. See the documentation of specific updaters to find out what they do.
"""

from hoomd import _hoomd;
import hoomd;
import sys;

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
            raise RuntimeError('Cannot create updater before initialization\n');

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
    def setupUpdater(self, period, phase=0):
        self.phase = phase;

        if type(period) == type(1.0):
            period = int(period);

        if type(period) == type(1):
            hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, period, phase);
        elif type(period) == type(lambda n: n*2):
            hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, 1000, -1);
            hoomd.context.current.system.setUpdaterPeriodVariable(self.updater_name, period);
        else:
            hoomd.context.current.device.cpp_msg.error("I don't know what to do with a period of type " + str(type(period)) + "expecting an int or a function\n");
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
            hoomd.context.current.device.cpp_msg.error('Bug in hoomd. cpp_updater not set, please report\n');
            raise RuntimeError();

    def disable(self):
        R""" Disables the updater.

        Examples::

            updater.disable()

        Executing the disable command will remove the updater from the system.
        Any :py:func:`hoomd.run()` command executed after disabling an updater will not use that
        updater during the simulation. A disabled updater can be re-enabled
        with :py:meth:`enable()`
        """
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.current.device.cpp_msg.warning("Ignoring command to disable an updater that is already disabled");
            return;

        self.prev_period = hoomd.context.current.system.getUpdaterPeriod(self.updater_name);
        hoomd.context.current.system.removeUpdater(self.updater_name);
        hoomd.context.current.updaters.remove(self)
        self.enabled = False;

    def enable(self):
        R""" Enables the updater.

        Examples::

            updater.enable()

        See Also:
            :py:meth:`disable()`
        """

        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.current.device.cpp_msg.warning("Ignoring command to enable an updater that is already enabled");
            return;

        hoomd.context.current.system.addUpdater(self.cpp_updater, self.updater_name, self.prev_period, self.phase);
        hoomd.context.current.updaters.append(self)
        self.enabled = True;

    def set_period(self, period):
        R""" Changes the updater period.

        Args:
            period (int): New period to set.

        Examples::

            updater.set_period(100);
            updater.set_period(1);

        While the simulation is running, the action of each updater
        is executed every *period* time steps. Changing the period does
        not change the phase set when the analyzer was first created.
        """


        if type(period) == type(1.0):
            period = int(period);

        if type(period) == type(1):
            if self.enabled:
                hoomd.context.current.system.setUpdaterPeriod(self.updater_name, period, self.phase);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            hoomd.context.current.device.cpp_msg.warning("A period cannot be changed to a variable one");
        else:
            hoomd.context.current.device.cpp_msg.warning("I don't know what to do with a period of type " + str(type(period)) + " expecting an int or a function");

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled

        return data

    @classmethod
    def _gsd_state_name(cls):
        raise NotImplementedError("GSD Schema is not implemented for {}".format(cls.__name__));

    def _connect_gsd(self, gsd):
        # This is an internal method, and should not be called directly. See gsd.dump_state() instead
        if isinstance(gsd, hoomd.dump.gsd) and hasattr(self.cpp_updater, "connectGSDStateSignal"):
            self.cpp_updater.connectGSDStateSignal(gsd.cpp_analyzer, self._gsd_state_name());
        else:
            raise NotImplementedError("GSD Schema is not implemented for {}".format(self.__class__.__name__));

    def restore_state(self):
        """ Restore the state information from the file used to initialize the simulations
        """
        if isinstance(hoomd.context.current.state_reader, _hoomd.GSDReader) and hasattr(self.cpp_updater, "restoreStateGSD"):
            self.cpp_updater.restoreStateGSD(hoomd.context.current.state_reader, self._gsd_state_name());
        else:
            if hoomd.context.current.state_reader is None:
                hoomd.context.current.device.cpp_msg.error("Can only restore after the state reader has been initialized.\n");
            else:
                hoomd.context.current.device.cpp_msg.error("Restoring state from {reader_name} is not currently supported for {name}\n".format(reader_name=hoomd.context.current.state_reader.__name__, name=self.__class__.__name__));
            raise RuntimeError("Can not restore state information!");

class box_resize(_updater):
    R""" Rescale the system box size.

    Args:
        L (:py:mod:`hoomd.variant`): (if set) box length in the x,y, and z directions as a function of time (in distance units)
        Lx (:py:mod:`hoomd.variant`): (if set) box length in the x direction as a function of time (in distance units)
        Ly (:py:mod:`hoomd.variant`): (if set) box length in the y direction as a function of time (in distance units)
        Lz (:py:mod:`hoomd.variant`): (if set) box length in the z direction as a function of time (in distance units)
        xy (:py:mod:`hoomd.variant`): (if set) X-Y tilt factor as a function of time (dimensionless)
        xz (:py:mod:`hoomd.variant`): (if set) X-Z tilt factor as a function of time (dimensionless)
        yz (:py:mod:`hoomd.variant`): (if set) Y-Z tilt factor as a function of time (dimensionless)
        period (int): The box size will be updated every *period* time steps.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.
        scale_particles (bool): When True (the default), scale particles into the new box. When False, do not change particle positions when changing the box.

    Every *period* time steps, the system box dimensions is updated to values given by
    the user (in a variant). As an option, the particles can either be left in place
    as the box is changed or their positions can be scaled with the box.

    Note:
        If *period* is set to None, then the given box lengths are applied immediately and
        periodic updates are not performed.

    L, Lx, Ly, Lz, xy, xz, yz can either be set to a constant number or a :py:mod:`hoomd.variant`.
    if any of the box parameters are not specified, they are set to maintain the same value in the
    current box.

    Use L as a shorthand to specify Lx, Ly, and Lz to the same value.

    By default, particle positions are rescaled with the box. Set *scale_particles=False*
    to leave particles in place when changing the box.

    If, under rescaling, tilt factors get too large, the simulation may slow down due
    to too many ghost atoms being communicated. :py:class:`hoomd.update.box_resize`
    does NOT reset the box to orthorhombic shape if this occurs (and does not move
    the next periodic image into the primary cell).

    Examples::

        update.box_resize(L = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]))
        box_resize = update.box_resize(L = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]), period = 10)
        update.box_resize(Lx = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]),
                          Ly = hoomd.variant.linear_interp([(0, 20), (1e6, 60)]),
                          Lz = hoomd.variant.linear_interp([(0, 10), (1e6, 80)]))
        update.box_resize(Lx = hoomd.variant.linear_interp([(0, 20), (1e6, 50)]), Ly = 10, Lz = 10)

        # Shear the box in the xy plane using Lees-Edwards boundary conditions
        update.box_resize(xy = hoomd.variant.linear_interp([(0,0), (1e6, 1)]))
    """

    def __init__(self, Lx = None, Ly = None, Lz = None, xy = None, xz = None, yz = None, period = 1, L = None, phase=0, scale_particles=True):

        # initialize base class
        _updater.__init__(self);

        self.metadata_fields = ['period']

        if L is not None:
            Lx = L;
            Ly = L;
            Lz = L;

        if Lx is None and Ly is None and Lz is None and xy is None and xz is None and yz is None:
            hoomd.context.current.device.cpp_msg.warning("update.box_resize: Ignoring request to setup updater without parameters\n")
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
        self.cpp_updater.setParams(scale_particles);

        if period is None:
            self.cpp_updater.update(hoomd.context.current.system.getCurrentTimeStep());
        else:
            self.setupUpdater(period, phase);


# Global current id counter to assign updaters unique names
_updater.cur_id = 0;
