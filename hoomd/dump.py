# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Write system configurations to files.

Commands in the dump package write the system state out to a file every
*period* time steps. Check the documentation for details on which file format
each command writes.
"""

from hoomd import _hoomd
import hoomd;
import sys;

## Writes simulation snapshots in the HOOMD XML format
#
# Every \a period time steps, a new file will be created. The state of the
# particles at that time step is written to the file in the HOOMD XML format.
# All values are written in native HOOMD-blue units, see \ref page_units for more information.
#
# \sa \ref page_xml_file_format
# \MPI_SUPPORTED
class xml(hoomd.analyze._analyzer):
    ## Initialize the hoomd_xml writer
    #
    # \param filename (optional) Base of the file name
    # \param period (optional) Number of time steps between file dumps
    # \param params (optional) Any number of parameters that set_params() accepts
    # \param time_step (optional) Time step to write into the file (overrides the current simulation step). time_step
    #                  is ignored for periodic updates
    # \param phase When -1, start on the current time step. When >= 0, execute on steps where (step + phase) % period == 0.
    # \param restart When True, write only \a filename and don't save previous states.
    #
    # \b Examples:
    # \code
    # dump.xml(filename="atoms.dump", period=1000)
    # xml = dump.xml(filename="particles", period=1e5)
    # xml = dump.xml(filename="test.xml", vis=True)
    # xml = dump.xml(filename="restart.xml", all=True, restart=True, period=10000, phase=0);
    # \endcode
    #
    # If period is set and restart is False, a new file will be created every \a period steps. The time step at which
    # the file is created is added to the file name in a fixed width format to allow files to easily
    # be read in order. I.e. the write at time step 0 with \c filename="particles" produces the file
    # \c particles.0000000000.xml
    #
    # If period is set and restart is True, dump.xml() will write a temporary file and then move it to \a filename. This
    # stores only the most recent state of the simulation in the written file. It is useful for writing jobs that
    # are restartable - see :ref:`restartable-jobs`.
    #
    # By default, only particle positions are output to the dump files. This can be changed
    # with set_params(), or by specifying the options in the dump.xml() command.
    #
    # If \a period is not specified, then no periodic updates will occur. Instead, the file
    # \a filename is written immediately. \a time_step is passed on to write()
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename="dump", period=None, time_step=None, phase=-1, restart=False, **params):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        # check restart options
        self.restart = restart;
        if restart and period is None:
            raise ValueError("a period must be specified with restart=True");

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.HOOMDDumpWriter(hoomd.context.current.system_definition, filename, restart);
        hoomd.util.quiet_status();
        self.set_params(**params);
        hoomd.util.unquiet_status();

        if period is not None:
            self.setupAnalyzer(period, phase);
            self.enabled = True;
            self.prev_period = 1;
        elif filename != "dump":
            hoomd.util.quiet_status();
            self.write(filename, time_step);
            hoomd.util.unquiet_status();
        else:
            self.enabled = False;

        # store metadata
        self.filename = filename
        self.period = period
        self.metadata_fields = ['filename','period']

    ## Change xml write parameters
    #
    # \param all (if True) Enables the output of all optional parameters below
    # \param vis (if True) Enables options commonly used for visualization.
    # - Specifically, vis=True sets position, mass, diameter, type, body, bond, angle, dihedral, improper, charge
    # \param position (if set) Set to True/False to enable/disable the output of particle positions in the xml file
    # \param image (if set) Set to True/False to enable/disable the output of particle images in the xml file
    # \param velocity (if set) Set to True/False to enable/disable the output of particle velocities in the xml file
    # \param mass (if set) Set to True/False to enable/disable the output of particle masses in the xml file
    # \param diameter (if set) Set to True/False to enable/disable the output of particle diameters in the xml file
    # \param type (if set) Set to True/False to enable/disable the output of particle types in the xml file
    # \param body (if set) Set to True/False to enable/disable the output of the particle bodies in the xml file
    # \param bond (if set) Set to True/False to enable/disable the output of bonds in the xml file
    # \param angle (if set) Set to True/False to enable/disable the output of angles in the xml file
    # \param dihedral (if set) Set to True/False to enable/disable the output of dihedrals in the xml file
    # \param improper (if set) Set to True/False to enable/disable the output of impropers in the xml file
    # \param constraint (if set) Set to True/False to enable/disable the output of constraints in the xml file
    # \param acceleration (if set) Set to True/False to enable/disable the output of particle accelerations in the xml
    # \param charge (if set) Set to True/False to enable/disable the output of particle charge in the xml
    # \param orientation (if set) Set to True/False to enable/disable the output of particle orientations in the xml file
    # \param angmom (if set) Set to True/False to enable/disable the output of particle angular momenta in the xml file
    # \param inertia (if set) Set to True/False to enable/disable the output of particle moments of inertia in the xml file
    # \param vizsigma (if set) Set to a floating point value to include as vizsigma in the xml file
    #
    # Using set_params() requires that the %dump was saved in a variable when it was specified.
    # \code
    # xml = dump.xml(filename="particles", period=1e5)
    # \endcode
    #
    # \b Examples:
    # \code
    # xml.set_params(type=False)
    # xml.set_params(position=False, type=False, velocity=True)
    # xml.set_params(type=True, position=True)
    # xml.set_params(bond=True)
    # xml.set_params(all=True)
    # \endcode
    def set_params(self,
                   all=None,
                   vis=None,
                   position=None,
                   image=None,
                   velocity=None,
                   mass=None,
                   diameter=None,
                   type=None,
                   body=None,
                   bond=None,
                   angle=None,
                   dihedral=None,
                   improper=None,
                   constraint=None,
                   acceleration=None,
                   charge=None,
                   orientation=None,
                   angmom=None,
                   inertia=None,
                   vizsigma=None):
        hoomd.util.print_status_line();
        self.check_initialization();

        if all:
            position = image = velocity = mass = diameter = type = bond = angle = dihedral = improper = constraint = True;
            acceleration = charge = body = orientation = angmom = inertia = True;

        if vis:
            position = mass = diameter = type = body = bond = angle = dihedral = improper = charge = True;

        if position is not None:
            self.cpp_analyzer.setOutputPosition(position);

        if image is not None:
            self.cpp_analyzer.setOutputImage(image);

        if velocity is not None:
            self.cpp_analyzer.setOutputVelocity(velocity);

        if mass is not None:
            self.cpp_analyzer.setOutputMass(mass);

        if diameter is not None:
            self.cpp_analyzer.setOutputDiameter(diameter);

        if type is not None:
            self.cpp_analyzer.setOutputType(type);

        if body is not None:
            self.cpp_analyzer.setOutputBody(body);

        if bond is not None:
            self.cpp_analyzer.setOutputBond(bond);

        if angle is not None:
            self.cpp_analyzer.setOutputAngle(angle);

        if dihedral is not None:
            self.cpp_analyzer.setOutputDihedral(dihedral);

        if improper is not None:
            self.cpp_analyzer.setOutputImproper(improper);

        if constraint is not None:
            self.cpp_analyzer.setOutputConstraint(constraint);

        if acceleration is not None:
            self.cpp_analyzer.setOutputAccel(acceleration);

        if charge is not None:
            self.cpp_analyzer.setOutputCharge(charge);

        if orientation is not None:
            self.cpp_analyzer.setOutputOrientation(orientation);

        if angmom is not None:
            self.cpp_analyzer.setOutputAngularMomentum(angmom);

        if inertia is not None:
            self.cpp_analyzer.setOutputMomentInertia(inertia);

        if vizsigma is not None:
            self.cpp_analyzer.setVizSigma(vizsigma);

    ## Write a file at the current time step
    #
    # \param filename File name to write to
    # \param time_step (if set) Time step value to write out to the file
    #
    # The periodic file writes can be temporarily overridden and a file with any file name
    # written at the current time step.
    #
    # \note When \a time_step is None, the current system time step is written to the file. When specified,
    #       \a time_step overrides this value.
    #
    # Executing write() requires that the %dump was saved in a variable when it was specified.
    # \code
    # xml = dump.xml()
    # \endcode
    #
    # \b Examples:
    # \code
    # xml.write(filename="start.xml")
    # xml.write(filename="start.xml", time_step=0)
    # \endcode
    def write(self, filename, time_step = None):
        hoomd.util.print_status_line();
        self.check_initialization();

        if time_step is None:
            time_step = hoomd.context.current.system.getCurrentTimeStep()

        self.cpp_analyzer.writeFile(filename, time_step);

    ## Write a restart file at the current time step
    #
    # This only works when dump.xml() is in **restart** mode. write_restart() writes out a restart file at the current
    # time step. Put it at the end of a script to ensure that the system state is written out before exiting.
    def write_restart(self):
        hoomd.util.print_status_line();

        if not self.restart:
            raise ValueError("Cannot write_restart() when restart=False");

        self.cpp_analyzer.analyze(hoomd.context.current.system.getCurrentTimeStep());

class dcd(hoomd.analyze._analyzer):
    R""" Writes simulation snapshots in the DCD format

    Args:
        filename (str): File name to write.
        period (int): Number of time steps between file dumps.
        group (:py:mod:`hoomd.group`): Particle group to output to the dcd file. If left as None, all particles will be written.
        overwrite (bool): When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
                          file *filename* will be overwritten.
        unwrap_full (bool): When False, (the default) particle coordinates are always written inside the simulation box.
                            When True, particles will be unwrapped into their current box image before writing to the dcd file.
        unwrap_rigid (bool): When False, (the default) individual particles are written inside the simulation box which
               breaks up rigid bodies near box boundaries. When True, particles belonging to the same rigid body will be
               unwrapped so that the body is continuous. The center of mass of the body remains in the simulation box, but
               some particles may be written just outside it. *unwrap_rigid* is ignored when *unwrap_full* is True.
        angle_z (bool): When True, the particle orientation angle is written to the z component (only useful for 2D simulations)
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.

    Every *period* time steps a new simulation snapshot is written to the
    specified file in the DCD file format. DCD only stores particle positions
    but is decently space efficient and fast to read and write.

    Particle positions are written directly in distance units - see :ref:`page-units`.

    Due to constraints of the DCD file format, once you stop writing to
    a file via :py:meth:`disable()`, you cannot continue writing to the same file,
    nor can you change the period of the dump at any time. Either of these tasks
    can be performed by creating a new dump file with the needed settings.

    Examples::

        dump.dcd(filename="trajectory.dcd", period=1000)
        dcd = dump.dcd(filename"data/dump.dcd", period=1000)

    Warning:
        When you use dump.dcd to append to an existing dcd file:

        * The period must be the same or the time data in the file will not be consistent.
        * dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
          consistent timeline
    """
    def __init__(self, filename, period, group=None, overwrite=False, unwrap_full=False, unwrap_rigid=False, angle_z=False, phase=-1):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        # create the c++ mirror class
        reported_period = period;
        try:
            reported_period = int(period);
        except TypeError:
            reported_period = 1;

        if group is None:
            hoomd.util.quiet_status();
            group = hoomd.group.all();
            hoomd.util.unquiet_status();

        self.cpp_analyzer = _hoomd.DCDDumpWriter(hoomd.context.current.system_definition, filename, int(reported_period), group.cpp_group, overwrite);
        self.cpp_analyzer.setUnwrapFull(unwrap_full);
        self.cpp_analyzer.setUnwrapRigid(unwrap_rigid);
        self.cpp_analyzer.setAngleZ(angle_z);
        self.setupAnalyzer(period, phase);

        # store metadata
        self.filename = filename
        self.period = period
        self.group = group
        self.metadata_fields = ['filename','period','group']

    def enable(self):
        hoomd.util.print_status_line();

        if self.enabled == False:
            hoomd.context.msg.error("you cannot re-enable DCD output after it has been disabled\n");
            raise RuntimeError('Error enabling updater');

    def set_period(self, period):
        hoomd.util.print_status_line();

        hoomd.context.msg.error("you cannot change the period of a dcd dump writer\n");
        raise RuntimeError('Error changing updater period');

class gsd(hoomd.analyze._analyzer):
    R""" Writes simulation snapshots in the GSD format

    Args:
        filename (str): File name to write
        period (int): Number of time steps between file dumps, or None to write a single file immediately.
        group (:py:mod:`hoomd.group`): Particle group to output to the gsd file.
        overwrite (bool): When False (the default), any existing GSD file will be appended to. When True, an existing DCD
                          file *filename* will be overwritten.
        truncate (bool): When False (the default), frames are appended to the GSD file. When True, truncate the file and
                         write a new frame 0 every time.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.
        time_step (int): Time step to write to the file (only used when period is None)
        static (list): A list of quantity categories that are static.

    Write a simulation snapshot to the specified GSD file at regular intervals.
    GSD is capable of storing all particle and bond data fields that hoomd stores,
    in every frame of the trajectory. This allows GSD to store simulations where the
    number of particles, number of particle types, particle types, diameter, mass,
    charge, or anything is changing over time.

    To save on space, GSD does not write values that are all set at defaults. So if
    all masses are left set at the default of 1.0, mass will not take up any space in
    the file. To save even more on space, flag fields as static (the default) and
    dump.gsd will only write them out to frame 0. When reading data from frame *i*,
    any data field not present will be read from frame 0. This makes every single frame
    of a GSD file fully specified and simulations initialized with init.read_gsd() can
    select any frame of the file.

    The **static** option applies to groups of fields:

    * ``attribute``

        * particles/N
        * particles/types
        * particles/typeid
        * particles/mass
        * particles/charge
        * particles/diameter
        * particles/body
        * particles/moment_inertia

    * ``property``

        * particles/position
        * particles/orientation

    * ``momentum``

        * particles/velocity
        * particles/angmom
        * particles/image

    * ``topology``

        * bonds/
        * angles/
        * dihedrals/
        * impropers/
        * constraints/

    See https://bitbucket.org/glotzer/gsd for more information on GSD files.

    If you only need to store a subset of the system, you can save file size and time spent analyzing data by
    specifying a group to write out. :py:class:`dump.gsd` will write out all of the particles in the group in ascending
    tag order. When the group is not :py:func:`group.all()`, :py:class:`dump.gsd` will not write the topology fields.

    To write restart files with gsd, set `truncate=True`. This will cause :py:class:`dump.gsd` to write a new frame 0
    to the file every period steps.

    dump.gsd writes static quantities from frame 0 only. Even if they change, it will not write them to subsequent
    frames. Quantity categories **not** listed in *static* are dynamic. :py:class:`dump.gsd` writes dynamic quantities to every frame.
    The default is only to write particle properties (position, orientation) on each frame, and hold all others fixed.
    In most simulations, attributes and topology do not vary - remove these from static if they do and you wish to
    save that information in a trajectory for later access. Particle momentum are always changing, but the default is
    to not include these quantities to save on file space.

    Examples::

        dump.gsd(filename="trajectory.gsd", period=1000, group=group.all(), phase=0)
        dump.gsd(filename="restart.gsd", truncate=True, period=10000, group=group.all(), phase=0)
        dump.gsd(filename="configuration.gsd", overwrite=True, period=None, group=group.all(), time_step=0)
        dump.gsd(filename="saveall.gsd", overwrite=True, period=1000, group=group.all(), static=[])

    """
    def __init__(self,
                 filename,
                 period,
                 group,
                 overwrite=False,
                 truncate=False,
                 phase=-1,
                 time_step=None,
                 static=['attribute', 'momentum', 'topology']):
        hoomd.util.print_status_line();

        for v in static:
            if v not in ['attribute', 'property', 'momentum', 'topology']:
                hoomd.context.msg.warning("dump.gsd: static quantity", v, "is not recognized");

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        self.cpp_analyzer = _hoomd.GSDDumpWriter(hoomd.context.current.system_definition, filename, group.cpp_group, overwrite, truncate);

        self.cpp_analyzer.setWriteAttribute('attribute' not in static);
        self.cpp_analyzer.setWriteProperty('property' not in static);
        self.cpp_analyzer.setWriteMomentum('momentum' not in static);
        self.cpp_analyzer.setWriteTopology('topology' not in static);

        if period is not None:
            self.setupAnalyzer(period, phase);
        else:
            if time_step is None:
                time_step = hoomd.context.current.system.getCurrentTimeStep()
            self.cpp_analyzer.analyze(time_step);

        # store metadata
        self.filename = filename
        self.period = period
        self.group = group
        self.phase = phase
        self.metadata_fields = ['filename','period','group', 'phase']

class pos(hoomd.analyze._analyzer):
    R""" Writes simulation snapshots in the POS format

    Args:
        filename (str): File name to write
        period (int): (optional) Number of time steps between file dumps
        unwrap_rigid (bool): When False, (the default) individual particles are written inside
                             the simulation box which breaks up rigid bodies near box boundaries. When True,
                             particles belonging to the same rigid body will be unwrapped so that the body
                             is continuous. The center of mass of the body remains in the simulation box, but
                             some particles may be written just outside it.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.
        addInfo (callable): A user-defined python function that returns a string of additional information when it is called. This
                            information will be printed in the pos file beneath the shape definitions. The information returned by addInfo
                            may dynamically change over the course of the simulation; addInfo is a function of the simulation timestep only.

    The file is opened on initialization and a new frame is appended every \a period steps.

    Warning:
        :py:class:`dump.pos` is not restart compatible. It always overwrites the file on initialization.

    Examples::

        dump.pos(filename="dump.pos", period=1000)
        pos = dump.pos(filename="particles.pos", period=1e5)
    """
    def __init__(self, filename, period=None, unwrap_rigid=False, phase=-1, addInfo=None):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        # create the c++ mirror class
        self.cpp_analyzer = _hoomd.POSDumpWriter(hoomd.context.current.system_definition, filename);
        self.cpp_analyzer.setUnwrapRigid(unwrap_rigid);

        if addInfo is not None:
            self.cpp_analyzer.setAddInfo(addInfo);

        if period is not None:
            self.setupAnalyzer(period, phase);
            self.enabled = True;
            self.prev_period = 1;
        else:
            self.enabled = False;

        # store metadata
        self.filename = filename
        self.period = period
        self.unwrap_rigid = unwrap_rigid
        self.metadata_fields = ['filename', 'period', 'unwrap_rigid']

    def set_def(self, typ, shape):
        v = hoomd.context.current.system_definition.getParticleData().getTypeByName(typ);
        self.cpp_analyzer.setDef(v, shape)

    def set_info(self, addInfo):
        if addInfo is not None:
            self.cpp_analyzer.setAddInfo(addInfo);
