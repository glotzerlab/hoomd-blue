# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Deprecated trajectory file writers.

"""

from hoomd.analyze import _analyzer;
from hoomd.deprecated import _deprecated;
import hoomd;

class xml(hoomd.analyze._analyzer):
    R""" Writes simulation snapshots in the HOOMD XML format.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to dump
        filename (str): (optional) Base of the file name
        period (int): (optional) Number of time steps between file dumps
        params: (optional) Any number of parameters that set_params() accepts
        time_step (int): (optional) Time step to write into the file (overrides the current simulation step). time_step
                     is ignored for periodic updates
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where &(step + phase) % period == 0*.
        restart (bool): When True, write only *filename* and don't save previous states.

    .. deprecated:: 2.0
       GSD is the new default file format for HOOMD-blue. It can store everything that an XML file can in
       an efficient binary format that is easy to access. See :py:class:`hoomd.dump.gsd`.

    Every *period* time steps, a new file will be created. The state of the
    particles in *group* at that time step is written to the file in the HOOMD XML format.
    All values are written in native HOOMD-blue units, see :ref:`page-units` for more information.

    If you only need to store a subset of the system, you can save file size and time spent analyzing data by
    specifying a group to write out. :py:class:`xml` will write out all of the particles in *group* in ascending
    tag order. When the group is not :py:func:`hoomd.group.all()`, :py:class:`xml` will not write the topology fields
    (bond, angle, dihedral, improper, constraint).

    Examples::

        deprecated.dump.xml(group=group.all(), filename="atoms.dump", period=1000)
        xml = deprecated.dump.xml(group=group.all(), filename="particles", period=1e5)
        xml = deprecated.dump.xml(group=group.all(), filename="test.xml", vis=True)
        xml = deprecated.dump.xml(group=group.all(), filename="restart.xml", all=True, restart=True, period=10000, phase=0);
        xml = deprecated.dump.xml(group=group.type('A'), filename="A", period=1e3)


    If period is set and restart is False, a new file will be created every *period* steps. The time step at which
    the file is created is added to the file name in a fixed width format to allow files to easily
    be read in order. I.e. the write at time step 0 with ``filename="particles"`` produces the file
    ``particles.0000000000.xml``.

    If period is set and restart is True, :py:class:`xml` will write a temporary file and then move it to *filename*. This
    stores only the most recent state of the simulation in the written file. It is useful for writing jobs that
    are restartable - see :ref:`restartable-jobs`. Note that this causes high metadata traffic on lustre filesystems
    and may cause your account to be blocked at some supercomputer centers. Use :py:class:`hoomd.dump.gsd` for efficient
    restart files.

    By default, only particle positions are output to the dump files. This can be changed
    with set_params(), or by specifying the options in the :py:class:`xml` command.

    If *period* is not specified, then no periodic updates will occur. Instead, the file
    *filename* is written immediately. *time_step* is passed on to write()

    """
    def __init__(self, group, filename="dump", period=None, time_step=None, phase=0, restart=False, **params):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        # check restart options
        self.restart = restart;
        if restart and period is None:
            raise ValueError("a period must be specified with restart=True");

        # create the c++ mirror class
        self.group = group
        self.cpp_analyzer = _deprecated.HOOMDDumpWriter(hoomd.context.current.system_definition, filename, self.group.cpp_group, restart);
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
        self.metadata_fields = ['group','filename','period']

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
        R""" Change xml write parameters.

        Args:
            all (bool): (if True) Enables the output of all optional parameters below
            vis (bool): (if True) Enables options commonly used for visualization.
              - Specifically, vis=True sets position, mass, diameter, type, body, bond, angle, dihedral, improper, charge
            position (bool): (if set) Set to True/False to enable/disable the output of particle positions in the xml file
            image (bool): (if set) Set to True/False to enable/disable the output of particle images in the xml file
            velocity (bool): (if set) Set to True/False to enable/disable the output of particle velocities in the xml file
            mass (bool): (if set) Set to True/False to enable/disable the output of particle masses in the xml file
            diameter (bool): (if set) Set to True/False to enable/disable the output of particle diameters in the xml file
            type (bool): (if set) Set to True/False to enable/disable the output of particle types in the xml file
            body (bool): (if set) Set to True/False to enable/disable the output of the particle bodies in the xml file
            bond (bool): (if set) Set to True/False to enable/disable the output of bonds in the xml file
            angle (bool): (if set) Set to True/False to enable/disable the output of angles in the xml file
            dihedral (bool): (if set) Set to True/False to enable/disable the output of dihedrals in the xml file
            improper (bool): (if set) Set to True/False to enable/disable the output of impropers in the xml file
            constraint (bool): (if set) Set to True/False to enable/disable the output of constraints in the xml file
            acceleration (bool): (if set) Set to True/False to enable/disable the output of particle accelerations in the xml
            charge (bool): (if set) Set to True/False to enable/disable the output of particle charge in the xml
            orientation (bool): (if set) Set to True/False to enable/disable the output of particle orientations in the xml file
            angmom (bool): (if set) Set to True/False to enable/disable the output of particle angular momenta in the xml file
            inertia (bool): (if set) Set to True/False to enable/disable the output of particle moments of inertia in the xml file
            vizsigma (bool): (if set) Set to a floating point value to include as vizsigma in the xml file

        Examples::

            xml.set_params(type=False)
            xml.set_params(position=False, type=False, velocity=True)
            xml.set_params(type=True, position=True)
            xml.set_params(bond=True)
            xml.set_params(all=True)

        .. attention::
            The simulation topology (bond, angle, dihedral, improper, constraint) cannot be output when
            the group for :py:class:`xml` is not :py:class:`hoomd.group.all`. An error will
            be raised.
        """

        hoomd.util.print_status_line();
        self.check_initialization();

        if all:
            position = image = velocity = mass = diameter = type = bond = angle = dihedral = improper = constraint = True;
            acceleration = charge = body = orientation = angmom = inertia = True;

        if vis:
            position = mass = diameter = type = body = bond = angle = dihedral = improper = charge = True;

        # validate that
        if bond or angle or dihedral or improper or constraint:
            hoomd.util.quiet_status()
            group_all = hoomd.group.all()
            hoomd.util.unquiet_status()
            if self.group != group_all:
                hoomd.context.msg.error("Cannot output topology when not all particles are dumped!\n")
                raise ValueError("Cannot output topology when not all particles are dumped!")

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

    def write(self, filename, time_step = None):
        R""" Write a file at the current time step.

        Args:
            filename (str): File name to write to
            time_step (int): (if set) Time step value to write out to the file

        The periodic file writes can be temporarily overridden and a file with any file name
        written at the current time step.

        When *time_step* is None, the current system time step is written to the file. When specified,
        *time_step* overrides this value.

        Examples::

            xml.write(filename="start.xml")
            xml.write(filename="start.xml", time_step=0)

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if time_step is None:
            time_step = hoomd.context.current.system.getCurrentTimeStep()

        self.cpp_analyzer.writeFile(filename, time_step);

    def write_restart(self):
        R""" Write a restart file at the current time step.

        This only works when dump.xml() is in **restart** mode. write_restart() writes out a restart file at the current
        time step. Put it at the end of a script to ensure that the system state is written out before exiting.
        """
        hoomd.util.print_status_line();

        if not self.restart:
            raise ValueError("Cannot write_restart() when restart=False");

        self.cpp_analyzer.analyze(hoomd.context.current.system.getCurrentTimeStep());

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
        addInfo (`callable`): A user-defined python function that returns a string of additional information when it is called. This
                            information will be printed in the pos file beneath the shape definitions. The information returned by addInfo
                            may dynamically change over the course of the simulation; addInfo is a function of the simulation timestep only.

    .. deprecated:: 2.0

    The file is opened on initialization and a new frame is appended every \a period steps.

    Warning:
        :py:class:`pos` is not restart compatible. It always overwrites the file on initialization.

    Examples::

        dump.pos(filename="dump.pos", period=1000)
        pos = dump.pos(filename="particles.pos", period=1e5)

    """
    def __init__(self, filename, period=None, unwrap_rigid=False, phase=0, addInfo=None):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        # create the c++ mirror class
        self.cpp_analyzer = _deprecated.POSDumpWriter(hoomd.context.current.system_definition, filename);
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
        R""" Set a pos def string for a given type

        Args:
            typ (str): Type name to set shape def
            shape (str): Shape def string to set

        """
        v = hoomd.context.current.system_definition.getParticleData().getTypeByName(typ);
        self.cpp_analyzer.setDef(v, shape)

    def set_info(self, addInfo):
        if addInfo is not None:
            self.cpp_analyzer.setAddInfo(addInfo);
