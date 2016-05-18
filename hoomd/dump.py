# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Write system configurations to files.

Commands in the dump package write the system state out to a file every
*period* time steps. Check the documentation for details on which file format
each command writes.
"""

from hoomd import _hoomd
import hoomd;
import sys;

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
    specified file in the DCD file format. DCD only stores particle positions, in distance
    units - see :ref:`page-units`.

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

    See https://bitbucket.org/glotzer/gsd and http://gsd.readthedocs.io/ for more information on GSD files.

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
