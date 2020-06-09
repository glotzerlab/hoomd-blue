# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Write system configurations to files.

Commands in the dump package write the system state out to a file every
*period* time steps. Check the documentation for details on which file format
each command writes.
"""

from collections import namedtuple;
from hoomd import _hoomd
import hoomd;
import json;
import os;
import sys;
import types;

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
    def __init__(self, filename, period, group=None, overwrite=False, unwrap_full=False, unwrap_rigid=False, angle_z=False, phase=0):
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
        """ The DCD dump writer cannot be re-enabled """
        hoomd.util.print_status_line();

        if self.enabled == False:
            hoomd.context.msg.error("you cannot re-enable DCD output after it has been disabled\n");
            raise RuntimeError('Error enabling updater');

    def set_period(self, period):
        hoomd.util.print_status_line();

        hoomd.context.msg.error("you cannot change the period of a dcd dump writer\n");
        raise RuntimeError('Error changing updater period');

class getar(hoomd.analyze._analyzer):
    """Analyzer for dumping system properties to a getar file at intervals.

    Getar files are a simple interface on top of archive formats (such
    as zip and tar) for storing trajectory data efficiently. A more
    thorough description of the format and a description of a python
    API to read and write these files is available at `the libgetar
    documentation <http://libgetar.readthedocs.io>`_.

    Properties to dump can be given either as a
    :py:class:`getar.DumpProp` object or a name. Supported property
    names are specified in the Supported Property Table in
    :py:class:`hoomd.init.read_getar`.

    Files can be opened in write, append, or one-shot mode. Write mode
    overwrites files with the same name, while append mode adds to
    them. One-shot mode is intended for restorable system backups and
    is described below.

    **One-shot mode**

    In one-shot mode, activated by passing mode='1' to the getar
    constructor, properties are written to a temporary file, which
    then overwrites the file with the given filename. In this way, the
    file with the given filename should always have the most recent
    frame of successfully written data. This mode is designed for
    being able to dump restoration data often without wasting large
    amounts of space saving earlier data. Note that this
    create-and-overwrite process can be stressful on filesystems,
    particularly lustre filesystems, and can get your account blocked
    on some supercomputer resources if overused.

    For convenience, you can also specify **composite properties**,
    which are expanded according to the table below.

    .. tabularcolumns:: |p{0.25 \textwidth}|p{0.75 \textwidth}|
    .. csv-table::
       :header: "Name", "Result"
       :widths: 1, 3

       "global_all", "box, dimensions"
       "angle_all", "angle_type_names, angle_tag, angle_type"
       "bond_all", "bond_type_names, bond_tag, bond_type"
       "dihedral_all", "dihedral_type_names, dihedral_tag, dihedral_type"
       "improper_all", "improper_type_names, improper_tag, improper_type"
       "particle_all", "angular_momentum, body, charge, diameter, image, mass, moment_inertia, orientation, position, type, type_names, velocity"
       "all", "particle_all, angle_all, bond_all, dihedral_all, improper_all, global_all"
       "viz_static", "type, type_names, dimensions"
       "viz_dynamic", "position, box"
       "viz_all", "viz_static, viz_dynamic"
       "viz_aniso_dynamic", "viz_dynamic, orientation"
       "viz_aniso_all", "viz_static, viz_aniso_dynamic"

    **Particle-related metadata**

    Metadata about particle shape (for later visualization or use in
    restartable scripts) can be stored in a simple form through
    :py:func:`hoomd.dump.getar.writeJSON`, which encodes JSON records
    as strings and stores them inside the dump file. Currently,
    classes inside :py:mod:`hoomd.dem` and :py:mod:`hoomd.hpmc` are
    equipped with `get_type_shapes()` methods which can provide
    per-particle-type shape information as a list.

    Example::

        dump = hoomd.dump.getar.simple('dump.sqlite', 1e3,
            static=['viz_static'],
            dynamic=['viz_aniso_dynamic'])

        dem_wca = hoomd.dem.WCA(nlist, radius=0.5)
        dem_wca.setParams('A', vertices=vertices, faces=faces)
        dump.writeJSON('type_shapes.json', dem_wca.get_type_shapes())

        mc = hpmc.integrate.convex_polygon(seed=415236)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
        dump.writeJSON('type_shapes.json', mc.get_type_shapes(), dynamic=True)

    """

    class DumpProp(namedtuple('DumpProp', ['name', 'highPrecision', 'compression'])):
        """Simple, internal, read-only namedtuple wrapper for specifying how
        getar properties will be dumped"""

        def __new__(self, name, highPrecision=False,
                     compression=_hoomd.GetarCompression.FastCompress):
            """Initialize a property dump description tuple.

            :param name: property name (string; see `Supported Property Table`_)
            :param highPrecision: if True, try to save this data in high-precision
            :param compression: one of `hoomd.dump.getar.Compression.{NoCompress, FastCompress, MediumCompress, SlowCompress`}.
            """
            return super(getar.DumpProp, self).__new__(
                self, name=name, highPrecision=highPrecision,
                compression=compression);

    Compression = _hoomd.GetarCompression;

    dump_modes = {'w': _hoomd.GetarDumpMode.Overwrite,
                  'a': _hoomd.GetarDumpMode.Append,
                  '1': _hoomd.GetarDumpMode.OneShot};

    substitutions = {
        'all': ['particle_all', 'angle_all', 'bond_all',
                'dihedral_all', 'improper_all', 'global_all'],
        'particle_all':
            ['angular_momentum', 'body', 'charge', 'diameter', 'image', 'mass', 'moment_inertia',
             'orientation', 'position', 'type', 'type_names', 'velocity'],
        'angle_all': ['angle_type_names', 'angle_tag', 'angle_type'],
        'bond_all': ['bond_type_names', 'bond_tag', 'bond_type'],
        'dihedral_all': ['dihedral_type_names', 'dihedral_tag', 'dihedral_type'],
        'improper_all': ['improper_type_names', 'improper_tag', 'improper_type'],
        'global_all': ['box', 'dimensions'],
        'viz_dynamic': ['position', 'box'],
        'viz_static': ['type', 'type_names', 'dimensions'],
        'viz_all': ['viz_static', 'viz_dynamic'],
        'viz_aniso_dynamic': ['viz_dynamic', 'orientation'],
        'viz_aniso_all': ['viz_static', 'viz_aniso_dynamic']};

    # List of properties we know how to dump and their enums
    known_properties = {'angle_type_names': _hoomd.GetarProperty.AngleNames,
                        'angle_tag': _hoomd.GetarProperty.AngleTags,
                        'angle_type': _hoomd.GetarProperty.AngleTypes,
                        'angular_momentum': _hoomd.GetarProperty.AngularMomentum,
                        'body': _hoomd.GetarProperty.Body,
                        'bond_type_names': _hoomd.GetarProperty.BondNames,
                        'bond_tag': _hoomd.GetarProperty.BondTags,
                        'bond_type': _hoomd.GetarProperty.BondTypes,
                        'box': _hoomd.GetarProperty.Box,
                        'charge': _hoomd.GetarProperty.Charge,
                        'diameter': _hoomd.GetarProperty.Diameter,
                        'dihedral_type_names': _hoomd.GetarProperty.DihedralNames,
                        'dihedral_tag': _hoomd.GetarProperty.DihedralTags,
                        'dihedral_type': _hoomd.GetarProperty.DihedralTypes,
                        'dimensions': _hoomd.GetarProperty.Dimensions,
                        'image': _hoomd.GetarProperty.Image,
                        'improper_type_names': _hoomd.GetarProperty.ImproperNames,
                        'improper_tag': _hoomd.GetarProperty.ImproperTags,
                        'improper_type': _hoomd.GetarProperty.ImproperTypes,
                        'mass': _hoomd.GetarProperty.Mass,
                        'moment_inertia': _hoomd.GetarProperty.MomentInertia,
                        'orientation': _hoomd.GetarProperty.Orientation,
                        'position': _hoomd.GetarProperty.Position,
                        'potential_energy': _hoomd.GetarProperty.PotentialEnergy,
                        'type': _hoomd.GetarProperty.Type,
                        'type_names': _hoomd.GetarProperty.TypeNames,
                        'velocity': _hoomd.GetarProperty.Velocity,
                        'virial': _hoomd.GetarProperty.Virial};

    # List of properties we know how to dump and their enums
    known_resolutions = {'angle_type_names': _hoomd.GetarResolution.Text,
                         'angle_tag': _hoomd.GetarResolution.Individual,
                         'angle_type': _hoomd.GetarResolution.Individual,
                         'angular_momentum': _hoomd.GetarResolution.Individual,
                         'body': _hoomd.GetarResolution.Individual,
                         'bond_type_names': _hoomd.GetarResolution.Text,
                         'bond_tag': _hoomd.GetarResolution.Individual,
                         'bond_type': _hoomd.GetarResolution.Individual,
                         'box': _hoomd.GetarResolution.Uniform,
                         'charge': _hoomd.GetarResolution.Individual,
                         'diameter': _hoomd.GetarResolution.Individual,
                         'dihedral_type_names': _hoomd.GetarResolution.Text,
                         'dihedral_tag': _hoomd.GetarResolution.Individual,
                         'dihedral_type': _hoomd.GetarResolution.Individual,
                         'dimensions': _hoomd.GetarResolution.Uniform,
                         'image': _hoomd.GetarResolution.Individual,
                         'improper_type_names': _hoomd.GetarResolution.Text,
                         'improper_tag': _hoomd.GetarResolution.Individual,
                         'improper_type': _hoomd.GetarResolution.Individual,
                         'mass': _hoomd.GetarResolution.Individual,
                         'moment_inertia': _hoomd.GetarResolution.Individual,
                         'orientation': _hoomd.GetarResolution.Individual,
                         'position': _hoomd.GetarResolution.Individual,
                         'potential_energy': _hoomd.GetarResolution.Individual,
                         'type': _hoomd.GetarResolution.Individual,
                         'type_names': _hoomd.GetarResolution.Text,
                         'velocity': _hoomd.GetarResolution.Individual,
                         'virial': _hoomd.GetarResolution.Individual};

    # List of properties which can't run in MPI mode
    bad_mpi_properties = ['potential_energy', 'virial'];

    def _getStatic(self, val):
        """Helper method to parse a static property specification element"""
        if type(val) == type(''):
            return self.DumpProp(name=val);
        else:
            return val;

    def _expandNames(self, vals):
        result = [];
        for val in vals:
            val = self._getStatic(val);
            if val.name in self.substitutions:
                subs = [self.DumpProp(name, val.highPrecision, val.compression) for name in
                        self.substitutions[val.name]];
                result.extend(self._expandNames(subs));
            else:
                result.append(val);

        return result;

    def __init__(self, filename, mode='w', static=[], dynamic={}, _register=True):
        """Initialize a getar dumper. Creates or appends an archive at the given file
        location according to the mode and prepares to dump the given
        sets of properties.

        Args:
            filename (str): Name of the file to open
            mode (str): Run mode; see mode list below.
            static (list): List of static properties to dump immediately
            dynamic (dict): Dictionary of {prop: period} periodic dumps
            _register (bool): If True, register as a hoomd analyzer (internal)

        Note that zip32-format archives can not be appended to at the
        moment; for details and solutions, see the libgetar
        documentation, section "Zip vs. Zip64." The gtar.fix module was
        explicitly made for this purpose, but be careful not to call it
        from within a running GPU HOOMD simulation due to strangeness in
        the CUDA driver.

        Valid mode arguments:

        * 'w': Write, and overwrite if file exists
        * 'a': Write, and append if file exists
        * '1': One-shot mode: keep only one frame of data. For details on one-shot mode, see the "One-shot mode" section of :py:class:`getar`.

        Property specifications can be either a property name (as a string) or
        :py:class:`DumpProp` objects if you desire greater control over how the
        property will be dumped.

        Example::

            # detailed API; see `dump.getar.simple` for simpler wrappers
            zip = dump.getar('dump.zip', static=['types'],
                      dynamic={'orientation': 10000,
                               'velocity': 5000,
                               dump.getar.DumpProp('position', highPrecision=True): 10000})

        """

        self._static = self._expandNames(static);
        self._dynamic = {};

        for key in dynamic:
            period = dynamic[key];
            for prop in self._expandNames([key]):
                self._dynamic[prop] = period;

        if _register:
            hoomd.analyze._analyzer.__init__(self);
            self.analyzer_name = "dump.getar%d" % (hoomd.analyze._analyzer.cur_id - 1);

        for val in self._static:
            if prop.name not in self.known_properties:
                raise RuntimeError('Unknown static property in dump.getar: {}'.format(val));

        for val in self._dynamic:
            if val.name not in self.known_properties:
                raise RuntimeError('Unknown dynamic property in dump.getar: {}'.format(val));

        try:
            dumpMode = self.dump_modes[mode];
        except KeyError:
            raise RuntimeError('Unknown open mode: {}'.format(mode));

        if dumpMode == self.dump_modes['a'] and not os.path.isfile(filename):
            dumpMode = self.dump_modes['w'];

        self.cpp_analyzer = _hoomd.GetarDumpWriter(hoomd.context.current.system_definition,
                                                filename, dumpMode,
                                                hoomd.context.current.system.getCurrentTimeStep());

        for val in set(self._static):
            prop = self._getStatic(val);
            if hoomd.comm.get_num_ranks() > 1 and prop.name in self.bad_mpi_properties:
                raise RuntimeError(('dump.getar: Can\'t dump property {} '
                                    'with MPI!').format(prop.name));
            else:
                self.cpp_analyzer.setPeriod(self.known_properties[prop.name],
                                            self.known_resolutions[prop.name],
                                            _hoomd.GetarBehavior.Constant,
                                            prop.highPrecision, prop.compression, 0);

        for prop in self._dynamic:
            try:
                if hoomd.comm.get_num_ranks() > 1 and prop.name in self.bad_mpi_properties:
                    raise RuntimeError(('dump.getar: Can\'t dump property {} '
                                        'with MPI!').format(prop.name));
                else:
                    for period in self._dynamic[prop]:
                        self.cpp_analyzer.setPeriod(self.known_properties[prop.name],
                                                    self.known_resolutions[prop.name],
                                                    _hoomd.GetarBehavior.Discrete,
                                                    prop.highPrecision, prop.compression,
                                                    int(period));
            except TypeError: # We got a single value, not an iterable
                if hoomd.comm.get_num_ranks() > 1 and prop.name in self.bad_mpi_properties:
                    raise RuntimeError(('dump.getar: Can\'t dump property {} '
                                        'with MPI!').format(prop.name));
                else:
                    self.cpp_analyzer.setPeriod(self.known_properties[prop.name],
                                                self.known_resolutions[prop.name],
                                                _hoomd.GetarBehavior.Discrete,
                                                prop.highPrecision, prop.compression,
                                                int(self._dynamic[prop]));

        if _register:
            self.setupAnalyzer(int(self.cpp_analyzer.getPeriod()));

    def writeJSON(self, name, contents, dynamic=True):
        """Encodes the given JSON-encodable object as a string and writes it
        (immediately) as a quantity with the given name. If dynamic is
        True, writes the record as a dynamic record with the current
        timestep.

        Args:
            name (str): Name of the record to save
            contents (str): Any datatype encodable by the :py:mod:`json` module
            dynamic (bool): If True, dump a dynamic quantity with the current timestep; otherwise, dump a static quantity

        Example::

            dump = hoomd.dump.getar.simple('dump.sqlite', 1e3,
                static=['viz_static'], dynamic=['viz_dynamic'])
            dump.writeJSON('params.json', dict(temperature=temperature, pressure=pressure))
            dump.writeJSON('metadata.json', hoomd.meta.dump_metadata())
        """
        if dynamic:
            timestep = hoomd.context.current.system.getCurrentTimeStep()
        else:
            timestep = -1

        self.cpp_analyzer.writeStr(name, json.dumps(contents), timestep)

    @classmethod
    def simple(cls, filename, period, mode='w', static=[], dynamic=[], high_precision=False):
        """Create a :py:class:`getar` dump object with a simpler interface.

        Static properties will be dumped once immediately, and dynamic
        properties will be dumped every `period` steps. For detailed
        explanation of arguments, see :py:class:`getar`.

        Args:
            filename (str): Name of the file to open
            period (int): Period to dump the given dynamic properties with
            mode (str): Run mode; see mode list in :py:class:`getar`.
            static (list): List of static properties to dump immediately
            dynamic (list): List of properties to dump every `period` steps
            high_precision (bool): If True, dump precision properties

        Example::

            # [optionally] dump metadata beforehand with libgetar
            with gtar.GTAR('dump.sqlite', 'w') as trajectory:
                metadata = json.dumps(hoomd.meta.dump_metadata())
                trajectory.writeStr('hoomd_metadata.json', metadata)
            # for later visualization of anisotropic systems
            zip2 = hoomd.dump.getar.simple(
                 'dump.sqlite', 100000, 'a', static=['viz_static'], dynamic=['viz_aniso_dynamic'])
            # as backup to restore from later
            backup = hoomd.dump.getar.simple(
                'backup.tar', 10000, '1', static=['viz_static'], dynamic=['viz_aniso_dynamic'])

        """
        dynamicDict = {cls.DumpProp(name, highPrecision=high_precision): period for name in dynamic};
        return cls(filename=filename, mode=mode, static=static, dynamic=dynamicDict);

    @classmethod
    def immediate(cls, filename, static, dynamic):
        """Immediately dump the given static and dynamic properties to the given filename.

        For detailed explanation of arguments, see :py:class:`getar`.

        Example::

            hoomd.dump.getar.immediate(
                'snapshot.tar', static=['viz_static'], dynamic=['viz_dynamic'])

        """
        hoomd.util.quiet_status();
        dumper = getar(filename, 'w', static, {key: 1 for key in dynamic}, _register=False);
        dumper.cpp_analyzer.analyze(hoomd.context.current.system.getCurrentTimeStep());
        dumper.close();
        del dumper.cpp_analyzer;
        hoomd.util.unquiet_status();

    def close(self):
        """Closes the trajectory if it is open. Finalizes any IO beforehand."""
        self.cpp_analyzer.close();

class gsd(hoomd.analyze._analyzer):
    R""" Writes simulation snapshots in the GSD format

    Args:
        filename (str): File name to write
        period (int): Number of time steps between file dumps, or None to write a single file immediately.
        group (:py:mod:`hoomd.group`): Particle group to output to the gsd file.
        overwrite (bool): When False (the default), any existing GSD file will be appended to. When True, an existing GSD
                          file *filename* will be overwritten.
        truncate (bool): When False (the default), frames are appended to the GSD file. When True, truncate the file and
                         write a new frame 0 every time.
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.
        time_step (int): Time step to write to the file (only used when period is None)
        dynamic (list): A list of quantity categories to save every frame. (added in version 2.2)
        static (list): A list of quantity categories save only in frame 0 (may not be set in conjunction with *dynamic*, deprecated in version 2.2).

    Write a simulation snapshot to the specified GSD file at regular intervals. GSD is capable of storing all particle
    and bond data fields in hoomd, in every frame of the trajectory. This allows GSD to store simulations where the
    number of particles, number of particle types, particle types, diameter, mass, charge, and other quantities change
    over time. GSD can also store integrator state information necessary for restarting simulations and user-defined log
    quantities.

    To save on space, GSD does not write values that are all set at defaults. So if all masses, orientations, angular
    momenta, etc... are left default, these fields  will not take up any space in the file. Additionally, only
    **dynamic** quantities are written to all frames, non-dynamic quantities are only written to frame 0. The GSD schema
    defines that data not present in frame *i* is to be read from frame 0. This makes every single frame of a GSD file
    fully specified and simulations initialized with :py:func:`hoomd.init.read_gsd()` can select any frame of the file.

    You can control what quantities are dynamic by category. ``property`` is always dynamic. The categories listed in
    the **dynamic** will also be written out to every frame, but only if they have changed from the defaults.

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
        * particles/orientation (*only written when changed from default*)

    * ``momentum``

        * particles/velocity
        * particles/angmom (*only written when changed from default*)
        * particles/image

    * ``topology``

        * bonds/
        * angles/
        * dihedrals/
        * impropers/
        * constraints/
        * pairs/

    See https://github.com/glotzerlab/gsd and http://gsd.readthedocs.io/ for more information on GSD files.

    If you only need to store a subset of the system, you can save file size and time spent analyzing data by
    specifying a group to write out. :py:class:`gsd` will write out all of the particles in the group in ascending
    tag order. When the group is not :py:func:`hoomd.group.all()`, :py:class:`gsd` will not write the topology fields.

    To write restart files with gsd, set `truncate=True`. This will cause :py:class:`gsd` to write a new frame 0
    to the file every period steps.

    .. rubric:: State data

    :py:class:`gsd` can save internal state data for the following hoomd objects:

        * :py:class:`HPMC integrators <hoomd.hpmc.integrate.mode_hpmc>`

    Call :py:meth:`dump_state` with the object as an argument to enable saving its state. State saved in this way
    can be restored after initializing the system with :py:meth:`hoomd.init.read_gsd`.

    .. rubric:: User-defined log quantities

    Associate a name with a callable python object that returns a numpy array in :py:attr:`log`, and :py:class:`gsd`
    will save the data you provide on every frame. Prefix per-particle quantities with ``particles/`` and per-bond
    quantities with ``bonds/`` so that visualization tools such as `OVITO <https://www.ovito.org/>`_ will make them
    available in their pipelines. OVITO also understand scalar values (length 1 numpy arrays) and strings encoded
    as uint8 numpy arrays.

    Examples::

        dump.gsd(filename="trajectory.gsd", period=1000, group=group.all(), phase=0)
        dump.gsd(filename="restart.gsd", truncate=True, period=10000, group=group.all(), phase=0)
        dump.gsd(filename="configuration.gsd", overwrite=True, period=None, group=group.all(), time_step=0)
        dump.gsd(filename="momentum_too.gsd", period=1000, group=group.all(), phase=0, dynamic=['momentum'])
        dump.gsd(filename="saveall.gsd", overwrite=True, period=1000, group=group.all(), dynamic=['attribute', 'momentum', 'topology'])

    """
    def __init__(self,
                 filename,
                 period,
                 group,
                 overwrite=False,
                 truncate=False,
                 phase=0,
                 time_step=None,
                 static=None,
                 dynamic=None):
        hoomd.util.print_status_line();

        if static is not None and dynamic is not None:
            raise ValueError("Cannot specify both static and dynamic arguments");

        categories = ['attribute', 'property', 'momentum', 'topology'];
        dynamic_quantities = ['property']

        # process inputs and build list of dynamic quantities
        if static is not None:
            hoomd.context.msg.warning("The static argument to hoomd.dump.gsd is deprecated, use dynamic instead.\n");

            # notify they user of possible typos
            for v in static:
                if v not in categories:
                    hoomd.context.msg.warning("dump.gsd: static quantity " + v + " is not recognized\n");

            # invert the sense of the static arg
            dynamic_quantities = []
            for v in categories:
                if v not in static:
                    dynamic_quantities.append(v);

        if dynamic is not None:
            for v in dynamic:
                if v not in categories:
                    hoomd.context.msg.warning("dump.gsd: dynamic quantity " + v + " is not recognized\n");

            dynamic_quantities = ['property'] + dynamic;

        # initialize base class
        hoomd.analyze._analyzer.__init__(self);

        self.cpp_analyzer = _hoomd.GSDDumpWriter(hoomd.context.current.system_definition, filename, group.cpp_group, overwrite, truncate);

        self.cpp_analyzer.setWriteAttribute('attribute' in dynamic_quantities);
        self.cpp_analyzer.setWriteProperty('property' in dynamic_quantities);
        self.cpp_analyzer.setWriteMomentum('momentum' in dynamic_quantities);
        self.cpp_analyzer.setWriteTopology('topology' in dynamic_quantities);

        if period is not None:
            self.setupAnalyzer(period, phase);
        else:
            if time_step is None:
                time_step = hoomd.context.current.system.getCurrentTimeStep()
            self.cpp_analyzer.analyze(time_step);
            hoomd.context.current.analyzers.remove(self)

        # store metadata
        self.filename = filename
        self.period = period
        self.group = group
        self.phase = phase
        self.metadata_fields = ['filename','period','group', 'phase']

    def write_restart(self):
        """ Write a restart file at the current time step.

        Call :py:meth:`write_restart` at the end of a simulation where are writing a gsd restart file with
        ``truncate=True`` to ensure that you have the final frame of the simulation written before exiting.
        See :ref:`restartable-jobs` for examples.
        """

        time_step = hoomd.context.current.system.getCurrentTimeStep()
        self.cpp_analyzer.analyze(time_step);

    def dump_state(self, obj):
        """Write state information for a hoomd object.

        Call :py:meth:`dump_state` if you want to write the state of a hoomd object
        to the gsd file.

        .. versionadded:: 2.2
        """
        if hasattr(obj, '_connect_gsd') and type(getattr(obj, '_connect_gsd')) == types.MethodType:
            obj._connect_gsd(self);
        else:
            hoomd.context.msg.warning("GSD is not currently support for {name}".format(obj.__class__.__name__));

    def dump_shape(self, obj):
        """Writes particle shape information stored by a hoomd object.

        This method writes particle shape information into a GSD file, in the
        chunk :code:`particle/type_shapes`. This information can be used by
        other libraries for visualization. The following classes support
        writing shape information to GSD files:

        * :py:class:`hoomd.hpmc.integrate.sphere`
        * :py:class:`hoomd.hpmc.integrate.convex_polyhedron`
        * :py:class:`hoomd.hpmc.integrate.convex_spheropolyhedron`
        * :py:class:`hoomd.hpmc.integrate.polyhedron`
        * :py:class:`hoomd.hpmc.integrate.convex_polygon`
        * :py:class:`hoomd.hpmc.integrate.convex_spheropolygon`
        * :py:class:`hoomd.hpmc.integrate.simple_polygon`
        * :py:class:`hoomd.hpmc.integrate.ellipsoid`
        * :py:class:`hoomd.dem.pair.WCA`
        * :py:class:`hoomd.dem.pair.SWCA`
        * :py:class:`hoomd.md.pair.gb`

        See the `Shape Visualization Specification <https://gsd.readthedocs.io/en/stable/shapes.html>`_
        section of the GSD package documentation for a detailed description of shape definitions.

        .. versionadded:: 2.7
        """
        if hasattr(obj, '_connect_gsd_shape_spec') and type(getattr(obj, '_connect_gsd_shape_spec')) == types.MethodType:
            obj._connect_gsd_shape_spec(self);
        else:
            hoomd.context.msg.warning("GSD is not currently support for {}".format(obj.__class__.__name__));

    @property
    def log(self):
        """Dictionary mapping user-defined names to callbacks.

        Add an item to :py:attr:`log` to save user-defined data in the gsd file. The key provides the name of the data
        chunk in the gsd file (e.g. ``particles/lj_potential_energy``). The value is a callable python object that takes
        the current time step as an argument and returns a numpy array that has 1 or 2 dimensions and has a data type
        supported by `gsd <https://gsd.readthedocs.io>`_.

        Delete a key from :py:attr:`log` to stop logging that quantity.

        .. note::

            All logged data chunks must be present in the first frame in the gsd file to provide the default value. Some
            (or all) chunks may be omitted on later frames:

        .. note::

            In MPI parallel simulations, the callback will be called on all ranks. :py:class:`gsd` will write the data
            returned by the root rank. Return values from all other ranks are ignored (and may be None).

        .. versionadded:: 2.7
        """
        return self.cpp_analyzer.user_log;
