# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Data initialization commands

Commands in the :py:mod:`hoomd.init` package initialize the particle system.

"""

from hoomd import _hoomd
import hoomd;

import math;
import sys;
import gc;
import os;
import re;
import platform;

## Tests if the system has been initialized
#
# Returns True if a previous init.create* or init.read* command has completed successfully and initialized the system.
# Returns False otherwise.
def is_initialized():
    if hoomd.context.current is None or hoomd.context.current.system is None:
        return False;
    else:
        return True;

def create_lattice(unitcell, n):
    R""" Create a lattice.

    Args:
        unitcell (:py:class:`hoomd.lattice.unitcell`): The unit cell of the lattice.
        n (list): Number of replicates in each direction.

    :py:func:`create_lattice` take a unit cell and replicates it the requested number of times in each direction.
    The resulting simulation box is commensurate with the given unit cell. A generic :py:class:`hoomd.lattice.unitcell`
    may have arbitrary vectors :math:`\vec{a}_1`, :math:`\vec{a}_2`, and :math:`\vec{a}_3`. :py:func:`create_lattice`
    will rotate the unit cell so that :math:`\vec{a}_1` points in the :math:`x` direction and :math:`\vec{a}_2`
    is in the :math:`xy` plane so that the lattice may be represented as a HOOMD simulation box.

    When *n* is a single value, the lattice is replicated *n* times in each direction. When *n* is a list, the
    lattice is replicated *n[0]* times in the :math:`\vec{a}_1` direction, *n[1]* times in the :math:`\vec{a}_2`
    direction and *n[2]* times in the :math:`\vec{a}_3` direction.

    Examples::

        hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=1.0),
                                  n=[2,4,2]);

        hoomd.init.create_lattice(unitcell=hoomd.lattice.bcc(a=1.0),
                                  n=10);

        hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=1.2),
                                  n=[100,10]);

        hoomd.init.create_lattice(unitcell=hoomd.lattice.hex(a=1.0),
                                  n=[100,58]);
    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    hoomd.util.quiet_status();

    snap = unitcell.get_snapshot();
    try:
        l = len(n);
    except TypeError:
        l = snap.box.dimensions;
        n = [n]*l;

    if l != snap.box.dimensions:
        hoomd.context.msg.error("n must have length equal to the number of dimensions in the unit cell\n");
        raise RuntimeError("Error initializing");

    if snap.box.dimensions == 3:
        snap.replicate(n[0],n[1],n[2])
    if snap.box.dimensions == 2:
        snap.replicate(n[0],n[1],1)

    read_snapshot(snapshot=snap);

    hoomd.util.unquiet_status();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

def read_getar(filename, modes={'any': 'any'}):
    """Initialize a system from a trajectory archive (.tar, .getar,
    .sqlite) file. Returns a HOOMD `system_data` object.

    Args:
        filename (str): Name of the file to read from
        modes (dict): dictionary of {property: frame} values; see below

    Getar files are a simple interface on top of archive formats (such
    as zip and tar) for storing trajectory data efficiently. A more
    thorough description of the format and a description of a python
    API to read and write these files is available at `the libgetar
    documentation <http://libgetar.readthedocs.io>`_.

    The **modes** argument is a dictionary. The keys of this
    dictionary should be either property names (see the Supported
    Property Table below) or tuples of property names.

    If the key is a tuple of property names, data for those names will
    be restored from the same frame. Other acceptable keys are "any" to
    restore any properties which are present from the file, "angle_any"
    to restore any angle-related properties present, "bond_any", and so
    forth. The values associated with each key in the dictionary should
    be "any" (in which case any frame present for the data will be
    restored, even if the frames are different for two property names in
    a tuple), "latest" (grab the most recent frame data), "earliest", or
    a specific timestep value.

    Example::

        # creating file to initialize beforehand using libgetar
        with gtar.GTAR('init.zip', 'w') as traj:
            traj.writePath('position.f32.ind', positions)
            traj.writePath('velocity.f32.ind', velocities)
            traj.writePath('metadata.json', json.dumps(metadata))
        system = hoomd.init.read_getar('init.zip')
        # using the backup created in the `hoomd.dump.getar.simple` example
        system = hoomd.init.read_getar('backup.tar')

    **Supported Property Table**

    .. tabularcolumns:: |p{0.25 \textwidth}|p{0.1 \textwidth}|p{0.2 \textwidth}|p{0.45 \textwidth}|
    .. csv-table::
       :header: "Name", "Type", "Shape", "Notes"
       :widths: 1, 1, 1, 5

       "angle_type_names", "JSON [String]", "(N_angle_types,)", "list containing the name of each angle type in JSON format"
       "angle_tag", "unsigned int", "(N_angle, 3)", "array of particle tags for each angle interaction"
       "angle_type", "unsigned int", "(N_angle,)", "array of angle interaction types"
       "angular_momentum", "float", "(N, 4)", "per-particle angular momentum quaternion"
       "body", "int", "(N,)", "particle rigid body index"
       "bond_type_names", "JSON [String]", "(N_bond_types,)", "list containing the name of each bond type in JSON format"
       "bond_tag", "unsigned int", "(N_bond, 2)", "array of particle tags for each bond interaction"
       "bond_type", "unsigned int", "(N_bond,)", "array of bond interaction types"
       "box", "float", "(6,)", "vector of box lengths (x, y, z, tilt_xy, tilt_xz, tilt_yz); can be high precision"
       "charge", "float", "(N,)", "particle charge"
       "diameter", "float", "(N,)", "particle diameter"
       "dihedral_type_names", "JSON [String]", "(N_dihedral_types,)", "list containing the name of each dihedral type in JSON format"
       "dihedral_tag", "unsigned int", "(N_dihedral, 4)", "array of particle tags for each dihedral interaction"
       "dihedral_type", "unsigned int", "(N_dihedral,)", "array of dihedral interaction types"
       "dimensions", "unsigned int", "1", "number of dimensions of the system"
       "image", "int", "(N, 3)", "how many times each particle has passed through the periodic boundary conditions"
       "improper_type_names", "JSON [String]", "(N_improper_types,)", "list containing the name of each improper type in JSON format"
       "improper_tag", "unsigned int", "(N_improper, 4)", "array of particle tags for each improper interaction"
       "improper_type", "unsigned int", "(N_improper,)", "array of improper interaction types"
       "mass", "float", "(N,)", "particle mass"
       "moment_inertia", "float", "(N, 3)", "moment of inertia of each particle (diagonalized)."
       "orientation", "float", "(N, 4)", "particle orientation, expressed as a quaternion in the order (real, imag_i, imag_j, imag_k); can be high precision"
       "position", "float", "(N, 3)", "the position of each particle in the system (can be high precision)"
       "potential_energy", "float", "(N,)", "per-particle potential energy; can't be used in MPI runs"
       "type", "unsigned int", "(N,)", "particle numerical type index"
       "type_names", "JSON [String]", "(N_types,)", "list containing the name of each particle type in JSON format"
       "velocity", "float", "(N, 3)", "velocity of each particle in the system"

    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    newModes = _parse_getar_modes(modes);
    # read in the data
    initializer = _hoomd.GetarInitializer(hoomd.context.exec_conf, filename);
    snapshot = initializer.initialize(newModes);

    # broadcast snapshot metadata so that all ranks have _global_box (the user may have set box only on rank 0)
    snapshot._broadcast_box(hoomd.context.exec_conf);

    try:
        box = snapshot._global_box;
    except AttributeError:
        box = snapshot.box;

    my_domain_decomposition = _create_domain_decomposition(box);
    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(
            snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(
            snapshot, hoomd.context.exec_conf);

    hoomd.context.current.system = _hoomd.System(
        hoomd.context.current.system_definition, initializer.getTimestep());

    _perform_common_init_tasks();

    if (hoomd.data.get_snapshot_box(snapshot).dimensions == 2 and
        any(abs(z) > 1e-5 for z in snapshot.particles.position[:, 2])):
        raise RuntimeWarning('Initializing a 2D system with some z '
                             'components out-of-plane');

    return hoomd.data.system_data(hoomd.context.current.system_definition);

def read_snapshot(snapshot):
    R""" Initializes the system from a snapshot.

    Args:
        snapshot (:py:mod:`hoomd.data` snapshot): The snapshot to initialize the system.

    Snapshots temporarily store system data. Snapshots contain the complete simulation state in a
    single object. Snapshots are set to time_step 0, and should not be used to restart a simulation.

    Example use cases in which a simulation may be started from a snapshot include user code that generates initial
    particle positions.

    Example::

        snapshot = my_system_create_routine(.. parameters ..)
        system = init.read_snapshot(snapshot)

    See Also:
        :py:mod:`hoomd.data`
    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    # broadcast snapshot metadata so that all ranks have _global_box (the user may have set box only on rank 0)
    snapshot._broadcast_box(hoomd.context.exec_conf);
    my_domain_decomposition = _create_domain_decomposition(snapshot._global_box);

    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, 0);

    _perform_common_init_tasks();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

def read_gsd(filename, restart = None, frame = 0, time_step = None):
    R""" Read initial system state from an GSD file.

    Args:
        filename (str): File to read.
        restart (str): If it exists, read the file *restart* instead of *filename*.
        frame (int): Index of the frame to read from the GSD file. Negative values index from the end of the file.
        time_step (int): (if specified) Time step number to initialize instead of the one stored in the GSD file.

    All particles, bonds, angles, dihedrals, impropers, constraints, and box information
    are read from the given GSD file at the given frame index. To read and write GSD files
    outside of hoomd, see http://gsd.readthedocs.io/. :py:class:`hoomd.dump.gsd` writes GSD files.

    For restartable jobs, specify the initial condition in *filename* and the restart file in *restart*.
    :py:func:`hoomd.init.read_gsd` will read the restart file if it exists, otherwise it will read *filename*.

    If *time_step* is specified, its value will be used as the initial time
    step of the simulation instead of the one read from the GSD file *filename*.
    *time_step* is not applied when the file *restart* is read.

    The result of :py:func:`hoomd.init.read_gsd` can be saved in a variable and later used to read and/or
    change particle properties later in the script. See :py:mod:`hoomd.data` for more information.

    See Also:
        :py:class:`hoomd.dump.gsd`
    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    filename = _hoomd.mpi_bcast_str(filename, hoomd.context.exec_conf);
    restart = _hoomd.mpi_bcast_str(restart, hoomd.context.exec_conf);

    if restart is not None and os.path.exists(restart):
        reader = _hoomd.GSDReader(hoomd.context.exec_conf, restart, abs(frame), frame < 0);
        time_step = reader.getTimeStep();
    else:
        reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, abs(frame), frame < 0);
        if time_step is None:
            time_step = reader.getTimeStep();

    snapshot = reader.getSnapshot();

    # broadcast snapshot metadata so that all ranks have _global_box (the user may have set box only on rank 0)
    snapshot._broadcast_box(hoomd.context.exec_conf);
    my_domain_decomposition = _create_domain_decomposition(snapshot._global_box);

    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, time_step);

    _perform_common_init_tasks();
    hoomd.context.current.state_reader = reader;
    hoomd.context.current.state_reader.clearSnapshot();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

def restore_getar(filename, modes={'any': 'any'}):
    """Restore a subset of the current system's parameters from a
    trajectory archive (.tar, .zip, .sqlite) file. For a detailed
    discussion of arguments, see :py:func:`read_getar`.

    Args:
        filename (str): Name of the file to read from
        modes (dict): dictionary of {property: frame} values, as described in :py:func:`read_getar`
    """
    hoomd.util.print_status_line();

    # the getar initializer opens the file on all ranks: need to broadcast the string from rank 0
    filename_bcast = _hoomd.mpi_bcast_str(filename, hoomd.context.exec_conf);
    initializer = _hoomd.GetarInitializer(hoomd.context.exec_conf, filename_bcast);

    newModes = _parse_getar_modes(modes);

    initializer.restore(newModes, hoomd.context.current.system_definition);
    del initializer;

## Performs common initialization tasks
#
# \internal
# Initialization tasks that are performed for every simulation are to
# be done here. For example, setting up communication, registering the
# SFCPackUpdater, initializing the log writer, etc...
def _perform_common_init_tasks():
    # create the sorter
    hoomd.context.current.sorter = hoomd.update.sort();

    # create the default compute.thermo on the all group
    hoomd.util.quiet_status();
    all = hoomd.group.all();
    hoomd.compute._get_unique_thermo(group=all);
    hoomd.util.unquiet_status();

    # set up Communicator, and register it with the System
    if _hoomd.is_MPI_available():
        cpp_decomposition = hoomd.context.current.system_definition.getParticleData().getDomainDecomposition();
        if cpp_decomposition is not None:
            # create the c++ Communicator
            if not hoomd.context.exec_conf.isCUDAEnabled():
                cpp_communicator = _hoomd.Communicator(hoomd.context.current.system_definition, cpp_decomposition)
            else:
                cpp_communicator = _hoomd.CommunicatorGPU(hoomd.context.current.system_definition, cpp_decomposition)

            # set Communicator in C++ System
            hoomd.context.current.system.setCommunicator(cpp_communicator)

## Create a DomainDecomposition object
# \internal
def _create_domain_decomposition(box):
    if not _hoomd.is_MPI_available():
        return None

    # if we are only running on one processor, we use optimized code paths
    # for single-GPU execution
    if hoomd.context.exec_conf.getNRanks() == 1:
        return None

    # okay, we want a decomposition but one isn't set, so make a default one
    if hoomd.context.current.decomposition is None:
        # this is happening transparently to the user, so hush this up
        hoomd.util.quiet_status()
        hoomd.context.current.decomposition = hoomd.comm.decomposition()
        hoomd.util.unquiet_status()

    return hoomd.context.current.decomposition._make_cpp_decomposition(box)

def _parse_getar_modes(modes):
    newModes = {}
    for key in modes:
        if type(key) == str:
            newModes[(key,)] = str(modes[key])
        else:
            newModes[tuple(key)] = str(modes[key])

    return newModes
