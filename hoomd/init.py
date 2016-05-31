# Copyright (c) 2009-2016 The Regents of the University of Michigan
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
    if hoomd.context.current.system is None:
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
    hoomd.util.print_status_line();

    hoomd.context._verify_init();

    # check if initialization has already occured
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


def read_snapshot(snapshot):
    R""" Initializes the system from a snapshot.

    Args:
        snapshot (:py:class:`hoomd.data.snapshot`): The snapshot to initialize the system.

    Snapshots temporarily store system data. Snapshots contain the complete simulation state in a
    single object. They can be used to start or restart a simulation.

    Example use cases in which a simulation may be started from a snapshot include user code that generates initial
    particle positions.

    Example::

        snapshot = my_system_create_routine(.. parameters ..)
        system = init.read_snapshot(snapshot)

    See Also:
        :py:mod:`hoomd.data`
    """
    hoomd.util.print_status_line();

    hoomd.context._verify_init();

    # check if initialization has already occured
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    # broadcast snapshot metadata so that all ranks have _global_box (the user may have set box only on rank 0)
    snapshot._broadcast(hoomd.context.exec_conf);
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
        frame (int): Index of the frame to read from the GSD file.
        time_step (int): (if specified) Time step number to initialize instead of the one stored in the GSD file.

    All particles, bonds, angles, dihedrals, impropers, constraints, and box information
    are read from the given GSD file at the given frame index. To read and write GSD files
    outside of hoomd, see http://gsd.readthedocs.io/. :py:class:`hoomd.dump.gsd` writes GSD files.

    For restartable jobs, specify the initial condition in *filename* and the restart file in *restart*.
    :py:func:`hoomd.init.read_gsd` will read the restart file if it exists, otherwise it will read *filename*.

    If *time_step* is specified, its value will be used as the initial time
    step of the simulation instead of the one read from the GSD file.

    The result of :py:func:`hoomd.init.read_gsd` can be saved in a variable and later used to read and/or
    change particle properties later in the script. See :py:mod:`hoomd.data` for more information.

    See Also:
        :py:class:`hoomd.dump.gsd`
    """
    hoomd.util.print_status_line();

    hoomd.context._verify_init();

    # check if initialization has already occured
    if is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");


    reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, frame);
    snapshot = reader.getSnapshot();
    if time_step is None:
        time_step = reader.getTimeStep();

    # broadcast snapshot metadata so that all ranks have _global_box (the user may have set box only on rank 0)
    snapshot._broadcast(hoomd.context.exec_conf);
    my_domain_decomposition = _create_domain_decomposition(snapshot._global_box);

    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, time_step);

    _perform_common_init_tasks();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

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

