# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD system initialization

Commands to initialize the MPCD system data. Currently, random initialization
and snapshot initialization (see :py:mod:`hoomd.mpcd.data`) are supported.
Random initialization is useful for large systems where a snapshot is impractical.
Snapshot initialization is useful when you require fine control over the particle
properties and initial configuration.

"""

import hoomd

from . import _mpcd
from . import data

def make_random(N, kT, seed):
    R"""Initialize particles randomly

    Args:
        N (int): Total number of MPCD particles
        kT (float): Temperature of MPCD particles (in energy units)
        seed (int): Random seed for initialization

    Returns:
        Initialized MPCD system data (:py:class:`hoomd.mpcd.data.system`)

    MPCD particles are randomly initialized into the simulation box.
    An MPCD system can be randomly initialized only **after** the HOOMD system
    is first initialized (see :py:mod:`hoomd.init`). The system can only be
    initialized one time. The total number of particles *N* is evenly divided
    between all domains. Random positions are then drawn uniformly within the
    (local) box. Particle velocities are drawn from a Maxwell-Boltzmann
    distribution consistent with temperature *kT*. All MPCD particles are given
    unit mass and type A.

    Examples::

        mpcd.init.make_random(N=1250000000, kT=1.0, seed=42)

    Notes:
        Random number generation is performed using C++11 ``mt19937`` seeded by
        *seed* plus the rank number in MPI simulations. This random number
        generator is separate from other generators used in MPCD, so *seed* can
        be reasonably recycled elsewhere.

    """
    hoomd.util.print_status_line()

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("mpcd: HOOMD system must be initialized before mpcd\n")
        raise RuntimeError("HOOMD system not initialized")

    if hoomd.context.current.mpcd is not None:
        hoomd.context.msg.error("mpcd: system is already initialized, cannot reinitialize\n")
        raise RuntimeError("mpcd system already initialized")

    # make particle data first
    sysdef = hoomd.context.current.system_definition
    box = sysdef.getParticleData().getBox()
    if hoomd.context.current.decomposition:
        pdata = _mpcd.MPCDParticleData(N, box, kT, seed, sysdef.getNDimensions(), hoomd.context.exec_conf, hoomd.context.current.decomposition.cpp_dd)
    else:
        pdata = _mpcd.MPCDParticleData(N, box, kT, seed, sysdef.getNDimensions(), hoomd.context.exec_conf)

    # then make mpcd system
    hoomd.context.current.mpcd = data.system(_mpcd.SystemData(sysdef,pdata))
    return hoomd.context.current.mpcd

def read_snapshot(snapshot):
    R"""Initialize from a snapshot

    Args:
        snapshot (:py:class:`hoomd.mpcd.data.snapshot`): MPCD system data snapshot

    Returns:
        Initialized MPCD system data (:py:class:`hoomd.mpcd.data.system`)

    An MPCD system can be initialized from a snapshot **after** the HOOMD system
    is first initialized (see :py:mod:`hoomd.init`). The system can only be
    initialized one time. If no type is specified in the snapshot, a default type
    *A* will be assigned to the MPCD particles.

    Examples::

        snap = mpcd.data.make_snapshot(N=10)
        snap.particles.position[:] = L * np.random.random((10,3))
        mpcd.init.read_snapshot(snap)

    Notes:
        It is expected that the snapshot has the same box size as the HOOMD
        system. By default, this is how a new snapshot is initialized. If the
        HOOMD system is resized after the MPCD snapshot is created and before
        initialization from the MPCD snapshot, an error will be raised if the
        MPCD snapshot is not properly resized.

    """
    hoomd.util.print_status_line();

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("mpcd: HOOMD system must be initialized before mpcd\n")
        raise RuntimeError("HOOMD system not initialized")

    if hoomd.context.current.mpcd is not None:
        hoomd.context.msg.error("mpcd: system is already initialized, cannot reinitialize\n")
        raise RuntimeError("mpcd system already initialized")

    hoomd.context.current.mpcd = data.system(_mpcd.SystemData(snapshot.sys_snap))
    return hoomd.context.current.mpcd
