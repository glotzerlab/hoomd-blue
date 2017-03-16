# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD system initialization

Commands to initialize the MPCD system data. Currently, only snapshot
initialization (see :py:mod:`hoomd.mpcd.data`) is supported.

"""

import hoomd

from . import _mpcd
from . import data

def read_snapshot(snapshot):
    R"""Initialize from a snapshot

    Args:
        snapshot (:py:class:`hoomd.mpcd.data.snapshot`): MPCD system data snapshot

    Returns:
        Initialized MPCD system data (:py:class:`hoomd.mpcd.data.system`)

    An MPCD system can be initialized from a snapshot **after** the HOOMD system
    is first initialized (see :py:mod:`hoomd.init`). The system can only be
    initialized one time.

    Examples::

        snap = mpcd.data.make_snapshot(N=10)
        snap.particles.positions[:] = L * np.random((10,3))
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
