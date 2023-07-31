# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

R""" MPCD data structures

.. rubric:: MPCD and HOOMD

MPCD data is currently initialized in a secondary step from HOOMD using a
snapshot interface. Even if only MPCD particles are present in the system, an
empty HOOMD system must first be created. Once the HOOMD system has been
initialized (see ``hoomd.init``), an MPCD snapshot can be created using::

    >>> snap = mpcd.data.make_snapshot(100)

The MPCD system can then initialized from the snapshot (see :py:mod:`hoomd.mpcd.init`)::

    >>> mpcd.init.read_snapshot(snap)

Because the MPCD data is stored separately from the HOOMD data, special care
must be taken when using certain commands that operate on the HOOMD system data.
For example, the HOOMD box size is not permitted to be changed after
the MPCD particle data has been initialized (see :py:mod:`hoomd.mpcd.init`), so any
resizes or replications of the HOOMD system must occur before then::

    >>> hoomd_sys.replicate(2,2,2)
    >>> snap.replicate(2,2,2)
    >>> mpcd_sys = mpcd.init.read_snapshot(snap)
    >>> hoomd_sys.replicate(2,1,1)
    **ERROR**

Similarly, :py:class:`~hoomd.update.BoxResize` will also fail after the MPCD
system has been initialized.

During a simulation, the MPCD particle data can be read, modified, and restored
using :py:meth:`~hoomd.mpcd.data.system.take_snapshot()` and
:py:meth:`~hoomd.mpcd.data.system.restore_snapshot()`::

    snap = mpcd_sys.take_snapshot()
    # modify snapshot
    mpcd_sys.restore_snapshot(snap)

.. rubric:: MPCD and MPI

MPCD supports MPI parallelization through domain decomposition. The MPCD data
in the snapshot is only valid on rank 0, and is distributed to all ranks through
the snapshot collective calls.

.. rubric:: Particle data

All MPCD particle data is accessible through the ``particles`` snapshot property.
The size of the MPCD particle data ``N`` can be resized::

    >>> snap.particles.resize(200)
    >>> print(snap.particles.N)
    200

Because the number of MPCD particles in a simulation is large, fewer particle
properties are tracked per particle than for standard HOOMD particles. All
particle data can be set as for standard snapshots using numpy arrays. Each
particle is assigned a tag from 0 to ``N`` (exclusive) that is tracked. The
following particle properties are recorded:

* Particle positions are stored as an Nx3 numpy array::

    >>> snap.particles.position[4] = [1., 2., 3.]
    >>> print(snap.particles.position[4])
    [ 1. 2. 3.]

  By default, all positions are initialized with zeros.

* Particle velocities can similarly be manipulated as an Nx3 numpy array::

    >>> snap.particles.velocity[2] = [0.5, 1.5, -0.25]
    >>> print(snap.particles.velocity[2])
    [0.5 1.5 -0.25]

  By default, all velocities are initialized with zeros. It is important to
  reassign these to a sensible value consistent with the temperature of the
  system.

* Each particle can be assigned a type (a name for the kind of the particle).
  First, a list of possible types for the system should be set::

    >>> snap.particles.types = ['A','B']
    print(snapshot.particles.types)

  Then, an index is assigned to each particle corresponding to the type::

    >>> snap.particles.typeid[1] = 1 # B
    >>> snap.particles.typeid[2] = 0 # A

  By default, all particles are assigned a type index of 0, and no types are
  set. If no types are specified, type A is created by default at initialization.

* All MPCD particles have the same mass, which can be accessed or set::

    >>> snap.particles.mass = 1.5
    >>> print(snap.mass)
    1.5

  By default, all particles are assigned unit mass.

"""

import hoomd
from hoomd import _hoomd
from . import _mpcd
from . import update


class system():
    R""" MPCD system data

    Args:
        sysdata (object): C++ representation of the MPCD system data

    This class is not intended to be initialized by the user, but is the result
    returned by :py:mod:`hoomd.mpcd.init`.

    """

    def __init__(self, sysdata):
        self.data = sysdata
        hoomd.context.current.system.addCompute(self.cell, "mpcd_cl")

        # create a cell thermo for the system for collision method and logger
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._thermo = _mpcd.CellThermoCompute(self.data)
        else:
            self._thermo = _mpcd.CellThermoComputeGPU(self.data)
        hoomd.context.current.system.addCompute(self._thermo, "mpcd_thermo")
        # extra thermo for the Andersen thermostat
        self._at_thermo = None

        # if MPI is enabled, automatically add a communicator to the system
        if hoomd.context.current.device.comm.num_ranks > 1:
            if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
                self.comm = _mpcd.Communicator(self.data)
            else:
                self.comm = _mpcd.CommunicatorGPU(self.data)
        else:
            self.comm = None

        self.sorter = None
        self.sorter = update.sort(self)

        # no stream rule by default
        self._stream = None

        # no collision rule by default
        self._collide = None

    @property
    def particles(self):
        return self.data.getParticleData()

    @property
    def cell(self):
        return self.data.getCellList()

    def restore_snapshot(self, snapshot):
        R""" Replaces the current MPCD system state

        Args:
            snapshot (:py:class:`hoomd.mpcd.data.snapshot`): MPCD system snapshot

        The MPCD system data is replaced by the contents of *snapshot*.

        Examples::

            snap = mpcd_sys.take_snapshot()
            snap.particles.typeid[2] = 1
            mpcd_sys.restore_snapshot(snap)

        """

        self.data.initializeFromSnapshot(snapshot.sys_snap)

    def set_params(self, cell=None):
        R""" Set parameters of the MPCD system

        Args:
            cell (float): Edge length of an MPCD cell.

        Every MPCD system is given a cell list for binning particles (see
        :py:mod:`.mpcd.collide`). The size of the cell list sets the length
        scale over which hydrodynamic interactions are resolved. By default,
        the system is given a cell size of 1.0, i.e., the cell sets the unit
        of length, which is typical for most use cases. If your simulation
        has a different fundamental unit of length, you can adjust the
        cell size, but be aware that this will also change the fluid properties.

        """
        if cell is not None:
            self.cell.cell_size = cell
