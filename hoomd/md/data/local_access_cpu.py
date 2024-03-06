# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement local access classes for the CPU."""

from hoomd.md.data.local_access import _ForceLocalAccessBase, \
    _NeighborListLocalAccessBase
from hoomd.data.array import HOOMDArray
from hoomd import _hoomd
from hoomd.md import _md


class ForceLocalAccess(_ForceLocalAccessBase):
    """Access HOOMD-Blue force data buffers on the CPU.

    Attributes:
        force ((N_particles, 3) `hoomd.data.array` of ``float``):
            Local force data. :math:`[\\mathrm{force}]`
        potential_energy ((N_particles,) `hoomd.data.array` of ``float``):
            Local potential energy data. :math:`[\\mathrm{energy}]`
        torque ((N_particles, 3) `hoomd.data.array` of ``float``):
            Local torque data. :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`
        virial ((N_particles, 6) `hoomd.data.array` of ``float``):
            Local virial data. :math:`[\\mathrm{energy}]`
    """

    _cpp_cls = _hoomd.LocalForceComputeDataHost
    _array_cls = HOOMDArray


class NeighborListLocalAccess(_NeighborListLocalAccessBase):
    """Access HOOMD-Blue neighbor list data buffers on the CPU.

    The internal `NeighborList` implementation of HOOMD is comprised of
    essentially three array buffers. The buffers are:

    * ``nlist``: Ragged array of neighbor data.
    * ``head_list``: Indexes for particles to read from the neighbor list.
    * ``n_neigh``: Number of neighbors for each particle.

    The neighbor indices of particle :math:`i` are stored in the slice
    ``nlist[head_list[i]:head_list[i]+n_neigh[i]]``. The result of access
    outside of these bounds is undefined. The `half_nlist` property is used to
    query whether the neighbor list stores a single copy for each pair (True),
    or two copies for each pair (False). Under MPI, pairs that cross domains
    are stored twice, once in each domain rank.

    Attributes:
        head_list ((N_particles,) `hoomd.data.array` of ``unsigned long``):
            Local head list.
        n_neigh ((N_particles,) `hoomd.data.array` of ``unsigned int``):
            Number of neighbors.
        nlist ((...) `hoomd.data.array` of ``unsigned int``):
            Raw neighbor list data.
        half_nlist (``bool``):
            Convenience property to check if the storage mode is 'half'.
    """

    _cpp_cls = _md.LocalNeighborListDataHost
    _array_cls = HOOMDArray
