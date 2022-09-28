# Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

    Attributes:
        head_list ((N_particles,) `hoomd.data.array` of ``int``):
            Local force data. :math:`[\\mathrm{force}]`
        n_neigh ((N_particles,) `hoomd.data.array` of ``int``):
            Local potential energy data. :math:`[\\mathrm{energy}]`
        nlist (() `hoomd.data.array` of ``int``):
            Local torque data. :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`
    """

    _cpp_cls = _md.LocalNeighborListDataHost
    _array_cls = HOOMDArray
