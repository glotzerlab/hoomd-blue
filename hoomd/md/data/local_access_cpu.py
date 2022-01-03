# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement local access classes for the CPU."""

from hoomd.md.data.local_access import _ForceLocalAccessBase
from hoomd.data.array import HOOMDArray
from hoomd import _hoomd


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
