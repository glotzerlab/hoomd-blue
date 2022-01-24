# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement local access classes for the GPU."""

import hoomd
from hoomd import _hoomd
from hoomd.data.array import HOOMDGPUArray
from hoomd.md.data.local_access import _ForceLocalAccessBase

if hoomd.version.gpu_enabled:

    class ForceLocalAccessGPU(_ForceLocalAccessBase):
        """Access force array data on the GPU."""
        _cpp_cls = _hoomd.LocalForceComputeDataDevice
        _array_cls = HOOMDGPUArray

else:
    from hoomd.util import _NoGPU

    class ForceLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass


_gpu_force_access_docs = """
Access HOOMD-Blue force data buffers on the GPU.

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

ForceLocalAccessGPU.__doc__ = _gpu_force_access_docs
