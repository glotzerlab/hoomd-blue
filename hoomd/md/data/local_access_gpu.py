# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement local access classes for the GPU."""

import hoomd
from hoomd.md import _md
from hoomd.data.array import HOOMDGPUArray
from hoomd.md.data.local_access import ForceLocalAccessBase

if hoomd.version.gpu_enabled:

    class ForceLocalAccessGPU(ForceLocalAccessBase):
        """Access force array data on the GPU."""
        _cpp_cls = _md.LocalForceComputeDataDevice
        _array_cls = HOOMDGPUArray

else:
    from hoomd.util import _NoGPU

    class ForceLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass
