# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement local access classes for the CPU."""

from hoomd.md.data.local_access import ForceLocalAccessBase
from hoomd.data.array import HOOMDArray
from hoomd.md import _md


class ForceLocalAccess(ForceLocalAccessBase):
    """Access force data on the CPU."""

    _cpp_cls = _md.LocalForceComputeDataHost
    _array_cls = HOOMDArray

    def __init__(self, force_obj):
        super().__init__(force_obj)
