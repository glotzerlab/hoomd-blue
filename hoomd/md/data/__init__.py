# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Force data local access."""

from .local_access import _ForceLocalAccessBase
from .local_access_cpu import ForceLocalAccess
from .local_access_gpu import ForceLocalAccessGPU
