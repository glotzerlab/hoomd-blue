# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Force data local access.

`ForceLocalAccess`, `ForceLocalAccessGPU`, and related classes provide direct
access to the data buffers managed by `hoomd.md.force.Force`. This means that
MPI rank locality must be considered in accessing the arrays in a multi-rank
simulation.

See Also:
    `hoomd.md.force.Force`

    `hoomd.md.force.Custom`
"""

from .local_access import _ForceLocalAccessBase, _NeighborListLocalAccessBase
from .local_access_cpu import ForceLocalAccess, NeighborListLocalAccess
from .local_access_gpu import ForceLocalAccessGPU, NeighborListLocalAccessGPU
