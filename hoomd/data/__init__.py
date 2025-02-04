# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Access State data on the local rank.

`LocalSnapshot`, `LocalSnapshotGPU`, and related classes provide direct access
to the data buffers managed by `hoomd.State`.

See Also:
    `hoomd.State`
"""

from .array import HOOMDArray, HOOMDGPUArray
from .local_access import (AngleLocalAccessBase, BondLocalAccessBase,
                           ConstraintLocalAccessBase, DihedralLocalAccessBase,
                           ImproperLocalAccessBase, PairLocalAccessBase,
                           ParticleLocalAccessBase)
from .local_access_cpu import LocalSnapshot
from .local_access_gpu import LocalSnapshotGPU
