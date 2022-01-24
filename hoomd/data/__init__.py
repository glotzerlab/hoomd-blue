# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Particle data local access."""

from .array import HOOMDArray, HOOMDGPUArray
from .local_access import (AngleLocalAccessBase, BondLocalAccessBase,
                           ConstraintLocalAccessBase, DihedralLocalAccessBase,
                           ImproperLocalAccessBase, PairLocalAccessBase,
                           ParticleLocalAccessBase)
from .local_access_cpu import LocalSnapshot
from .local_access_gpu import LocalSnapshotGPU
