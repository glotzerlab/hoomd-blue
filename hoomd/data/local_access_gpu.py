# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement local access classes for the GPU."""

from hoomd import _hoomd
from hoomd.data.local_access import (
    ParticleLocalAccessBase, BondLocalAccessBase, ConstraintLocalAccessBase,
    DihedralLocalAccessBase, AngleLocalAccessBase, ImproperLocalAccessBase,
    PairLocalAccessBase, _LocalSnapshot)

from hoomd.data.array import HOOMDGPUArray
import hoomd

if hoomd.version.gpu_enabled:

    class ParticleLocalAccessGPU(ParticleLocalAccessBase):
        """Access particle data on the GPU."""
        _cpp_cls = _hoomd.LocalParticleDataDevice
        _array_cls = HOOMDGPUArray

    class BondLocalAccessGPU(BondLocalAccessBase):
        """Access bond data on the GPU."""
        _cpp_cls = _hoomd.LocalBondDataDevice
        _array_cls = HOOMDGPUArray

    class AngleLocalAccessGPU(AngleLocalAccessBase):
        """Access angle data on the GPU."""
        _cpp_cls = _hoomd.LocalAngleDataDevice
        _array_cls = HOOMDGPUArray

    class DihedralLocalAccessGPU(DihedralLocalAccessBase):
        """Access dihedral data on the GPU."""
        _cpp_cls = _hoomd.LocalDihedralDataDevice
        _array_cls = HOOMDGPUArray

    class ImproperLocalAccessGPU(ImproperLocalAccessBase):
        """Access improper data on the GPU."""
        _cpp_cls = _hoomd.LocalImproperDataDevice
        _array_cls = HOOMDGPUArray

    class ConstraintLocalAccessGPU(ConstraintLocalAccessBase):
        """Access constraint data on the GPU."""
        _cpp_cls = _hoomd.LocalConstraintDataDevice
        _array_cls = HOOMDGPUArray

    class PairLocalAccessGPU(PairLocalAccessBase):
        """Access special pair data on the GPU."""
        _cpp_cls = _hoomd.LocalPairDataDevice
        _array_cls = HOOMDGPUArray

    class LocalSnapshotGPU(_LocalSnapshot):
        """Access system state data on the GPU."""

        def __init__(self, state):
            super().__init__(state)
            self._particles = ParticleLocalAccessGPU(state)
            self._bonds = BondLocalAccessGPU(state)
            self._angles = AngleLocalAccessGPU(state)
            self._dihedrals = DihedralLocalAccessGPU(state)
            self._impropers = ImproperLocalAccessGPU(state)
            self._pairs = PairLocalAccessGPU(state)
            self._constraints = ConstraintLocalAccessGPU(state)

else:
    from hoomd.error import _NoGPU

    class BondLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class AngleLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class DihedralLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class ImproperLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class ConstraintLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class PairLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class ParticleLocalAccessGPU(_NoGPU):
        """GPU data access is not available in CPU builds."""
        pass

    class LocalSnapshotGPU(_NoGPU, _LocalSnapshot):
        """GPU data access is not available in CPU builds."""
        pass


_gpu_snapshot_docs = """
Provides context manager access to HOOMD-blue GPU data buffers.

The interface of a `LocalSnapshot` is similar to that of the `hoomd.Snapshot`.
Data is MPI rank local so for MPI parallel simulations only the data possessed
by a rank is exposed. This means that users must handle the domain decomposition
directly. One consequence of this is that access to ghost particle data is
provided. A ghost particle is a particle that is not owned by a rank, but
nevertheless is required for operations that use particle neighbors. Also,
changing the global or local box within a `LocalSnapshot` context manager is not
allowed.

For every property (e.g. ``data.particles.position``), only grabs the
data for the regular (non-ghost) particles. The property can be prefixed
with ``ghost_`` to grab the ghost particles in a read only manner. Likewise,
suffixing with ``_with_ghost`` will grab all data on the rank (regular and
ghost particles) in a read only array.

All array-like properties return a `hoomd.data.array.HOOMDGPUArray` object which
prevents invalid memory accesses.

See Also:
    Access the local snapshot of a state via
    `hoomd.State.gpu_local_snapshot`.
"""

LocalSnapshotGPU.__doc__ = _gpu_snapshot_docs
