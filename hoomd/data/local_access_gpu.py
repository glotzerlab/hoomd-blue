from hoomd import _hoomd
from hoomd.data.local_access import (
    _ParticleLocalAccess, _GroupLocalAccess, _LocalSnapshotBase)
from hoomd.array import HOOMDGPUArray

if _hoomd.isCUDAAvailable():
    class ParticleLocalAccessGPU(_ParticleLocalAccess):
        _cpp_cls = _hoomd.LocalParticleDataDevice
        _array_cls = HOOMDGPUArray

    class BondLocalAccessGPU(_GroupLocalAccess):
        _cpp_cls = _hoomd.LocalBondDataDevice
        _cpp_data_get_method = "getBondData"
        _array_cls = HOOMDGPUArray

    class AngleLocalAccessGPU(_GroupLocalAccess):
        _cpp_cls = _hoomd.LocalAngleDataDevice
        _cpp_data_get_method = "getAngleData"
        _array_cls = HOOMDGPUArray

    class DihedralLocalAccessGPU(_GroupLocalAccess):
        _cpp_cls = _hoomd.LocalDihedralDataDevice
        _cpp_data_get_method = "getDihedralData"
        _array_cls = HOOMDGPUArray

    class ImproperLocalAccessGPU(_GroupLocalAccess):
        _cpp_cls = _hoomd.LocalImproperDataDevice
        _cpp_data_get_method = "getImproperData"
        _array_cls = HOOMDGPUArray

    class ConstraintLocalAccessGPU(_GroupLocalAccess):
        _fields = {
            'value': 'getTypeVal',
            'group': 'getMembers',
            'tag': 'getTags',
            'rtag': 'getRTags'
        }
        _cpp_cls = _hoomd.LocalConstraintDataDevice
        _cpp_data_get_method = "getConstraintData"
        _array_cls = HOOMDGPUArray

    class PairLocalAccessGPU(_GroupLocalAccess):
        _cpp_cls = _hoomd.LocalPairDataDevice
        _cpp_data_get_method = "getPairData"
        _array_cls = HOOMDGPUArray

    class LocalSnapshotGPU(_LocalSnapshotBase):
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
    from hoomd.util import NoGPU

    class BondLocalAccessGPU(NoGPU):
        pass

    class AngleLocalAccessGPU(NoGPU):
        pass

    class DihedralLocalAccessGPU(NoGPU):
        pass

    class ImproperLocalAccessGPU(NoGPU):
        pass

    class ConstraintLocalAccessGPU(NoGPU):
        pass

    class PairLocalAccessGPU(NoGPU):
        pass

    class ParticleLocalAccessGPU(NoGPU):
        pass

    class LocalSnapshotGPU(NoGPU, _LocalSnapshotBase):
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
suffixing with ``_with_ghosts`` will grab all data on the rank (regular and
ghost particles) in a read only array.

All array-like properties return a `hoomd.array.HOOMDGPUArray` object which
prevents invalid memory accesses.

Note:
    See the CPU ``LocalAccess`` classes for available properties.
"""

LocalSnapshotGPU.__doc__ = _gpu_snapshot_docs
