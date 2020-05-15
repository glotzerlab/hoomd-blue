from hoomd import _hoomd
from hoomd.local_access import _ParticleLocalAccess, _LocalSnapshotBase
from hoomd.hoomd_array import HOOMDGPUArray

if _hoomd.isCUDAAvailable():
    class ParticleLocalAccessGPU(_ParticleLocalAccess):
        _cpp_cls = _hoomd.LocalParticleDataDevice
        _array_cls = HOOMDGPUArray

    class LocalSnapshotGPU(_LocalSnapshotBase):
        def __init__(self, state):
            self._particles = ParticleLocalAccessGPU(state)

else:
    from hoomd.util import NoGPU

    class ParticleLocalAccessGPU(NoGPU):
        pass

    class LocalSnapshotGPU(NoGPU):
        pass
