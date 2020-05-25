from hoomd import _hoomd
from hoomd.local_access import (
    _ParticleLocalAccess, _GroupLocalAccess, _LocalSnapshotBase)
from hoomd.hoomd_array import HOOMDGPUArray

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

    class LocalSnapshotGPU(NoGPU):
        pass
