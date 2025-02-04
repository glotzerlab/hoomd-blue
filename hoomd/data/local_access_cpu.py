# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement local access classes for the CPU."""

from hoomd.data.local_access import (ParticleLocalAccessBase,
                                     BondLocalAccessBase, AngleLocalAccessBase,
                                     DihedralLocalAccessBase,
                                     ImproperLocalAccessBase,
                                     ConstraintLocalAccessBase,
                                     PairLocalAccessBase, _LocalSnapshot)
from hoomd.data.array import HOOMDArray
from hoomd import _hoomd


class ParticleLocalAccessCPU(ParticleLocalAccessBase):
    """Access particle data on the CPU."""
    _cpp_cls = _hoomd.LocalParticleDataHost
    _array_cls = HOOMDArray


class BondLocalAccessCPU(BondLocalAccessBase):
    """Access bond data on the GPU."""
    _cpp_cls = _hoomd.LocalBondDataHost
    _array_cls = HOOMDArray


class AngleLocalAccessCPU(AngleLocalAccessBase):
    """Access angle data on the GPU."""
    _cpp_cls = _hoomd.LocalAngleDataHost
    _array_cls = HOOMDArray


class DihedralLocalAccessCPU(DihedralLocalAccessBase):
    """Access dihedral data on the GPU."""
    _cpp_cls = _hoomd.LocalDihedralDataHost
    _array_cls = HOOMDArray


class ImproperLocalAccessCPU(ImproperLocalAccessBase):
    """Access improper data on the GPU."""
    _cpp_cls = _hoomd.LocalImproperDataHost
    _array_cls = HOOMDArray


class ConstraintLocalAccessCPU(ConstraintLocalAccessBase):
    """Access constraint data on the GPU."""
    _cpp_cls = _hoomd.LocalConstraintDataHost
    _array_cls = HOOMDArray


class PairLocalAccessCPU(PairLocalAccessBase):
    """Access special pair data on the GPU."""
    _cpp_cls = _hoomd.LocalPairDataHost
    _array_cls = HOOMDArray


class LocalSnapshot(_LocalSnapshot):
    """Provides context manager access to HOOMD-blue CPU data buffers.

    The interface of a `LocalSnapshot` is similar to that of the
    `hoomd.Snapshot`. Data is MPI rank local so for MPI parallel simulations
    only the data possessed by a rank is exposed. This means that users must
    handle the domain decomposition directly. One consequence of this is that
    access to ghost particle data is provided. A ghost particle is a particle
    that is not owned by a rank, but nevertheless is required for operations
    that use particle neighbors. Also, changing the global or local box within a
    `LocalSnapshot` context manager is not allowed.

    For every property (e.g. ``data.particles.position``), only grabs the
    data for the regular (non-ghost) particles. The property can be prefixed
    with ``ghost_`` to grab the ghost particles in a read only manner. Likewise,
    suffixing with ``_with_ghost`` will grab all data on the rank (regular and
    ghost particles) in a read only array.

    All array-like properties return a `hoomd.data.array.HOOMDArray` object
    which prevents invalid memory accesses.

    Note:
        For the ``LocalAccess`` classes the affixed attributes mentioned above
        are not shown. Also of interest, ghost data always come immediately
        after the regular data.

    See Also:
        Access the local snapshot of a state via
        `hoomd.State.cpu_local_snapshot`.
    """

    def __init__(self, state):
        super().__init__(state)
        self._particles = ParticleLocalAccessCPU(state)
        self._bonds = BondLocalAccessCPU(state)
        self._angles = AngleLocalAccessCPU(state)
        self._dihedrals = DihedralLocalAccessCPU(state)
        self._impropers = ImproperLocalAccessCPU(state)
        self._pairs = PairLocalAccessCPU(state)
        self._constraints = ConstraintLocalAccessCPU(state)
