# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.wall import _WallsMetaList
from hoomd.data.parameterdicts import ParameterDict
from hoomd.hpmc.external.field import ExternalField


def _to_hpmc_cpp_wall(wall):
    if isinstance(wall, hoomd.wall.Sphere):
        return hoomd.hpmc._hpmc.SphereWall(wall.radius, wall.origin.to_base(),
                                           wall.inside)
    if isinstance(wall, hoomd.wall.Cylinder):
        return hoomd.hpmc._hpmc.CylinderWall(wall.radius, wall.origin.to_base(),
                                             wall.axis.to_base(), wall.inside)
    if isinstance(wall, hoomd.wall.Plane):
        return hoomd.hpmc._hpmc.PlaneWall(wall.origin.to_base(),
                                          wall.normal.to_base())
    raise TypeError(f"Unknown wall type encountered {type(wall)}.")


class WallPotential(ExternalField):
    """HPMC wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            that confine particles to specific regions of space.

    This class is used to add walls to HPMC simulations by acting as a container
    for a collection of `hoomd.wall.WallGeometry` objects.  In HPMC, walls break
    space into forbidden and allowed regions, controlled by the ``inside``
    argument for spherical and cylindrical walls and the ``normal`` arguement
    for planar walls. See `hoomd.wall.WallGeometry` for more details on the
    different wall geometries available.

    To use walls in HPMC, first make a list of `hoomd.wall.WallGeometry`
    objects, and add them to a `hoomd.hpmc.external.wall.WallPotential` object.
    The ``external_potential`` attribute of the MC integrator can then be set to
    the `hoomd.hpmc.external.wall.WallPotential` object.

    Example::

        # assume mc defined as a hoomd.hpmc.integrate.HPMCIntegrator object
        walls = [hoomd.wall.Sphere(radius=4.0)]
        wall_potential = hoomd.hpmc.external.wall.WallPotential(walls)
        # add WallPotential to the integrator
        mc.external_potential = wall_potential

    """

    def __init__(self, walls):
        self._walls = None
        self._walls = hoomd.wall._WallsMetaList(walls, _to_hpmc_cpp_wall)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError('HPMC walls are not supported on the GPU.')
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, hoomd.hpmc.integrate.HPMCIntegrator):
            raise RuntimeError('Walls require a valid HPMC integrator.')

        cpp_cls_name = "Wall"
        cpp_cls_name += integrator.__class__.__name__
        cpp_cls = getattr(hoomd.hpmc._hpmc, cpp_cls_name)

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                integrator._cpp_obj)
        self._walls._sync({
            hoomd.wall.Sphere: self._cpp_obj.SphereWalls,
            hoomd.wall.Cylinder: self._cpp_obj.CylinderWalls,
            hoomd.wall.Plane: self._cpp_obj.PlaneWalls,
        })
        super()._attach()

    @property
    def walls(self):
        """`list` [`hoomd.wall.WallGeometry`]: \
            The walls associated with this wall potential."""
        return self._walls

    @walls.setter
    def walls(self, wall_list):
        if self._walls is wall_list:
            return
        self._walls = hoomd.wall._WallsMetaList(wall_list, _to_hpmc_cpp_wall)
