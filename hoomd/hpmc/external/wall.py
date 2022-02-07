# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.wall import _WallsMetaList
from hoomd.data.parameterdicts import ParameterDict
from hoomd.hpmc.external.field import ExternalField
from hoomd.logging import log


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
    r"""HPMC wall potential.

    Args:
        walls (`list` [`hoomd.wall.WallGeometry` ]): A list of wall definitions
            that confine particles to specific regions of space.

    `WallPotential` adds hard walls to HPMC simulations. Define the wall
    geometry with a collection of `wall.WallGeometry` objects.  These walls
    break space into forbidden and allowed regions, controlled by the ``inside``
    argument for spherical and cylindrical walls and the ``normal`` argument for
    planar walls. HPMC rejects trial moves that cause any part of the particle's
    shape to enter the the space defined by the points with a negative signed
    distance from any wall in the collection. Formally, the contribution of the
    particle-wall interactions to the potential energy of the system is given by

    .. math::

        U_{\mathrm{walls}} = \sum_{i=0}^{N_{\mathrm{particles}}}
        \sum_{j=0}^{N_{\mathrm{walls}}} U_{i,j},


    where the energy of interaction :math:`U_{i,j}` between particle :math:`i`
    and wall :math:`j` is given by

    .. math::
        :nowrap:

        \begin{eqnarray*}
        U_{i,j} &=& \infty \,\,\,\text{if } \,\, d_{i,j} <= 0 \\
                &=& 0; \,\,\, \text{if } \,\, d_{i,j} > 0,
        \end{eqnarray*}


    where :math:`d_{i,j} = \min{\{(\vec{r}_i - \vec{r}_j) \cdot \vec{n}_j :
    \vec{r}_i \in V_I, \vec{r}_j \in W_J\}}` is the minimum signed distance
    between all pairs of points :math:`\vec{r}_i` on the body of the particle
    :math:`V_I` and :math:`\vec{r}_j` on the surface of the wall :math:`W_J` and
    :math:`\vec{n}_j` is the vector normal to the surface of the wall at
    :math:`\vec{r}_j` pointing to the allowed region of space defined by the
    wall.


    Walls are enforced by the HPMC integrator. Assign a `WallPotential` instance
    to the `hpmc.integrate.HPMCIntegrator.external_potential` to activate the
    wall potential. Not all combinations of HPMC integrators and wall geometries
    have overlap checks implemented, and a `NotImplementedError` is raised if a
    wall geometry is attached to a simulation with a specific HPMC integrator
    attached and the overlap checks between the specific shape and wall geometry
    are not implemented. See the individual subclasses of
    `hoomd.hpmc.integrate.HPMCIntegrator` for their wall support.

    See Also:
        `hoomd.wall`

    Example::

        mc = hoomd.hpmc.integrate.Sphere()
        walls = [hoomd.wall.Sphere(radius=4.0)]
        wall_potential = hoomd.hpmc.external.wall.WallPotential(walls)
        mc.external_potential = wall_potential

    """

    def __init__(self, walls):
        self._walls = hoomd.wall._WallsMetaList(walls, _to_hpmc_cpp_wall)

    def _check_valid_shape_wall_pairs(self, walls, integrator):
        supported_shape_wall_overlap_checks = {
            hoomd.hpmc.integrate.Sphere: [
                hoomd.wall.Sphere, hoomd.wall.Cylinder, hoomd.wall.Plane
            ],
            hoomd.hpmc.integrate.ConvexPolyhedron: [
                hoomd.wall.Sphere, hoomd.wall.Cylinder, hoomd.wall.Plane
            ],
            hoomd.hpmc.integrate.ConvexSpheropolyhedron: [
                hoomd.wall.Sphere, hoomd.wall.Plane
            ]
        }
        integrator_type = type(integrator)
        for wt in (type(w) for w in walls):
            if wt not in supported_shape_wall_overlap_checks.get(
                    integrator_type, []):
                msg = f'Overlap checks between {wt} and {integrator_type} are '
                msg += 'not supported.'
                raise NotImplementedError(msg)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError('HPMC walls are not supported on the GPU.')
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, hoomd.hpmc.integrate.HPMCIntegrator):
            raise RuntimeError('Walls require a valid HPMC integrator.')

        self._check_valid_shape_wall_pairs(self._walls, integrator)

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
            The wall geometries associated with this potential."""
        return self._walls

    @walls.setter
    def walls(self, wall_list):
        if self._walls is wall_list:
            return
        if self._attached:
            integrator = self._simulation.operations.integrator
            self._check_valid_shape_wall_pairs(wall_list, integrator)
        self._walls = hoomd.wall._WallsMetaList(wall_list, _to_hpmc_cpp_wall)
        self._walls._sync({
            hoomd.wall.Sphere: self._cpp_obj.SphereWalls,
            hoomd.wall.Cylinder: self._cpp_obj.CylinderWalls,
            hoomd.wall.Plane: self._cpp_obj.PlaneWalls,
        })

    @log(requires_run=True)
    def overlaps(self):
        """int: The total number of overlaps between particles and walls."""
        timestep = self._simulation.timestep
        return self._cpp_obj.numOverlaps(timestep, False)
