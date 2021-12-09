# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Apply external fields to HPMC simulations.
"""

from hoomd import _hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.logging import log
from hoomd.operation import Compute
from hoomd.operation import _HOOMDBaseObject
import hoomd
import numpy as np


## \internal
# \brief Base class for external fields
#
# An external in hoomd reflects an ExternalField in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ external itself is tracked optionally passed
# to the hpmc integrator. While external fields are Compute types and are added
# to the System they will not be enforced unless they are added to the integrator.
# Only one external field can be held by the integrator so if multiple fields are
# required use the external_field_composite class to manage them.
class _external(Compute):
    ## \internal
    # \brief Initialize an empty external.
    #
    # \post nothing is done here yet.
    def __init__(self):
        _compute.__init__(self)
        self.cpp_compute = None
        # nothing else to do.


class ExternalField(_HOOMDBaseObject):
    """Base class external field.

    Provides common methods for all subclasses.

    Note:
        Users should use the subclasses and not instantiate `ExternalField`
        directly.
    """


class LatticeField(ExternalField):
    R"""Restrain particles on a lattice

    Args:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of
            `float`): the reference positions of the
            lattice :math:`[\mathrm{length}]`.
        reference_orientations ((*N_particles*, 4) `numpy.ndarray` of
            `float`): the reference orientations of the lattice.
        k_translational (`float`): translational spring constant.
        k_rotational (`float`): rotational spring constant.
        symmetries ((*N_sym*, 4) `numpy.ndarray` of
            `float`): the orientations that are equivalent through symmetry,
            i.e., the rotation quaternions that leave the particles unchanged.

    :py:class:`LatticeField` specifies that a harmonic spring is added to every
    particle:

    .. math::

        V_{i}(r_i)  = k_{trans} \cdot (r_i-r_{0,i})^2 \\
        V_{i}(q_i)  = k_{rot} \cdot (q_i-q_{0,i})^2

    where :math:`k_{trans}` and :math:`k_{rot}` correspond to the arguments
    ``k_translational`` and ``k_rotational``, respectively, :math:`r_i` and
    :math:`q_i` are the position and orientation of particle :math:`i`, and the
    :math:`0` subscripts lattice quantities.

    Note:
        The factor of 1/2 is not included in the formulas; specify your spring
        constants accordingly.

    """

    def __init__(self, reference_positions, reference_orientations,
                 k_translational, k_rotational, symmetries):
        param_dict = ParameterDict(
            reference_positions=NDArrayValidator(dtype=np.float32,
                                                 shape=(None, 3)),
            reference_orientations=NDArrayValidator(dtype=np.float32,
                                                    shape=(None, 4)),
            k_translational=float,
            k_rotational=float,
            symmetries=NDArrayValidator(dtype=np.float32, shape=(None, 4)),
        )
        param_dict['k_translational'] = k_translational
        param_dict['k_rotational'] = k_rotational
        param_dict['reference_positions'] = position
        param_dict['reference_orientations'] = orientation
        param_dict['symmetries'] = symmetries
        self._param_dict.update(param_dict)

    def _attach(self):
        cls = None

        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        # get device and system definition from simulation
        device = self._simulation.device
        cpp_sys_def = self._simulation.state._cpp_sys_def

        if not isinstance(device, hoomd.device.GPU):
            if isinstance(integrator, integrate.Sphere):
                cls = _hpmc.ExternalFieldLatticeSphere
            elif isinstance(integrator, integrate.ConvexPolygon):
                cls = _hpmc.ExternalFieldLatticeConvexPolygon
            elif isinstance(integrator, integrate.SimplePolygon):
                cls = _hpmc.ExternalFieldLatticeSimplePolygon
            elif isinstance(integrator, integrate.ConvexPolyhedron):
                cls = _hpmc.ExternalFieldLatticeConvexPolyhedron
            elif isinstance(integrator, integrate.ConvexSpheropolyhedron):
                cls = _hpmc.ExternalFieldLatticeSpheropolyhedron
            elif isinstance(integrator, integrate.Ellipsoid):
                cls = _hpmc.ExternalFieldLatticeEllipsoid
            elif isinstance(integrator, integrate.ConvexSpheropolygon):
                cls = _hpmc.ExternalFieldLatticeSpheropolygon
            elif isinstance(integrator, integrate.FacetedEllipsoid):
                cls = _hpmc.ExternalFieldLatticeFacetedEllipsoid
            elif isinstance(integrator, integrate.Polyhedron):
                cls = _hpmc.ExternalFieldLatticePolyhedron
            elif isinstance(integrator, integrate.Sphinx):
                cls = _hpmc.ExternalFieldLatticeSphinx
            elif isinstance(integrator, integrate.SphereUnion):
                cls = _hpmc.ExternalFieldLatticeSphereUnion
            elif isinstance(integrator, integrate.FacetedEllipsoidUnion):
                cls = _hpmc.ExternalFieldlatticeFacetedEllipsoidUnion
            elif isinstance(integrator, integrate.ConvexSpheropolyhedronUnion):
                cls = _hpmc.ExternalFieldLatticeConvexPolyhedronUnion
            else:
                msg = 'Error initializing hoomd.hpmc.field.LatticeField: '
                msg += 'unsupported integrator'
                raise RuntimeError(msg)
        else:
            msg = 'Error initializing hoomd.hpmc.field.LatticeField: '
            msg += 'GPU not supported.'
            raise RuntimeError(msg)

        self._cpp_obj = cls(
            cpp_sys_def,
            self.reference_positions,
            self.k_translational,
            self.reference_orientations,
            self.k_rotational,
            self.symmetries,
        )
        super()._attach()

    @log(requires_run=True)
    def energy(self):
        """float: The energy of the lattice field :math:`[\\mathrm{energy}]`.

        """
        timestep = self._simulation.timestep
        return self._cpp_obj.getEnergy(timestep)


class external_field_composite(_external):
    R""" Manage multiple external fields.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator (don't specify a new integrator later, external_field_composite will continue to use the old one)
        fields (list): List of external fields to combine together.

    :py:class:`external_field_composite` allows the user to create and compute multiple
    external fields. Once created use :py:meth:`add_field` to add a new field.

    Once initialized, the compute provides a log quantities that other external
    fields create. See those external fields to find the quantities.

    Examples::

        mc = hpmc.integrate.shape(...);
        walls = hpmc.field.walls(...)
        lattice = hpmc.field.lattice(...)
        composite_field = hpmc.field.external_field_composite(mc, fields=[walls, lattice])

    """

    def __init__(self, mc, fields=None):
        _external.__init__(self)
        cls = None
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ExternalFieldCompositeSphere
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ExternalFieldCompositeConvexPolygon
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ExternalFieldCompositeSimplePolygon
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.ExternalFieldCompositeConvexPolyhedron
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.ExternalFieldCompositeSpheropolyhedron
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ExternalFieldCompositeEllipsoid
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls = _hpmc.ExternalFieldCompositeSpheropolygon
            elif isinstance(mc, integrate.faceted_ellipsoid):
                cls = _hpmc.ExternalFieldCompositeFacetedEllipsoid
            elif isinstance(mc, integrate.polyhedron):
                cls = _hpmc.ExternalFieldCompositePolyhedron
            elif isinstance(mc, integrate.sphinx):
                cls = _hpmc.ExternalFieldCompositeSphinx
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.ExternalFieldCompositeSphereUnion
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.ExternalFieldCompositeFacetedEllipsoidUnion
            elif isinstance(mc, integrate.convex_spheropolyhedron_union):
                cls = _hpmc.ExternalFieldCompositeConvexPolyhedronUnion
            else:
                hoomd.context.current.device.cpp_msg.error(
                    "compute.position_lattice_field: Unsupported integrator.\n")
                raise RuntimeError(
                    "Error initializing compute.position_lattice_field")
        else:
            hoomd.context.current.device.cpp_msg.error("GPU not supported yet")
            raise RuntimeError(
                "Error initializing compute.position_lattice_field")

        self.compute_name = "composite_field"
        self.cpp_compute = cls(hoomd.context.current.system_definition)
        hoomd.context.current.system.addCompute(self.cpp_compute,
                                                self.compute_name)

        mc.set_external(self)

        if not fields is None:
            self.add_field(fields=fields)

    def add_field(self, fields):
        R""" Add an external field to the ensemble.

        Args:
            fields (list): list of fields to add

        Example::

            mc = hpmc.integrate.shape(...);
            composite_field = hpmc.compute.external_field_composite(mc)
            walls = hpmc.compute.walls(..., setup=False)
            lattice = hpmc.compute.lattice(..., setup=False)
            composite_field.add_field(fields=[walls, lattice])

        """
        if not type(fields) == list:
            fields = list(fields)
        for field in fields:
            self.cpp_compute.addExternal(field.cpp_compute)


class wall(_external):
    R""" Manage walls (an external field type).

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`):MC integrator.
        composite (bool): Set this to True when this field is part of a :py:class:`external_field_composite`.

    :py:class:`wall` allows the user to implement one or more walls. If multiple walls are added, then particles are
    confined by the INTERSECTION of all of these walls. In other words, particles are confined by all walls if they
    independently satisfy the confinement condition associated with each separate wall.
    Once you've created an instance of this class, use :py:meth:`add_sphere_wall`
    to add a new spherical wall, :py:meth:`add_cylinder_wall` to add a new cylindrical wall, or
    :py:meth:`add_plane_wall` to add a new plane wall.

    Specialized overlap checks have been written for supported combinations of wall types and particle shapes.
    These combinations are:
    * Sphere particles: sphere walls, cylinder walls, plane walls
    * Convex polyhedron particles: sphere walls, cylinder walls, plane walls
    * Convex spheropolyhedron particles: sphere walls

    Once initialized, the compute provides the following log quantities that can be logged via ``hoomd.analyze.log``:

    * **hpmc_wall_volume** : the volume associated with the intersection of implemented walls. This number is only meaningful
      if the user has initially provided it through :py:meth:`set_volume`. It will subsequently change when
      the box is resized and walls are scaled appropriately.
    * **hpmc_wall_sph_rsq-i** : the squared radius of the spherical wall indexed by i, beginning at 0 in the order the sphere
      walls were added to the system.
    * **hpmc_wall_cyl_rsq-i** : the squared radius of the cylindrical wall indexed by i, beginning at 0 in the order the
      cylinder walls were added to the system.

    Example::

        mc = hpmc.integrate.sphere(seed = 415236);
        ext_wall = hpmc.compute.wall(mc);
        ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
        ext_wall.set_volume(4./3.*np.pi);
        log = analyze.log(quantities=['hpmc_wall_volume','hpmc_wall_sph_rsq-0'], period=100, filename='log.dat', overwrite=True);

    """

    index = 0

    def __init__(self, mc, composite=False):
        _external.__init__(self)
        # create the c++ mirror class
        cls = None
        self.compute_name = "wall-" + str(wall.index)
        wall.index += 1
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.WallSphere
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.WallConvexPolyhedron
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.WallSpheropolyhedron
            else:
                hoomd.context.current.device.cpp_msg.error(
                    "compute.wall: Unsupported integrator.\n")
                raise RuntimeError("Error initializing compute.wall")
        else:
            hoomd.context.current.device.cpp_msg.error("GPU not supported yet")
            raise RuntimeError("Error initializing compute.wall")

        self.cpp_compute = cls(hoomd.context.current.system_definition,
                               mc.cpp_integrator)
        hoomd.context.current.system.addCompute(self.cpp_compute,
                                                self.compute_name)

        if not composite:
            mc.set_external(self)

    def count_overlaps(self, exit_early=False):
        R""" Count the overlaps associated with the walls.

        Args:
            exit_early (bool): When True, stop counting overlaps after the first one is found.

        Returns:
            The number of overlaps associated with the walls

        A particle "overlaps" with a wall if it fails to meet the confinement condition associated with the wall.

        Example:

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            run(100)
            num_overlaps = ext_wall.count_overlaps();

        """
        return self.cpp_compute.countOverlaps(
            hoomd.context.current.system.getCurrentTimeStep(), exit_early)

    def add_sphere_wall(self, radius, origin, inside=True):
        R""" Add a spherical wall to the simulation.

        Args:
            radius (float): radius of spherical wall
            origin (tuple): origin (center) of spherical wall.
            inside (bool): When True, particles are CONFINED by the wall if they exist entirely inside the sphere (in the portion of connected space that contains the origin).
                           When False, then particles are CONFINED by the wall if they exist entirely outside the sphere (in the portion of connected space that does not contain the origin).

        Quick Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);

        """
        self.cpp_compute.AddSphereWall(
            _hpmc.make_sphere_wall(radius, origin, inside))

    def set_sphere_wall(self, index, radius, origin, inside=True):
        R""" Change the parameters associated with a particular sphere wall.

        Args:
            index (int): index of the sphere wall to be modified. indices begin at 0 in the order the sphere walls were added to the system.
            radius (float): New radius of spherical wall
            origin (tuple): New origin (center) of spherical wall.
            inside (bool): New confinement condition. When True, particles are CONFINED by the wall if they exist entirely inside the sphere (in the portion of connected space that contains the origin).
                           When False, then particles are CONFINED by the wall if they exist entirely outside the sphere (in the portion of connected space that does not contain the origin).

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_sphere_wall(index = 0, radius = 3.0, origin = [0, 0, 0], inside = True);

        """
        self.cpp_compute.SetSphereWallParameter(
            index, _hpmc.make_sphere_wall(radius, origin, inside))

    def get_sphere_wall_param(self, index, param):
        R""" Access a parameter associated with a particular sphere wall.

        Args:
            index (int): index of the sphere wall to be accessed. indices begin at 0 in the order the sphere walls were added to the system.
            param (str): name of parameter to be accessed. options are "rsq" (squared radius of sphere wall), "origin" (origin of sphere wall), and "inside" (confinement condition associated with sphere wall)

        Returns:
            Value of queried parameter.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            rsq = ext_wall.get_sphere_wall_param(index = 0, param = "rsq");

        """
        t = self.cpp_compute.GetSphereWallParametersPy(index)
        if param == "rsq":
            return t[0]
        elif param == "origin":
            return t[1]
        elif param == "inside":
            return t[2]
        else:
            hoomd.context.current.device.cpp_msg.error(
                "compute.wall.get_sphere_wall_param: Parameter type is not valid. Choose from rsq, origin, inside."
            )
            raise RuntimeError("Error: compute.wall")

    def remove_sphere_wall(self, index):
        R""" Remove a particular sphere wall from the simulation.

        Args:
            index (int): index of the sphere wall to be removed. indices begin at 0 in the order the sphere walls were added to the system.

        Quick Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.remove_sphere_wall(index = 0);

        """
        self.cpp_compute.RemoveSphereWall(index)

    def get_num_sphere_walls(self):
        R""" Get the current number of sphere walls in the simulation.

        Returns: the current number of sphere walls in the simulation

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            num_sph_walls = ext_wall.get_num_sphere_walls();

        """
        return self.cpp_compute.getNumSphereWalls()

    def add_cylinder_wall(self, radius, origin, orientation, inside=True):
        R""" Add a cylindrical wall to the simulation.

        Args:
            radius (float): radius of cylindrical wall
            origin (tuple): origin (center) of cylindrical wall
            orientation (tuple): vector that defines the direction of the long axis of the cylinder. will be normalized automatically by hpmc.
            inside (bool): When True, then particles are CONFINED by the wall if they exist entirely inside the cylinder (in the portion of connected space that contains the origin).
                           When False, then particles are CONFINED by the wall if they exist entirely outside the cylinder (in the portion of connected space that does not contain the origin). DEFAULTS to True.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);

        """

        param = _hpmc.make_cylinder_wall(radius, origin, orientation, inside)
        self.cpp_compute.AddCylinderWall(param)

    def set_cylinder_wall(self,
                          index,
                          radius,
                          origin,
                          orientation,
                          inside=True):
        R""" Change the parameters associated with a particular cylinder wall.

        Args:
            index (int): index of the cylinder wall to be modified. indices begin at 0 in the order the cylinder walls were added to the system.
            radius (float): New radius of cylindrical wall
            origin (tuple): New origin (center) of cylindrical wall
            orientation (tuple): New vector that defines the direction of the long axis of the cylinder. will be normalized automatically by hpmc.
            inside (bool): New confinement condition. When True, then particles are CONFINED by the wall if they exist entirely inside the cylinder (in the portion of connected space that contains the origin).
                           When False, then particles are CONFINED by the wall if they exist entirely outside the cylinder (in the portion of connected space that does not contain the origin). DEFAULTS to True.


        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
            ext_wall.set_cylinder_wall(index = 0, radius = 3.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);

        """
        param = _hpmc.make_cylinder_wall(radius, origin, orientation, inside)
        self.cpp_compute.SetCylinderWallParameter(index, param)

    def get_cylinder_wall_param(self, index, param):
        R""" Access a parameter associated with a particular cylinder wall.

        Args:
            index (int): index of the cylinder wall to be accessed. indices begin at 0 in the order the cylinder walls were added to the system.
            param (str): name of parameter to be accessed. options are "rsq" (squared radius of cylinder wall), "origin" (origin of cylinder wall), "orientation" (orientation of cylinder wall),
                         and "inside" (confinement condition associated with cylinder wall).

        Returns:
            Value of queried parameter.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
            rsq = ext_wall.get_cylinder_wall_param(index = 0, param = "rsq");

        """
        t = self.cpp_compute.GetCylinderWallParametersPy(index)
        if param == "rsq":
            return t[0]
        elif param == "origin":
            return t[1]
        elif param == "orientation":
            return t[2]
        elif param == "inside":
            return t[3]
        else:
            hoomd.context.current.device.cpp_msg.error(
                "compute.wall.get_cylinder_wall_param: Parameter type is not valid. Choose from rsq, origin, orientation, inside."
            )
            raise RuntimeError("Error: compute.wall")

    def remove_cylinder_wall(self, index):
        R""" Remove a particular cylinder wall from the simulation.

        Args:
            index (int): index of the cylinder wall to be removed. indices begin at 0 in the order the cylinder walls were added to the system.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
            ext_wall.remove_cylinder_wall(index = 0);

        """
        self.cpp_compute.RemoveCylinderWall(index)

    def get_num_cylinder_walls(self):
        R""" Get the current number of cylinder walls in the simulation.

        Returns:
            The current number of cylinder walls in the simulation.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
            num_cyl_walls = ext_wall.get_num_cylinder_walls();

        """
        return self.cpp_compute.getNumCylinderWalls()

    def add_plane_wall(self, normal, origin):
        R""" Add a plane wall to the simulation.

        Args:
            normal (tuple): vector normal to the plane. this, in combination with a point on the plane, defines the plane entirely. It will be normalized automatically by hpmc.
                            The direction of the normal vector defines the confinement condition associated with the plane wall. If every part of a particle exists in the halfspace into which the normal points, then that particle is CONFINED by the plane wall.
            origin (tuple): a point on the plane wall. this, in combination with the normal vector, defines the plane entirely.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);

        """
        self.cpp_compute.AddPlaneWall(
            _hpmc.make_plane_wall(normal, origin, True))

    def set_plane_wall(self, index, normal, origin):
        R""" Change the parameters associated with a particular plane wall.

        Args:
            index (int): index of the plane wall to be modified. indices begin at 0 in the order the plane walls were added to the system.
            normal (tuple): new vector normal to the plane. this, in combination with a point on the plane, defines the plane entirely. It will be normalized automatically by hpmc.
                            The direction of the normal vector defines the confinement condition associated with the plane wall. If every part of a particle exists in the halfspace into which the normal points, then that particle is CONFINED by the plane wall.
            origin (tuple): new point on the plane wall. this, in combination with the normal vector, defines the plane entirely.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
            ext_wall.set_plane_wall(index = 0, normal = [0, 0, 1], origin = [0, 0, 1]);

        """
        self.cpp_compute.SetPlaneWallParameter(
            index, _hpmc.make_plane_wall(normal, origin, True))

    def get_plane_wall_param(self, index, param):
        R""" Access a parameter associated with a particular plane wall.

        Args:
            index (int): index of the plane wall to be accessed. indices begin at 0 in the order the plane walls were added to the system.
            param (str): name of parameter to be accessed. options are "normal" (vector normal to the plane wall), and "origin" (point on the plane wall)

        Returns:
            Value of queried parameter.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
            n = ext_wall.get_plane_wall_param(index = 0, param = "normal");

        """
        t = self.cpp_compute.GetPlaneWallParametersPy(index)
        if param == "normal":
            return t[0]
        elif param == "origin":
            return t[1]
        else:
            hoomd.context.current.device.cpp_msg.error(
                "compute.wall.get_plane_wall_param: Parameter type is not valid. Choose from normal, origin."
            )
            raise RuntimeError("Error: compute.wall")

    def remove_plane_wall(self, index):
        R""" Remove a particular plane wall from the simulation.

        Args:
            index (int): index of the plane wall to be removed. indices begin at 0 in the order the plane walls were added to the system.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
            ext_wall.remove_plane_wall(index = 0);

        """
        self.cpp_compute.RemovePlaneWall(index)

    def get_num_plane_walls(self):
        R""" Get the current number of plane walls in the simulation.

        Returns:
            The current number of plane walls in the simulation.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
            num_plane_walls = ext_wall.get_num_plane_walls();

        """
        return self.cpp_compute.getNumPlaneWalls()

    def set_volume(self, volume):
        R""" Set the volume associated with the intersection of all walls in the system.

        This number will subsequently change when the box is resized and walls are scaled appropriately.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_volume(4./3.*np.pi);

        """
        self.cpp_compute.setVolume(volume)

    def get_volume(self):
        R""" Get the current volume associated with the intersection of all walls in the system.

        If this quantity has not previously been set by the user, this returns a meaningless value.

        Returns:
            The current volume associated with the intersection of all walls in the system.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_volume(4./3.*np.pi);
            run(100)
            curr_vol = ext_wall.get_volume();

        """
        return self.cpp_compute.getVolume()

    def get_curr_box(self):
        R""" Get the simulation box that the wall class is currently storing.

        Returns:
            The boxdim object that the wall class is currently storing.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_volume(4./3.*np.pi);
            run(100)
            curr_box = ext_wall.get_curr_box();

        """
        return hoomd.data.boxdim(Lx=self.cpp_compute.GetCurrBoxLx(),
                                 Ly=self.cpp_compute.GetCurrBoxLy(),
                                 Lz=self.cpp_compute.GetCurrBoxLz(),
                                 xy=self.cpp_compute.GetCurrBoxTiltFactorXY(),
                                 xz=self.cpp_compute.GetCurrBoxTiltFactorXZ(),
                                 yz=self.cpp_compute.GetCurrBoxTiltFactorYZ())

    def set_curr_box(self,
                     Lx=None,
                     Ly=None,
                     Lz=None,
                     xy=None,
                     xz=None,
                     yz=None):
        R""" Set the simulation box that the wall class is currently storing.

        You may want to set this independently so that you can cleverly control whether or not the walls actually scale in case you manually resize your simulation box.
        The walls scale automatically when they get the signal that the global box, associated with the system definition, has scaled. They do so, however, with a scale factor associated with
        the ratio of the volume of the global box to the volume of the box that the walls class is currently storing. (After the scaling the box that the walls class is currently storing is updated appropriately.)
        If you want to change the simulation box WITHOUT scaling the walls, then, you must first update the simulation box that the walls class is storing, THEN update the global box associated with the system definition.

        Example::

            init_box = hoomd.data.boxdim(L=10, dimensions=3);
            snap = hoomd.data.make_snapshot(N=1, box=init_box, particle_types=['A']);
            system = hoomd.init.read_snapshot(snap);
            system.particles[0].position = [0,0,0];
            system.particles[0].type = 'A';
            mc = hpmc.integrate.sphere(seed = 415236);
            mc.shape_param.set('A', diameter = 2.0);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 3.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_curr_box(Lx=2.0*init_box.Lx, Ly=2.0*init_box.Ly, Lz=2.0*init_box.Lz, xy=init_box.xy, xz=init_box.xz, yz=init_box.yz);
            system.sysdef.getParticleData().setGlobalBox(ext_wall.get_curr_box()._getBoxDim())

        """
        # much of this is from hoomd's update.py box_resize class
        if Lx is None and Ly is None and Lz is None and xy is None and xz is None and yz is None:
            hoomd.context.current.device.cpp_msg.warning(
                "compute.wall.set_curr_box: Ignoring request to set the wall's box without parameters\n"
            )
            return

        # setup arguments
        if Lx is None:
            Lx = self.cpp_compute.GetCurrBoxLx()
        if Ly is None:
            Ly = self.cpp_compute.GetCurrBoxLy()
        if Lz is None:
            Lz = self.cpp_compute.GetCurrBoxLz()

        if xy is None:
            xy = self.cpp_compute.GetCurrBoxTiltFactorXY()
        if xz is None:
            xz = self.cpp_compute.GetCurrBoxTiltFactorXZ()
        if yz is None:
            yz = self.cpp_compute.GetCurrBoxTiltFactorYZ()

        self.cpp_compute.SetCurrBox(Lx, Ly, Lz, xy, xz, yz)


class frenkel_ladd_energy(Compute):
    R""" Compute the Frenkel-Ladd Energy of a crystal.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        ln_gamma (float): log of the translational spring constant
        q_factor (float): scale factor between the translational spring constant and rotational spring constant
        r0 (list): reference lattice positions
        q0 (list): reference lattice orientations
        drift_period (int): period call the remove drift updater
        symmetry (list): list of equivalent quaternions for the shape.

    :py:class:`frenkel_ladd_energy` interacts with :py:class:`.lattice_field`
    and :py:class:`hoomd.hpmc.update.remove_drift`.

    Once initialized, the compute provides the log quantities from the :py:class:`lattice_field`.

    .. warning::
        The lattice energies and standard deviations logged by
        :py:class:`lattice_field` are multiplied by the spring constant. As a result,
        when computing the free energies from :py:class:`frenkel_ladd_energy` class,
        instead of integrating the free energy over the spring constants, you should
        integrate over the natural log of the spring constants.

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=seed);
        mc.shape_param.set("A", vertices=verts)
        mc.set_params(d=0.005, a=0.005)
        #set the FL parameters
        fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)

    """

    def __init__(self,
                 mc,
                 ln_gamma,
                 q_factor,
                 r0,
                 q0,
                 drift_period,
                 symmetry=[]):
        import math
        import numpy
        # initialize base class
        _compute.__init__(self)

        if type(r0) == numpy.ndarray:
            self.lattice_positions = r0.tolist()
        else:
            self.lattice_positions = list(r0)

        if type(q0) == numpy.ndarray:
            self.lattice_orientations = q0.tolist()
        else:
            self.lattice_orientations = list(q0)

        self.mc = mc
        self.q_factor = q_factor
        self.trans_spring_const = math.exp(ln_gamma)
        self.rotat_spring_const = self.q_factor * self.trans_spring_const
        self.lattice = lattice_field(self.mc,
                                     position=self.lattice_positions,
                                     orientation=self.lattice_orientations,
                                     k=self.trans_spring_const,
                                     q=self.rotat_spring_const,
                                     symmetry=symmetry)
        self.remove_drift = hoomd.hpmc.update.remove_drift(self.mc,
                                                           self.lattice,
                                                           period=drift_period)

    def reset_statistics(self):
        R""" Reset the statistics counters.

        Example::

            mc = hpmc.integrate.sphere(seed=415236);
            fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)
            ks = np.linspace(1000, 0.01, 100);
            for k in ks:
              fl.set_params(ln_gamma=math.log(k), q_factor=10.0);
              fl.reset_statistics();
              run(1000)

        """
        self.lattice.reset(0)

    def set_params(self, ln_gamma=None, q_factor=None):
        R""" Set the Frenkel-Ladd parameters.

        Args:
            ln_gamma (float): log of the translational spring constant
            q_factor (float): scale factor between the translational spring constant and rotational spring constant

        Example::

            mc = hpmc.integrate.sphere(seed=415236);
            fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)
            ks = np.linspace(1000, 0.01, 100);
            for k in ks:
              fl.set_params(ln_gamma=math.log(k), q_factor=10.0);
              fl.reset_statistics();
              run(1000)

        """
        import math
        if not q_factor is None:
            self.q_factor = q_factor
        if not ln_gamma is None:
            self.trans_spring_const = math.exp(ln_gamma)
        self.rotat_spring_const = self.q_factor * self.trans_spring_const
        self.lattice.set_params(self.trans_spring_const,
                                self.rotat_spring_const)


class callback(_external):
    R""" Use a python-defined energy function in MC integration

    Args:

        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        callback (`callable`): A python function to evaluate the energy of a configuration
        composite (bool): True if this evaluator is part of a composite external field

    Example::

          def energy(snapshot):
              # evaluate the energy in a linear potential gradient along the x-axis
              gradient = (5,0,0)
              e = 0
              for p in snap.particles.position:
                  e -= numpy.dot(gradient,p)
              return e

          mc = hpmc.integrate.sphere(seed=415236);
          mc.shape_param.set('A',diameter=1.0)
          hpmc.field.callback(mc=mc, energy_function=energy);
          run(100)
    """

    def __init__(self, mc, energy_function, composite=False):
        _external.__init__(self)
        cls = None
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ExternalCallbackSphere
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ExternalCallbackConvexPolygon
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ExternalCallbackSimplePolygon
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.ExternalCallbackConvexPolyhedron
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.ExternalCallbackSpheropolyhedron
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ExternalCallbackEllipsoid
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls = _hpmc.ExternalCallbackSpheropolygon
            elif isinstance(mc, integrate.faceted_ellipsoid):
                cls = _hpmc.ExternalCallbackFacetedEllipsoid
            elif isinstance(mc, integrate.polyhedron):
                cls = _hpmc.ExternalCallbackPolyhedron
            elif isinstance(mc, integrate.sphinx):
                cls = _hpmc.ExternalCallbackSphinx
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.ExternalCallbackSphereUnion
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.ExternalCallbackFacetedEllipsoidUnion
            elif isinstance(mc, integrate.convex_spheropolyhedron_union):
                cls = _hpmc.ExternalCallbackConvexPolyhedronUnion
            else:
                hoomd.context.current.device.cpp_msg.error(
                    "hpmc.field.callback: Unsupported integrator.\n")
                raise RuntimeError("Error initializing python callback")
        else:
            hoomd.context.current.device.cpp_msg.error("GPU not supported")
            raise RuntimeError("Error initializing hpmc.field.callback")

        self.compute_name = "callback"
        self.cpp_compute = cls(hoomd.context.current.system_definition,
                               energy_function)
        hoomd.context.current.system.addCompute(self.cpp_compute,
                                                self.compute_name)
        if not composite:
            mc.set_external(self)
