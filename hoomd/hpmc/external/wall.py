# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.


class wall(_External):  # noqa: name will change in v3
    R"""Manage walls (an external field type).  # noqa

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
        _External.__init__(self)
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
        R"""Count the overlaps associated with the walls.  # noqa

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
        R"""Add a spherical wall to the simulation.  # noqa

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
        R"""Change the parameters associated with a particular sphere wall.  # noqa

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
        R"""Access a parameter associated with a particular sphere wall.  # noqa

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
                "compute.wall.get_sphere_wall_param: Parameter type is not \
                        valid. Choose from rsq, origin, inside.")
            raise RuntimeError("Error: compute.wall")

    def remove_sphere_wall(self, index):
        R"""Remove a particular sphere wall from the simulation.  # noqa

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
        R"""Get the current number of sphere walls in the simulation.  # noqa

        Returns: the current number of sphere walls in the simulation

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            num_sph_walls = ext_wall.get_num_sphere_walls();

        """
        return self.cpp_compute.getNumSphereWalls()

    def add_cylinder_wall(self, radius, origin, orientation, inside=True):
        R"""Add a cylindrical wall to the simulation.  # noqa

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
        R"""Change the parameters associated with a particular cylinder wall.  # noqa

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
        R"""Access a parameter associated with a particular cylinder wall.  # noqa

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
                "compute.wall.get_cylinder_wall_param: Parameter type is not \
                        valid. Choose from rsq, origin, orientation, inside.")
            raise RuntimeError("Error: compute.wall")

    def remove_cylinder_wall(self, index):
        R"""Remove a particular cylinder wall from the simulation.  # noqa

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
        R"""Get the current number of cylinder walls in the simulation.  # noqa

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
        R"""Add a plane wall to the simulation.  # noqa

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
        R"""Change the parameters associated with a particular plane wall.  # noqa

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
        R"""Access a parameter associated with a particular plane wall.  # noqa

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
                "compute.wall.get_plane_wall_param: Parameter type is not \
                        valid. Choose from normal, origin.")
            raise RuntimeError("Error: compute.wall")

    def remove_plane_wall(self, index):
        R"""Remove a particular plane wall from the simulation.  # noqa

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
        R"""Get the current number of plane walls in the simulation.  # noqa

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
        R"""Set the volume associated with the intersection of all walls in the system.  # noqa

        This number will subsequently change when the box is resized and walls are scaled appropriately.

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            ext_wall.set_volume(4./3.*np.pi);

        """
        self.cpp_compute.setVolume(volume)

    def get_volume(self):
        R"""Get the current volume associated with the intersection of all walls in the system.  # noqa

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
        R"""Get the simulation box that the wall class is currently storing.  # noqa

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
        R"""Set the simulation box that the wall class is currently storing.  # noqa

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
        if all((Lx is None, Ly is None, Lz is None, xy is None, xz is None,
                yz is None)):
            hoomd.context.current.device.cpp_msg.warning(
                "compute.wall.set_curr_box: Ignoring request to set the wall's \
                        box without parameters\n")
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
