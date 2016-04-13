## \package hpmc.compute
# \brief HPMC integration modes

from __future__ import print_function

from hoomd import _hoomd
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.compute import _compute
import hoomd

## Compute the free volume available to a test particle by stoachstic integration
#
# compute.free_volume computes the free volume of a particle assembly using stochastic integration with a test particle type.
# It works together with an HPMC integrator, which defines the particle types used in the simulation.
# As parameters it requires the number of MC integration samples (\b nsample), and the type of particle (\b test_type)
# to use for the integration.
#
# Once initialized, the compute provides a log quantity
# called **hpmc_free_volume**, that can be logged via analyze.log.
# If a suffix is specified, the log quantities name will be
# **hpmc_free_volume_suffix**.
#
class free_volume(_compute):
    ## Specifies the SDF analysis to perform
    # \param mc MC integrator (don't specify a new integrator later, sdf will continue to use the old one)
    # \param seed Random seed for MC integration (integer)
    # \param type Type of particle to use for integration
    # \param nsample Number of samples to use in MC integration
    # \param suffix Suffix to use for log quantity
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236)
    # compute.free_volume(mc=mc, seed=123, test_type='B', nsample=1000)
    # log = analyze.log(quantities=['hpmc_free_volume'], period=100, filename='log.dat', overwrite=True)
    # ~~~~~~~~~~~~~
    def __init__(self, mc, seed, suffix='', test_type=None, nsample=None):
        hoomd.util.print_status_line();

        # initialize base class
        _compute.__init__(self);

        # create the c++ mirror class
        cl = _hoomd.CellList(hoomd.context.current.system_definition);
        hoomd.context.current.system.addCompute(cl, "auto_cl3")

        cls = None;
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ComputeFreeVolumeSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ComputeFreeVolumeConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ComputeFreeVolumeSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('ComputeFreeVolumeConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('ComputeFreeVolumeSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ComputeFreeVolumeEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ComputeFreeVolumeSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.ComputeFreeVolumeFacetedSphere;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ComputeFreeVolumePolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ComputeFreeVolumeSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.ComputeFreeVolumeSphereUnion;
            else:
                hoomd.context.msg.error("compute.free_volume: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.free_volume");
        else:
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ComputeFreeVolumeGPUSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ComputeFreeVolumeGPUConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ComputeFreeVolumeGPUSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('ComputeFreeVolumeGPUConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('ComputeFreeVolumeGPUSpheropolyhedron',mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ComputeFreeVolumeGPUEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ComputeFreeVolumeGPUSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.ComputeFreeVolumeGPUFacetedSphere;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ComputeFreeVolumeGPUPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ComputeFreeVolumeGPUSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.ComputeFreeVolumeGPUSphereUnion;
            else:
                hoomd.context.msg.error("compute.free_volume: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.free_volume");

        if suffix is not '':
            suffix = '_' + suffix

        self.cpp_compute = cls(hoomd.context.current.system_definition,
                                mc.cpp_integrator,
                                cl,
                                seed,
                                suffix)

        if test_type is not None:
            itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(test_type)
            self.cpp_compute.setTestParticleType(itype)
        if nsample is not None:
            self.cpp_compute.setNumSamples(int(nsample))

        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name)
        self.enabled = True

## \internal
# \brief Base class for external fields
#
# An external in hoomd_script reflects an ExternalField in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ extrnal itself is tracked optionally passed
# to the hpmc integrator. While external fields are Compute types and are added
# to the System they will not be enforced unless they are added to the integrator.
# Only one external field can be held by the integrator so if multiple fields are
# required use the external_field_composite class to manage them.
class _external(_compute):
    ## \internal
    # \brief Initialize an empty external.
    #
    # \post nothing is done here yet.
    def __init__(self):
        _compute.__init__(self);
        self.cpp_compute = None;
        # nothing else to do.

## Restrain particles on a lattice
#
# The command hpmc.compute.lattice_field specifies that a harmonic spring is added
# to every particle and a
#
# \f{eqnarray*}
# V_{i}(r)  = k_r*(r_i-r_{oi})^2 \\
# V_{i}(q)  = k_q*(q_i-q_{oi})^2 \\
# \f}
#
# \note the 1/2 is not included here and \f$ k \f$ should be defined appropriately
#
# - \f$ k_r \f$ - \c translational spring constant (in energy units)
# - \f$ r_{o} \f$ - \c lattice positions (in distance units)
# - \f$ k_q \f$ - \c rotational spring constant (in energy units)
# - \f$ q_{o} \f$ - \c lattice orientations
#
#
# Once initialized, the compute provides the following log quantities that can be logged via analyze.log:
# **lattice_energy** -- total lattice energy
# **lattice_energy_pp_avg** -- average lattice energy per particle
# **lattice_energy_pp_sigma** -- standard deviation of the lattice energy per particle
# **lattice_translational_spring_constant** -- translational spring constant
# **lattice_rotational_spring_constant** -- rotational spring constant
# **lattice_num_samples** -- number of samples used to compute the average and standard deviation
# \b Example:
# \code
# mc = hpmc.integrate.sphere(seed=415236);
# hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
# log = analyze.log(quantities=['lattice_energy'], period=100, filename='log.dat', overwrite=True);
# \endcode
class lattice_field(_external):
    ## Specify the lattice field.
    #
    # \param mc MC integrator (don't specify a new integrator later, lattice will continue to use the old one)
    # \param position list of positions to restrain each particle.
    # \param orientation list of orientations to restrain each particle.
    # \param k translational spring constant
    # \param q rotational spring constant
    # \param symmetry list of equivalent quaternions for the shape.
    # \param setup if true pass the object created to the integrator.
    def __init__(self, mc, position = [], orientation = [], k = 0.0, q = 0.0, symmetry = [], setup=True):
        import numpy
        hoomd.util.print_status_line();
        _external.__init__(self);
        cls = None;
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ExternalFieldLatticeSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ExternalFieldLatticeConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ExternalFieldLatticeSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('ExternalFieldLatticeConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('ExternalFieldLatticeSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ExternalFieldLatticeEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ExternalFieldLatticeSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.ExternalFieldLatticeFacetedSphere;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ExternalFieldLatticePolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ExternalFieldLatticeSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.ExternalFieldLatticeSphereUnion;
            else:
                hoomd.context.msg.error("compute.position_lattice_field: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.position_lattice_field");
        else:
            hoomd.context.msg.error("GPU not supported yet")
            raise RuntimeError("Error initializing compute.position_lattice_field");

        if type(position) == numpy.ndarray:
            position = position.tolist();
        else:
            position = list(position);

        if type(orientation) == numpy.ndarray:
            orientation = orientation.tolist();
        else:
            orientation = list(orientation);

        self.compute_name = "lattice_field"
        self.cpp_compute = cls(hoomd.context.current.system_definition, position, k, orientation, q, symmetry);
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name)
        if setup :
            mc.set_external(self);

    ## Reset the reference positions or reference orientations
    #
    # \param position list of positions to restrain each particle.
    # \param orientation list of orientations to restrain each particle.
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed=415236);
    # lattice = hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
    # lattice.set_references(position=bcc_lattice)
    # \endcode
    def set_references(self, position = [], orientation = []):
        import numpy
        hoomd.util.print_status_line();
        if type(position) == numpy.ndarray:
            position = position.tolist();
        else:
            position = list(position);

        if type(orientation) == numpy.ndarray:
            orientation = orientation.tolist();
        else:
            orientation = list(orientation);

        self.cpp_compute.setReferences(position, orientation);

    ## Set the translational and rotational spring constants
    #
    # \param k translational spring constant
    # \param q rotational spring constant
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed=415236);
    # lattice = hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
    # ks = np.linspace(1000, 0.01, 100);
    # for k in ks:
    #   lattice.set_params(k=k, q=0.0);
    #   run(1000)
    # \endcode
    def set_params(self, k, q):
        hoomd.util.print_status_line();
        self.cpp_compute.setParams(float(k), float(q));

    ## Reset the statistics counters
    #
    # \param timestep the timestep to pass into the reset function.
    #
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed=415236);
    # lattice = hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
    # ks = np.linspace(1000, 0.01, 100);
    # for k in ks:
    #   lattice.set_params(k=k, q=0.0);
    #   lattice.reset();
    #   run(1000)
    # \endcode
    def reset(self, timestep = None):
        hoomd.util.print_status_line();
        if timestep == None:
            timestep = hoomd.context.current.system.getCurrentTimeStep();
        self.cpp_compute.reset(timestep);



## Manage multiple external fields
#
# compute.external_field_composite allows the user to create and compute multiple
# external fields. Once created use compute.external_field_composite.add_field
# to add a new field.
#
# Once initialized, the compute provides a log quantities that other external
# fields create. See those external fields to find the quantities
#
class external_field_composite(_external):
    ## Specifies the compute class for multiple external fields
    # \param mc MC integrator (don't specify a new integrator later, external_field_composite will continue to use the old one)
    # \param setup if setup is true then the object created will be added to the integrator
    # \param fields List of external fields to combine together.
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(...);
    # walls = hpmc.compute.walls(...)
    # lattice = hpmc.compute.lattice(...)
    # composite_field = hpmc.compute.external_field_composite(mc, fields=[walls, lattice])
    # ~~~~~~~~~~~~~
    def __init__(self, mc, setup=True, fields = None):
        _external.__init__(self);
        cls = None;
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.ExternalFieldCompositeSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.ExternalFieldCompositeConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.ExternalFieldCompositeSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('ExternalFieldCompositeConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('ExternalFieldCompositeSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.ExternalFieldCompositeEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.ExternalFieldCompositeSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.ExternalFieldCompositeFacetedSphere;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.ExternalFieldCompositePolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.ExternalFieldCompositeSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.ExternalFieldCompositeSphereUnion;
            else:
                hoomd.context.msg.error("compute.position_lattice_field: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.position_lattice_field");
        else:
            hoomd.context.msg.error("GPU not supported yet")
            raise RuntimeError("Error initializing compute.position_lattice_field");

        self.compute_name = "composite_field"
        self.cpp_compute = cls(hoomd.context.current.system_definition);
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);
        if setup:
            mc.set_external(self);

        if not fields is None:
            self.add_field(fields=fields);

    ## Add an external field to the ensemble
    #
    # \param fields list of fields to add
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(...);
    # composite_field = hpmc.compute.external_field_composite(mc)
    # walls = hpmc.compute.walls(..., setup=False)
    # lattice = hpmc.compute.lattice(..., setup=False)
    # composite_field.add_field(fields=[walls, lattice])
    # ~~~~~~~~~~~~
    def add_field(self, fields):
        if not type(fields) == list:
            fields = list(fields);
        for field in fields:
            self.cpp_compute.addExternal(field.cpp_compute);

## Manage walls (an external field type)
#
# compute.wall allows the user to implement one or more walls. If multiple walls are added, then particles are confined by the INTERSECTION of all of these walls. In other words,
# particles are confined by all walls if they independently satisfy the confinement condition associated with each separate wall.
# Once you've created an instance of this class, use compute.wall.add_sphere_wall
# to add a new spherical wall, compute.wall.add_cylinder_wall to add a new cylindrical wall, or
# compute.wall.add_plane_wall to add a new plane wall.
#
# Once initialized, the compute provides the following log quantities that can be logged via analyze.log:
# **hpmc_wall_volume** -- the volume associated with the intersection of implemented walls. This number is only meaningful
# if the user has initially provided it through compute.wall.set_volume(). It will subsequently change when
# the box is resized and walls are scaled appropriately.
# **hpmc_wall_sph_rsq-i** -- the squared radius of the spherical wall indexed by i, beginning at 0 in the order the sphere
# walls were added to the system.
# **hpmc_wall_cyl_rsq-i** -- the squared radius of the cylindrical wall indexed by i, beginning at 0 in the order the
# cylinder walls were added to the system.
# \b Example:
# \code
# mc = hpmc.integrate.sphere(seed = 415236);
# ext_wall = hpmc.compute.wall(mc);
# ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
# ext_wall.set_volume(4./3.*np.pi);
# log = analyze.log(quantities=['hpmc_wall_volume','hpmc_wall_sph_rsq-0'], period=100, filename='log.dat', overwrite=True);
# \endcode
class wall(_external):
    ## Specifies the compute class for the walls external field type.
    # \param mc MC integrator
    # \param setup if setup is true then the object created will be added to the integrator
    index=0;
    def __init__(self, mc, setup=True):
        hoomd.util.print_status_line();
        _external.__init__(self);
        # create the c++ mirror class
        cls = None;
        self.compute_name = "wall-"+str(wall.index)
        wall.index+=1
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.WallSphere;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('WallConvexPolyhedron', mc.max_verts);
            else:
                hoomd.context.msg.error("compute.wall: Unsupported integrator.\n");
                raise RuntimeError("Error initializing compute.wall");
        else:
            hoomd.context.msg.error("GPU not supported yet")
            raise RuntimeError("Error initializing compute.wall");

        self.cpp_compute = cls(hoomd.context.current.system_definition, mc.cpp_integrator);
        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

        if setup:
            mc.set_external(self);

    ## Count the overlaps associated with the walls. A particle "overlaps" with a wall if it fails to meet
    ## the confinement condition associated with the wall.
    #
    # \returns The number of overlaps associated with the walls
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # run(100)
    # num_overlaps = ext_wall.count_overlaps();
    # ~~~~~~~~~~~~
    def count_overlaps(self, exit_early=False):
        hoomd.util.print_status_line();
        return self.cpp_compute.countOverlaps(hoomd.context.current.system.getCurrentTimeStep(), exit_early);

    ## Add a spherical wall to the simulation
    #
    # \param radius radius of spherical wall
    # \param origin origin (center) of spherical wall.
    # \param inside if True, then particles are CONFINED by the wall if they exist entirely inside the sphere (in the portion of connected space that contains the origin).
    # if False, then particles are CONFINED by the wall if they exist entirely outside the sphere (in the portion of connected space that does not contain the origin). DEFAULTS to True.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ~~~~~~~~~~~~
    def add_sphere_wall(self, radius, origin, inside = True):
        hoomd.util.print_status_line();
        self.cpp_compute.AddSphereWall(_hpmc.make_sphere_wall(radius, origin, inside));

    ## Change the parameters associated with a particular sphere wall
    #
    # \param index index of the sphere wall to be modified. indices begin at 0 in the order the sphere walls were added to the system.
    # \param radius new radius of spherical wall
    # \param origin new origin (center) of spherical wall.
    # \param inside new confinement condition. if True, then particles are CONFINED by the wall if they exist entirely inside the sphere (in the portion of connected space that contains the origin).
    # if False, then particles are CONFINED by the wall if they exist entirely outside the sphere (in the portion of connected space that does not contain the origin). DEFAULTS to True.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ext_wall.set_sphere_wall(index = 0, radius = 3.0, origin = [0, 0, 0], inside = True);
    # ~~~~~~~~~~~~
    def set_sphere_wall(self, index, radius, origin, inside = True):
        hoomd.util.print_status_line();
        self.cpp_compute.SetSphereWallParameter(index, _hpmc.make_sphere_wall(radius, origin, inside));

    ## Access a parameter associated with a particular sphere wall
    #
    # \param index index of the sphere wall to be accessed. indices begin at 0 in the order the sphere walls were added to the system.
    # \param param name of parameter to be accessed. options are "rsq" (squared radius of sphere wall), "origin" (origin of sphere wall), and "inside" (confinement condition associated with sphere wall)
    # \returns value of queried parameter
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # rsq = ext_wall.get_sphere_wall_param(index = 0, param = "rsq");
    # ~~~~~~~~~~~~
    def get_sphere_wall_param(self, index, param):
        hoomd.util.print_status_line();
        t = self.cpp_compute.GetSphereWallParametersPy(index);
        if param == "rsq":
            return t[0];
        elif param == "origin":
            return t[1];
        elif param == "inside":
            return t[2];
        else:
            hoomd.context.msg.error("compute.wall.get_sphere_wall_param: Parameter type is not valid. Choose from rsq, origin, inside.");
            raise RuntimeError("Error: compute.wall");

    ## Remove a particular sphere wall from the simulation
    #
    # \param index index of the sphere wall to be removed. indices begin at 0 in the order the sphere walls were added to the system.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ext_wall.remove_sphere_wall(index = 0);
    # ~~~~~~~~~~~~
    def remove_sphere_wall(self, index):
        hoomd.util.print_status_line();
        self.cpp_compute.RemoveSphereWall(index);

    ## Get the current number of sphere walls in the simulation
    #
    # \returns the current number of sphere walls in the simulation
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # num_sph_walls = ext_wall.get_num_sphere_walls();
    # ~~~~~~~~~~~~
    def get_num_sphere_walls(self):
        hoomd.util.print_status_line();
        return self.cpp_compute.getNumSphereWalls();

    ## Add a cylindrical wall to the simulation
    #
    # \param radius radius of cylindrical wall
    # \param origin origin (center) of cylindrical wall
    # \param orientation vector that defines the direction of the long axis of the cylinder. will be normalized automatically by hpmc.
    # \param inside if True, then particles are CONFINED by the wall if they exist entirely inside the cylinder (in the portion of connected space that contains the origin).
    # if False, then particles are CONFINED by the wall if they exist entirely outside the cylinder (in the portion of connected space that does not contain the origin). DEFAULTS to True.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # ~~~~~~~~~~~~
    def add_cylinder_wall(self, radius, origin, orientation, inside = True):
        hoomd.util.print_status_line();
        param = _hpmc.make_cylinder_wall(radius, origin, orientation, inside);
        self.cpp_compute.AddCylinderWall(param);

    ## Change the parameters associated with a particular cylinder wall
    #
    # \param index index of the cylinder wall to be modified. indices begin at 0 in the order the cylinder walls were added to the system.
    # \param radius new radius of cylindrical wall
    # \param origin new origin (center) of cylindrical wall
    # \param orientation new orientation vector of cylindrical wall
    # \param inside new confinement condition. if True, then particles are CONFINED by the wall if they exist entirely inside the cylinder (in the portion of connected space that contains the origin).
    # if False, then particles are CONFINED by the wall if they exist entirely outside the cylinder (in the portion of connected space that does not contain the origin). DEFAULTS to True.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # ext_wall.set_cylinder_wall(index = 0, radius = 3.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # ~~~~~~~~~~~~
    def set_cylinder_wall(self, index, radius, origin, orientation, inside = True):
        hoomd.util.print_status_line();
        param = _hpmc.make_cylinder_wall(radius, origin, orientation, inside)
        self.cpp_compute.SetCylinderWallParameter(index, param);

    ## Access a parameter associated with a particular cylinder wall
    #
    # \param index index of the cylinder wall to be accessed. indices begin at 0 in the order the cylinder walls were added to the system.
    # \param param name of parameter to be accessed. options are "rsq" (squared radius of cylinder wall), "origin" (origin of cylinder wall), "orientation" (orientation of cylinder wall),
    # and "inside" (confinement condition associated with cylinder wall)
    # \returns value of queried parameter
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # rsq = ext_wall.get_cylinder_wall_param(index = 0, param = "rsq");
    # ~~~~~~~~~~~~
    def get_cylinder_wall_param(self, index, param):
        hoomd.util.print_status_line();
        t = self.cpp_compute.GetCylinderWallParametersPy(index);
        if param == "rsq":
            return t[0];
        elif param == "origin":
            return t[1];
        elif param == "orientation":
            return t[2];
        elif param == "inside":
            return t[3];
        else:
            hoomd.context.msg.error("compute.wall.get_cylinder_wall_param: Parameter type is not valid. Choose from rsq, origin, orientation, inside.");
            raise RuntimeError("Error: compute.wall");

    ## Remove a particular cylinder wall from the simulation
    #
    # \param index index of the cylinder wall to be removed. indices begin at 0 in the order the cylinder walls were added to the system.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # ext_wall.remove_cylinder_wall(index = 0);
    # ~~~~~~~~~~~~
    def remove_cylinder_wall(self, index):
        hoomd.util.print_status_line();
        self.cpp_compute.RemoveCylinderWall(index);

    ## Get the current number of cylinder walls in the simulation
    #
    # \returns the current number of cylinder walls in the simulation
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_cylinder_wall(radius = 1.0, origin = [0, 0, 0], orientation = [0, 0, 1], inside = True);
    # num_cyl_walls = ext_wall.get_num_cylinder_walls();
    # ~~~~~~~~~~~~
    def get_num_cylinder_walls(self):
        hoomd.util.print_status_line();
        return self.cpp_compute.getNumCylinderWalls();

    ## Add a plane wall to the simulation
    #
    # \param normal vector normal to the plane. this, in combination with a point on the plane, defines the plane entirely. It will be normalized automatically by hpmc.
    # The direction of the normal vector defines the confinement condition associated with the plane wall. If every part of a particle exists in the halfspace into which the normal points, then that particle is CONFINED by the plane wall.
    # \param origin a point on the plane wall. this, in combination with the normal vector, defines the plane entirely.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
    # ~~~~~~~~~~~~
    def add_plane_wall(self, normal, origin):
        hoomd.util.print_status_line();
        self.cpp_compute.AddPlaneWall(_hpmc.make_plane_wall(normal, origin, True));

    ## Change the parameters associated with a particular plane wall
    #
    # \param index index of the plane wall to be modified. indices begin at 0 in the order the plane walls were added to the system.
    # \param normal new vector normal to the plane
    # \param origin new point on the plane
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
    # ext_wall.set_plane_wall(index = 0, normal = [0, 0, 1], origin = [0, 0, 1]);
    # ~~~~~~~~~~~~
    def set_plane_wall(self, index, normal, origin):
        hoomd.util.print_status_line();
        self.cpp_compute.SetPlaneWallParameter(index, _hpmc.make_plane_wall(normal, origin, True));

    ## Access a parameter associated with a particular plane wall
    #
    # \param index index of the plane wall to be accessed. indices begin at 0 in the order the plane walls were added to the system.
    # \param param name of parameter to be accessed. options are "normal" (vector normal to the plane wall), and "origin" (point on the plane wall)
    # \returns value of queried parameter
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
    # n = ext_wall.get_plane_wall_param(index = 0, param = "normal");
    # ~~~~~~~~~~~~
    def get_plane_wall_param(self, index, param):
        hoomd.util.print_status_line();
        t = self.cpp_compute.GetPlaneWallParametersPy(index);
        if param == "normal":
            return t[0];
        elif param == "origin":
            return t[1];
        else:
            hoomd.context.msg.error("compute.wall.get_plane_wall_param: Parameter type is not valid. Choose from normal, origin.");
            raise RuntimeError("Error: compute.wall");

    ## Remove a particular plane wall from the simulation
    #
    # \param index index of the plane wall to be removed. indices begin at 0 in the order the plane walls were added to the system.
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
    # ext_wall.remove_plane_wall(index = 0);
    # ~~~~~~~~~~~~
    def remove_plane_wall(self, index):
        hoomd.util.print_status_line();
        self.cpp_compute.RemovePlaneWall(index);

    ## Get the current number of plane walls in the simulation
    #
    # \returns the current number of plane walls in the simulation
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_plane_wall(normal = [0, 0, 1], origin = [0, 0, 0]);
    # num_plane_walls = ext_wall.get_num_plane_walls();
    # ~~~~~~~~~~~~
    def get_num_plane_walls(self):
        hoomd.util.print_status_line();
        return self.cpp_compute.getNumPlaneWalls();

    ## Set the volume associated with the intersection of all walls in the system. This number will subsequently change when the box is resized and walls are scaled appropriately.
    #
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ext_wall.set_volume(4./3.*np.pi);
    # ~~~~~~~~~~~~
    def set_volume(self, volume):
        hoomd.util.print_status_line();
        self.cpp_compute.setVolume(volume);

    ## Get the current volume associated with the intersection of all walls in the system. If this quantity has not previously been set by the user, this returns a meaningless value.
    #
    # \returns the current volume associated with the intersection of all walls in the system
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ext_wall.set_volume(4./3.*np.pi);
    # run(100)
    # curr_vol = ext_wall.get_volume();
    # ~~~~~~~~~~~~
    def get_volume(self):
        hoomd.util.print_status_line();
        return self.cpp_compute.getVolume();

    ## Get the simulation box that the wall class is currently storing
    #
    # \returns the boxdim object that the wall class is currently storing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # ext_wall.set_volume(4./3.*np.pi);
    # run(100)
    # curr_box = ext_wall.get_curr_box();
    # ~~~~~~~~~~~~
    def get_curr_box(self):
        hoomd.util.print_status_line();
        return data.boxdim(Lx=self.cpp_compute.GetCurrBoxLx(),
                           Ly=self.cpp_compute.GetCurrBoxLy(),
                           Lz=self.cpp_compute.GetCurrBoxLz(),
                           xy=self.cpp_compute.GetCurrBoxTiltFactorXY(),
                           xz=self.cpp_compute.GetCurrBoxTiltFactorXZ(),
                           yz=self.cpp_compute.GetCurrBoxTiltFactorYZ());

    ## Set the simulation box that the wall class is currently storing.
    ## You may want to set this independently so that you can cleverly control whether or not the walls actually scale in case you manually resize your simulation box.
    ## The walls scale automatically when they get the signal that the global box, associated with the system definition, has scaled. They do so, however, with a scale factor associated with
    ## the ratio of the volume of the global box to the volume of the box that the walls class is currently storing. (After the scaling the box that the walls class is currently storing is updated appropriately.)
    ## If you want to change the simulation box WITHOUT scaling the walls, then, you must first update the simulation box that the walls class is storing, THEN update the global box associated with the system definition.
    #
    # \returns nothing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # init_box = hoomd.data.boxdim(L=10, dimensions=3);
    # system = hoomd.init.create_empty(N=1, box=init_box, particle_types=['A']);
    # system.particles[0].position = [0,0,0];
    # system.particles[0].type = 'A';
    # mc = hpmc.integrate.sphere(seed = 415236);
    # mc.shape_param.set('A', diameter = 2.0);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 3.0, origin = [0, 0, 0], inside = True);
    # ext_wall.set_curr_box(Lx=2.0*init_box.Lx, Ly=2.0*init_box.Ly, Lz=2.0*init_box.Lz, xy=init_box.xy, xz=init_box.xz, yz=init_box.yz);
    # system.sysdef.getParticleData().setGlobalBox(ext_wall.get_curr_box()._getBoxDim())
    # ~~~~~~~~~~~~
    def set_curr_box(self, Lx = None, Ly = None, Lz = None, xy = None, xz = None, yz = None):
        # much of this is from hoomd's update.py box_resize class
        hoomd.util.print_status_line();
        if Lx is None and Ly is None and Lz is None and xy is None and xz is None and yz is None:
            hoomd.context.msg.warning("compute.wall.set_curr_box: Ignoring request to set the wall's box without parameters\n")
            return

        # setup arguments
        if Lx is None:
            Lx = self.cpp_compute.GetCurrBoxLx();
        if Ly is None:
            Ly = self.cpp_compute.GetCurrBoxLy();
        if Lz is None:
            Lz = self.cpp_compute.GetCurrBoxLz();

        if xy is None:
            xy = self.cpp_compute.GetCurrBoxTiltFactorXY();
        if xz is None:
            xz = self.cpp_compute.GetCurrBoxTiltFactorXZ();
        if yz is None:
            yz = self.cpp_compute.GetCurrBoxTiltFactorYZ();

        self.cpp_compute.SetCurrBox(Lx, Ly, Lz, xy, xz, yz);

## Compute the Frenkel-Ladd Energy of a crystal
#
# The command hpmc.compute.frenkel_ladd_energy interacts with the hpmc.compute.lattice_field
# and hpmc.update.remove_drift
#
# Once initialized, the compute provides the log quantities from the hpmc.compute.lattice_field
class frenkel_ladd_energy(_compute):
    # \param ln_gamma log of the translational spring constant
    # \param q_factor scale factor between the translational spring constant and rotational spring constant
    # \param r0 reference lattice positions
    # \param q0 reference lattice orientations
    # \param drift_period period call the remove drift updater
    # \b Example:
    # \code
    # mc = hpmc.integrate.convex_polyhedron(seed=seed);
    # mc.shape_param.set("A", vertices=verts)
    # mc.set_params(d=0.005, a=0.005)
    # #set the FL parameters
    # fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)
    # \endcode
    def __init__(   self,
                    mc,
                    ln_gamma,
                    q_factor,
                    r0,
                    q0,
                    drift_period,
                    symmetry = []
                ):
        import math
        import numpy
        hoomd.util.print_status_line();
        # initialize base class
        _compute.__init__(self);

        if type(r0) == numpy.ndarray:
            self.lattice_positions = r0.tolist();
        else:
            self.lattice_positions = list(r0);

        if type(q0) == numpy.ndarray:
            self.lattice_orientations = q0.tolist();
        else:
            self.lattice_orientations = list(q0);


        self.mc = mc;
        self.q_factor = q_factor;
        self.trans_spring_const = math.exp(ln_gamma);
        self.rotat_spring_const = self.q_factor*self.trans_spring_const;
        self.lattice = lattice_field(   self.mc,
                                        position = self.lattice_positions,
                                        orientation = self.lattice_orientations,
                                        k = self.trans_spring_const,
                                        q = self.rotat_spring_const,
                                        symmetry=symmetry);
        self.remove_drift = hpmc.update.remove_drift(self.mc, self.lattice, period=drift_period);

    ## Reset the statistics counters
    #
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed=415236);
    # fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)
    # ks = np.linspace(1000, 0.01, 100);
    # for k in ks:
    #   fl.set_params(ln_gamma=math.log(k), q_factor=10.0);
    #   fl.reset_statistics();
    #   run(1000)
    # \endcode
    def reset_statistics(self):
        hoomd.util.print_status_line();
        self.lattice.reset(0);

    ## Set the Frenkel-Ladd parameters
    #
    # \param ln_gamma log of the translational spring constant
    # \param q_factor scale factor between the translational spring constant and rotational spring constant
    #
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed=415236);
    # fl = hpmc.compute.frenkel_ladd_energy(mc=mc, ln_gamma=0.0, q_factor=10.0, r0=rs, q0=qs, drift_period=1000)
    # ks = np.linspace(1000, 0.01, 100);
    # for k in ks:
    #   fl.set_params(ln_gamma=math.log(k), q_factor=10.0);
    #   fl.reset_statistics();
    #   run(1000)
    # \endcode
    def set_params(self, ln_gamma = None, q_factor = None):
        import math
        hoomd.util.print_status_line();
        if not q_factor is None:
            self.q_factor = q_factor;
        if not ln_gamma is None:
            self.trans_spring_const = math.exp(ln_gamma);
        self.rotat_spring_const = self.q_factor*self.trans_spring_const;
        self.lattice.set_params(self.trans_spring_const, self.rotat_spring_const);
