# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd
from hoomd.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.typeconverter import OnlyIf, to_type_converter
from hoomd.typeparam import TypeParameter
from hoomd.hpmc import _hpmc
from hoomd.integrate import _BaseIntegrator
from hoomd.logger import Loggable
import hoomd
import json


# Helper method to inform about implicit depletants citation
# TODO: figure out where to call this
def cite_depletants():
    _citation = hoomd.cite.article(
        cite_key='glaser2015',
        author=['J Glaser', 'A S Karas', 'S C Glotzer'],
        title='A parallel algorithm for implicit depletant simulations',
        journal='The Journal of Chemical Physics',
        volume=143,
        pages='184110',
        year='2015',
        doi='10.1063/1.4935175',
        feature='implicit depletants')
    hoomd.cite._ensure_global_bib().add(_citation)


class _HPMCIntegrator(_BaseIntegrator):
    """Base class hard particle Monte Carlo integrator.

    Note:
        :py:class:`_HPMCIntegrator` is the base class for all HPMC integrators.
        Users should not instantiate this class directly. The attributes
        documented here are available to all HPMC integrators.

    .. rubric:: Hard particle Monte Carlo

    In hard particle Monte Carlo systems, the particles in the
    `hoomd.Simulation` `hoomd.State` are extended objects with positions and
    orientations. During each time step of a `hoomd.Simulation.run`, `nselect`
    trial moves are attempted for each particle in the system.

    A trial move may be a rotation or a translation move, selected randomly
    according to the `move_ratio`. Translation trial moves are selected randomly
    from a sphere of radius `d`, where `d` is set independently for each
    particle type. Rotational trial moves are selected with a maximum move size
    of `a`, where `a` is set independently for each particle type. In 2D
    simulations, `a` is the maximum angle (in radians) by which a particle will
    be rotated. In 3D, `a` is the magnitude of the random rotation quaternion as
    defined in Frenkel and Smit. `move_ratio` can be set to 0 or 1 to enable
    only rotation or translation moves, respectively.

    The `seed` parameter sets the seed for the random number generator.
    Simulations with the same initial condition and same seed will follow
    the same trajectory.

    Note:
        Full trajectory reproducibility is only possible with the same HOOMD
        binary installation, hardware, and execution configuration.
        Recompiling with different options, using a different version of HOOMD,
        running on a different hardware platform, or changing the parallel
        execution configuration may produce different trajectories due to
        limited floating point precision or parallel algorithmic differences.

    After proposing the trial move, the HPMC integrator checks to see if the
    new particle configuration overlaps with any other particles in the system.
    If there are overlaps, it rejects the move. It accepts the move when there
    are no overlaps.

    Setting elements of `interaction_matrix` to False disables overlap checks
    between specific particle types. `interaction_matrix` is a particle types
    by particle types matrix allowing for non-additive systems.

    The `fugacity` parameter enables implicit depletants when non-zero.
    TODO: Describe implicit depletants algorithm. No need to write this now,
    as Jens is rewriting the implementation.

    .. rubric:: Parameters

    Attributes:
        a (TypeParameter[particle type, float]):
            Maximum size of rotation trial moves.

        d (TypeParameter[particle type, float]):
            Maximum size of displacement trial moves
            (distance units).

        fugacity (TypeParameter[particle type, float]):
            Depletant fugacity (in units of 1/volume) (**default:** ``0``)

        interaction_matrix (TypeParameter[\
                            Tuple[particle type, particle type], bool]):
            Set to ``False`` for a pair of particle types to allow disable
            overlap checks between particles of those types (**default:**
            ``True``).

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

        seed (int): Random number seed.

    .. rubric:: Attributes
    """

    _cpp_cls = None

    def __init__(self, seed, d, a, move_ratio, nselect):
        super().__init__()

        # Set base parameter dict for hpmc integrators
        param_dict = ParameterDict(seed=int(seed),
                                   move_ratio=float(move_ratio),
                                   nselect=int(nselect))
        self._param_dict.update(param_dict)

        # Set standard typeparameters for hpmc integrators
        typeparam_d = TypeParameter('d',
                                    type_kind='particle_types',
                                    param_dict=TypeParameterDict(float(d),
                                                                 len_keys=1))
        typeparam_a = TypeParameter('a',
                                    type_kind='particle_types',
                                    param_dict=TypeParameterDict(float(a),
                                                                 len_keys=1))

        typeparam_fugacity = TypeParameter('depletant_fugacity',
                                           type_kind='particle_types',
                                           param_dict=TypeParameterDict(
                                               0., len_keys=1))

        typeparam_inter_matrix = TypeParameter('interaction_matrix',
                                               type_kind='particle_types',
                                               param_dict=TypeParameterDict(
                                                   True, len_keys=2))

        self._extend_typeparam([
            typeparam_d, typeparam_a, typeparam_fugacity, typeparam_inter_matrix
        ])

    def attach(self, simulation):
        '''initialize the reflected c++ class'''
        sys_def = simulation.state._cpp_sys_def
        if (simulation.device.mode == 'gpu'
                and (self._cpp_cls + 'GPU') in _hpmc.__dict__):
            self._cpp_cell = _hoomd.CellListGPU(sys_def)
            if simulation._system_communicator is not None:
                self._cpp_cell.setCommunicator(simulation._system_communicator)
            self._cpp_obj = getattr(_hpmc,
                                    self._cpp_cls + 'GPU')(sys_def,
                                                           self._cpp_cell,
                                                           self.seed)
        else:
            if simulation.device.mode == 'gpu':
                simulation.device.cpp_msg.warning(
                    "Falling back on CPU. No GPU implementation for shape.\n")
            self._cpp_obj = getattr(_hpmc, self._cpp_cls)(sys_def, self.seed)
            self._cpp_cell = None

        super().attach(simulation)

    # Set the external field
    def set_external(self, ext):
        self._cpp_obj.setExternalField(ext.cpp_compute)

    # Set the patch
    def set_PatchEnergyEvaluator(self, patch):
        self._cpp_obj.setPatchEnergy(patch.cpp_evaluator)

    # TODO need to validate somewhere that quaternions are normalized

    @property
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Subclasses override this method to provide shape specific results.
        """
        raise NotImplementedError(
            "You are using a shape type that is not implemented! "
            "If you want it, please modify the "
            "hoomd.hpmc.integrate._HPMCIntegrator.get_type_shapes function.")

    def _return_type_shapes(self):
        if not self.is_attached:
            return None
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret

    @Loggable.log(flag='multi')
    def map_overlaps(self):
        """List[Tuple[int, int]]: List of overlapping particles.

        The list contains one entry for each overlapping pair of particles. When
        a tuple ``(i,j)`` is present in the list, there is an overlap between
        the particles with tags ``i`` and ``j``.

        Attention:
            `map_overlaps` does not support MPI parallel simulations.
        """

        if not self.is_attached:
            return None
        return self._cpp_obj.mapOverlaps()

    def map_energies(self):
        R""" Build an energy map of the system

        Returns:
            List of tuples. The i,j entry contains the pairwise interaction
            energy of the ith and jth particles (by tag)

        Note:
            :py:meth:`map_energies` does not support MPI parallel simulations.

        Example:
            mc = hpmc.integrate.shape(...)
            mc.shape_param.set(...)
            energy_map = np.asarray(mc.map_energies())
        """

        # TODO: update map_energies to new API

        self.update_forces()
        N = hoomd.context.current.system_definition.getParticleData(
        ).getMaximumTag() + 1
        energy_map = self.cpp_integrator.mapEnergies()
        return list(zip(*[iter(energy_map)] * N))

    @Loggable.log
    def overlaps(self):
        """int: Number of overlapping particle pairs.
        """
        if not self.is_attached:
            return None
        self._cpp_obj.communicate(True)
        return self._cpp_obj.countOverlaps(False)

    def test_overlap(self,
                     type_i,
                     type_j,
                     rij,
                     qi,
                     qj,
                     use_images=True,
                     exclude_self=False):
        """Test overlap between two particles.

        Args:
            type_i (str): Type of first particle
            type_j (str): Type of second particle
            rij (tuple): Separation vector **rj**-**ri** between the particle
              centers
            qi (tuple): Orientation quaternion of first particle
            qj (tuple): Orientation quaternion of second particle
            use_images (bool): If True, check for overlap between the periodic
              images of the particles by adding
              the image vector to the separation vector
            exclude_self (bool): If both **use_images** and **exclude_self** are
              true, exclude the primary image

        For two-dimensional shapes, pass the third dimension of **rij** as zero.

        Returns:
            True if the particles overlap.
        """
        self.update_forces()

        ti = hoomd.context.current.system_definition.getParticleData(
        ).getTypeByName(type_i)
        tj = hoomd.context.current.system_definition.getParticleData(
        ).getTypeByName(type_j)

        rij = hoomd.util.listify(rij)
        qi = hoomd.util.listify(qi)
        qj = hoomd.util.listify(qj)
        return self._cpp_obj.py_test_overlap(ti, tj, rij, qi, qj, use_images,
                                             exclude_self)

    @Loggable.log(flag='multi')
    def translate_moves(self):
        """int: Count of the accepted and rejected translate moves.

        Note:
            The count is reset to 0 at the start of each `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getCounters(1).translate

    @Loggable.log(flag='multi')
    def rotate_moves(self):
        """int: Count of the accepted and rejected rotate moves.

        Note:
            The count is reset to 0 at the start of each `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getCounters(1).rotate

    @Loggable.log
    def mps(self):
        """float: Number of trial moves performed per second.

        Note:
            The count of trial moves is reset at the start of each
            `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getMPS()

    @property
    def counters(self):
        """Trial move counters

        The counter object has the following attributes:

        * ``translate``: Tuple[`int`, `int`] - Number of accepted and rejected
          translate trial moves.
        * ``rotate``: Tuple[`int`, `int`] - Number of accepted and rejected
          rotate trial moves.
        * ``ovelap_checks``: `int` - Number of overlap checks performed.
        * ``overlap_errors``: `int` - Number of overlap checks that were too
          close to resolve.

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.  """
        return self._cpp_obj.getCounters(1)


class Sphere(_HPMCIntegrator):
    """Hard sphere Monte Carlo.

    Perform hard particle Monte Carlo of spheres defined by their diameter
    (see `shape`). When the shape parameter ``orientable`` is False (the
    default), `Sphere` only applies translation trial moves and ignores
    ``move_ratio``.

    Tip:
        Use spheres with ``diameter=0`` in conjunction with `jit` potentials
        for Monte Carlo simulations of particles interacting by pair potential
        with no hard core.

    Tip:
        Use `Sphere` in a 2D simulation to perform Monte Carlo on hard disks.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Examples::

        mc = hoomd.hpmc.integrate.Sphere(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(diameter=1.0)
        mc.shape["B"] = dict(diameter=2.0)
        mc.shape["C"] = dict(diameter=1.0, orientable=True)
        print('diameter = ', mc.shape["A"]["diameter"])

    Depletants Example::

        mc = hoomd.hpmc.integrate.Sphere(seed=415236, d=0.3, a=0.4, nselect=8)
        mc.shape["A"] = dict(diameter=1.0)
        mc.shape["B"] = dict(diameter=1.0)
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``diameter`` (`float`, **required**) - Sphere diameter
              (distance units).
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``orientable`` (`bool`, **default:** False) - set to True for
              spheres with orientation.
    """
    _cpp_cls = 'IntegratorHPMCMonoSphere'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameter=float,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Examples:
            The types will be 'Sphere' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'Sphere', 'diameter': 1},
             {'type': 'Sphere', 'diameter': 2}]
        """
        return super()._return_type_shapes()


class ConvexPolygon(_HPMCIntegrator):
    """Hard convex polygon Monte Carlo.

    Perform hard particle Monte Carlo of convex polygons defined by their
    vertices (see `shape`).

    Important:
        `ConvexPolygon` simulations must be performed in 2D systems.

    See Also:
        Use `SimplePolygon` for concave polygons.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Examples::

        mc = hoomd.hpmc.integrate.ConvexPolygon(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5),
                                       (0.5, -0.5),
                                       (0.5, 0.5),
                                       (-0.5, 0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys.

            * ``vertices`` (List[Tuple[float,float]], **required**) -
              vertices of the polygon (distance units).

              * Vertices **MUST** be specified in a *counter-clockwise* order.
              * The origin **MUST** be contained within the polygon.
              * Points inside the polygon **MUST NOT** be included.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `ConvexPolygon` shares data structures with
              `ConvexSpheropolygon`.

          Warning:
              HPMC does not check that all vertex requirements are met.
              Undefined behavior will result when they are violated.

    """
    _cpp_cls = 'IntegratorHPMCMonoConvexPolygon'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            ignore_statistics=False,
                                            sweep_radius=0.0,
                                            len_keys=1,
                                        ))

        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(ConvexPolygon, self)._return_type_shapes()


class ConvexSpheropolygon(_HPMCIntegrator):
    """Hard convex spheropolygon Monte Carlo.

    Perform hard particle Monte Carlo of convex spheropolygons defined by their
    vertices and a sweep radius (see `shape`). A spheropolygon is is a polygon
    rounded by a disk swept along the perimeter. The sweep radius may be 0.

    Important:
        `ConvexSpheropolygon` simulations must be performed in 2D systems.

    Tip:
        A 1-vertex spheropolygon is a disk and a 2-vertex spheropolygon is a
        rounded rectangle.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Examples::

        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5),
                                       (0.5, -0.5),
                                       (0.5, 0.5),
                                       (-0.5, 0.5)],
                             sweep_radius=0.1);

        mc.shape["A"] = dict(vertices=[(0,0)],
                             sweep_radius=0.5,
                             ignore_statistics=True);

        print('vertices = ', mc.shape["A"]["vertices"])

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (List[Tuple[float,float]], **required**) -
              vertices of the polygon  (distance units).

              * The origin **MUST** be contained within the spheropolygon.
              * Points inside the polygon should not be included.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (**default:** 0.0) - radius of the disk swept
              around the edges of the polygon (distance units). Set a non-zero
              ``sweep_radius`` to create a spheropolygon.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoSpheropolygon'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))

        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0.1,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(ConvexSpheropolygon, self)._return_type_shapes()


class SimplePolygon(_HPMCIntegrator):
    """Hard simple polygon Monte Carlo.

    Perform hard particle Monte Carlo of simple polygons defined by their
    vertices (see `shape`).

    Important:
        `SimplePolygon` simulations must be performed in 2D systems.

    See Also:
        Use `ConvexPolygon` for faster performance with convex polygons.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Examples::

        mc = hpmc.integrate.SimplePolygon(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(vertices=[(0, 0.5),
                                       (-0.5, -0.5),
                                       (0, 0),
                                       (0.5, -0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])


    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (List[Tuple[float,float]], **required**) -
              vertices of the polygon (distance units).

              * Vertices **MUST** be specified in a *counter-clockwise* order.
              * The polygon may be concave, but edges must not cross.
              * The origin may be inside or outside the shape.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `SimplePolygon` shares data structures with
              `ConvexSpheropolygon`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.

    """

    _cpp_cls = 'IntegratorHPMCMonoSimplePolygon'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            ignore_statistics=False,
                                            sweep_radius=0,
                                            len_keys=1))

        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(SimplePolygon, self)._return_type_shapes()


class Polyhedron(_HPMCIntegrator):
    """Hard polyhedra Monte Carlo.

    Perform hard particle Monte Carlo of general polyhedra defined by their
    vertices and faces (see `shape`). `Polyhedron` supports triangle meshes and
    spheres only. The mesh must be free of self-intersections.

    See Also:
        Use `ConvexPolyhedron` for faster performance with convex polyhedra.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Note:

        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent faces in the tree, different
        values of the number of faces per leaf node may yield different
        optimal performance. The capacity of leaf nodes is configurable.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.Polyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5, -0.5),
                                       (-0.5, -0.5, 0.5),
                                       (-0.5, 0.5, -0.5),
                                       (-0.5, 0.5, 0.5),
                                       (0.5, -0.5, -0.5),
                                       (0.5, -0.5, 0.5),
                                       (0.5, 0.5, -0.5),
                                       (0.5, 0.5, 0.5)],
                            faces=[[0, 2, 6],
                                   [6, 4, 0],
                                   [5, 0, 4],
                                   [5, 1, 0],
                                   [5, 4, 6],
                                   [5, 6, 7],
                                   [3, 2, 0],
                                   [3, 0, 1],
                                   [3, 6, 2],
                                   [3, 7, 6],
                                   [3, 1, 5],
                                   [3, 5, 7]])
        print('vertices = ', mc.shape["A"]["vertices"])
        print('faces = ', mc.shape["A"]["faces"])

    Depletants Example::

        mc = hpmc.integrate.Polyhedron(seed=415236, d=0.3, a=0.4, nselect=1)
        cube_verts = [(-0.5, -0.5, -0.5),
                      (-0.5, -0.5, 0.5),
                      (-0.5, 0.5, -0.5),
                      (-0.5, 0.5, 0.5),
                      (0.5, -0.5, -0.5),
                      (0.5, -0.5, 0.5),
                      (0.5, 0.5, -0.5),
                      (0.5, 0.5, 0.5)];
        cube_faces = [[0, 2, 6],
                      [6, 4, 0],
                      [5, 0, 4],
                      [5,1,0],
                      [5,4,6],
                      [5,6,7],
                      [3,2,0],
                      [3,0,1],
                      [3,6,2],
                      [3,7,6],
                      [3,1,5],
                      [3,5,7]]
        tetra_verts = [(0.5, 0.5, 0.5),
                       (0.5, -0.5, -0.5),
                       (-0.5, 0.5, -0.5),
                       (-0.5, -0.5, 0.5)];
        tetra_faces = [[0, 1, 2], [3, 0, 2], [3, 2, 1], [3,1,0]];

        mc.shape["A"] = dict(vertices=cube_verts, faces=cube_faces);
        mc.shape["B"] = dict(vertices=tetra_verts,
                             faces=tetra_faces,
                             origin = (0,0,0));
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (List[Tuple[float,float,float]], **required**) -
              vertices of the polyhedron (distance units).

              * The origin **MUST** strictly be contained in the generally
                nonconvex volume defined by the vertices and faces.
              * The origin centered sphere that encloses all vertices should
                be of minimal size for optimal performance.

            * ``faces`` (List[Tuple[int,int,int], **required**) -
              Vertex indices for every triangle in the mesh.

              * For visualization purposes, the faces **MUST** be defined with
                a counterclockwise winding order to produce an outward normal.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - radius of the
              sphere swept around the surface of the polyhedron (distance
              units). Set a non-zero sweep_radius to create a spheropolyhedron.
            * ``overlap`` (List[bool], **default:** None) - Check for overlaps
              between faces when ``overlap [i] & overlap[j]`` is nonzero (``&``
              is the bitwise AND operator). When not None, ``overlap`` must have
              a length equal to that of ``faces``. When None (the default),
              ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``origin`` (Tuple[float, float, float], **default:** (0,0,0)) -
              a point strictly inside the shape, needed for correctness of
              overlap checks.
            * ``hull_only`` (`bool`, **default:** False) - When True, only
              check for intersections between the convex hulls.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoPolyhedron'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(vertices=[(float, float, float)],
                                         faces=[(int, int, int)],
                                         sweep_radius=0.0,
                                         capacity=4,
                                         origin=(0., 0., 0.),
                                         hull_only=False,
                                         overlap=OnlyIf(to_type_converter(
                                             [bool]), allow_none=True),
                                         ignore_statistics=False,
                                         len_keys=1,
                                         _defaults={'overlap': None}))

        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Mesh', 'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
              [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]],
              'faces': [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]}]
        """
        return super(Polyhedron, self)._return_type_shapes()


class ConvexPolyhedron(_HPMCIntegrator):
    """Hard convex polyhedron Monte Carlo.

    Perform hard particle Monte Carlo of convex polyhedra defined by their
    vertices (see `shape`).

    See Also:
        Use `Polyhedron` for concave polyhedra.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.ConvexPolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(vertices=[(0.5, 0.5, 0.5),
                                       (0.5, -0.5, -0.5),
                                       (-0.5, 0.5, -0.5),
                                       (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])

    Depletants Example::

        mc = hpmc.integrate.ConvexPolyhedron(seed=415236,
                                             d=0.3,
                                             a=0.4,
                                             nselect=1)
        mc.shape["A"] = dict(vertices=[(0.5, 0.5, 0.5),
                                       (0.5, -0.5, -0.5),
                                       (-0.5, 0.5, -0.5),
                                       (-0.5, -0.5, 0.5)]);
        mc.shape["B"] = dict(vertices=[(0.05, 0.05, 0.05),
                                       (0.05, -0.05, -0.05),
                                       (-0.05, 0.05, -0.05),
                                       (-0.05, -0.05, 0.05)]);
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys.

            * ``vertices`` (List[Tuple[float,float,float]], **required**) -
              vertices of the polyhedron (distance units).

              * The origin **MUST** be contained within the polyhedron.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `ConvexPolyhedron` shares data structures with
              `ConvexSpheropolyhedron`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.

    """

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedron'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexPolyhedron, self)._return_type_shapes()


class FacetedEllipsoid(_HPMCIntegrator):
    """Hard faceted ellipsoid Monte Carlo.

    Perform hard particle Monte Carlo of faceted ellipsoids. A faceted ellipsoid
    is the intersection of an ellipsoid with a convex polyhedron defined through
    halfspaces (see `shape`). The equation defining each halfspace is given by:

    .. math::
        \\vec{n}_i\\cdot \\vec{r} + b_i \\le 0

    where :math:`\\vec{n}_i` is the face normal, and :math:`b_i` is  the offset.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.FacetedEllipsoid(seed=415236, d=0.3, a=0.4)

        # half-space intersection
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5]

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],
                      [-.1,-.5,.5],
                      [-.1,.5,.5],
                      [-.1,.5,-.5],
                      [1,-.5,-.5],
                      [1,-.5,.5],
                      [1,.5,.5],
                      [1,.5,-.5]]

        mc.shape["A"] = dict(normals=slab_normals,
                             offsets=slab_offsets,
                             vertices=slab_verts,
                             a=1.0,
                             b=0.5,
                             c=0.5);
        print('a = {}, b = {}, c = {}',
              mc.shape["A"]["a"], mc.shape["A"]["b"], mc.shape["A"]["c"])

    Depletants Example::

        mc = hpmc.integrate.FacetedEllipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(normals=[(-1,0,0),
                                      (1,0,0),
                                      (0,-1,0),
                                      (0,1,0),
                                      (0,0,-1),
                                      (0,0,1)],
                             a=1.0,
                             b=0.5,
                             c=0.25);
        # depletant sphere
        mc.shape["B"] = dict(normals=[], a=0.1, b=0.1, c=0.1);
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``normals`` (List[Tuple[float, float, float], **required**) -
              facet normals :math:`\\vec{n}_i`.
            * ``offsets`` (List[float], **required**) - list of offsets
              :math:`b_i` (squared distance units)
            * ``a`` (`float`, **required**) - half axis of ellipsoid in the *x*
              direction (distance units)
            * ``b`` (`float`, **required**) - half axis of ellipsoid in the *y*
              direction (distance units)
            * ``c`` (`float`, **required**) - half axis of ellipsoid in the *z*
              direction (distance units)
            * ``vertices`` (List[Tuple[float, float, float]], **default:** []) -
              list of vertices for intersection polyhedron (see note below).
            * ``origin`` (Tuple[float, float, float], **default:** (0,0,0)) -
              A point inside the shape.
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.

            Important:
                The origin must be chosen so as to lie **inside the shape**, or
                the overlap check will not work. This condition is not checked.

            Warning:
                Planes must not be coplanar.

            Note:
                For simple intersections with planes that do not intersect
                within the sphere, the vertices list can be left empty. When
                specified, the half-space intersection of the normals must match
                the convex polyhedron defined by the vertices (if non-empty),
                the half-space intersection is **not** calculated automatically.
    """

    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoid'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(normals=[(float, float, float)],
                                         offsets=[float],
                                         a=float,
                                         b=float,
                                         c=float,
                                         vertices=OnlyIf(
                                             to_type_converter(
                                                 [(float, float, float)]),
                                             allow_none=True),
                                         origin=(float, float, float),
                                         ignore_statistics=False,
                                         len_keys=1,
                                         _defaults={'vertices': None}))
        self._add_typeparam(typeparam_shape)


class Sphinx(_HPMCIntegrator):
    """Hard sphinx particle Monte Carlo.

    Perform hard particle Monte Carlo of sphere unions and differences defined
    by their positive and negative diameters (see `shape`).

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.Sphinx(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
        print('diameters = ', mc.shape["A"]["diameters"])

    Depletants Example::

        mc = hpmc.integrate.Sphinx(seed=415236, d=0.3, a=0.4, nselect=1)
        mc.shape["A"] = dict(centers=[(0,0,0), (1,0,0)], diameters=[1, -.25])
        mc.shape["B"] = dict(centers=[(0,0,0)], diameters=[.15])
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``diameters`` (List[float], **required**) -
              diameters of spheres (positive OR negative real numbers) (distance
              units).
            * ``centers`` (List[Tuple[float, float, float], **required**) -
              centers of spheres in local coordinate frame (distance units).
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoSphinx'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameters=[float],
                                            centers=[(float, float, float)],
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)


class ConvexSpheropolyhedron(_HPMCIntegrator):
    """Hard convex spheropolyhedron Monte Carlo.

    Perform hard particle Monte Carlo of convex spheropolyhedra defined by their
    vertices and a sweep radius (see `shape`).

    Tip:
        A 1-vertex spheropolygon is a sphere and a 2-vertex spheropolygon is a
        spherocylinder.

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.ConvexSpheropolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape['tetrahedron'] = dict(vertices=[(0.5, 0.5, 0.5),
                                                 (0.5, -0.5, -0.5),
                                                 (-0.5, 0.5, -0.5),
                                                 (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape['tetrahedron']["vertices"])

        mc.shape['SphericalDepletant'] = dict(vertices=[],
                                              sweep_radius=0.1,
                                              ignore_statistics=True);

    Depletants example::

        mc = hpmc.integrate.ConvexSpheropolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape["tetrahedron"] = dict(vertices=[(0.5, 0.5, 0.5),
                                                 (0.5, -0.5, -0.5),
                                                 (-0.5, 0.5, -0.5),
                                                 (-0.5, -0.5, 0.5)]);
        mc.shape["SphericalDepletant"] = dict(vertices=[], sweep_radius=0.1);
        mc.depletant_fugacity["SphericalDepletant"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (List[Tuple[float,float,float]], **required**) -
              vertices of the polyhedron (distance units).

              * The origin **MUST** be contained within the polyhedron.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - radius of the
              sphere swept around the surface of the polyhedron (distance
              units). Set a non-zero sweep_radius to create a spheropolyhedron.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoSpheropolyhedron'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0.1,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexSpheropolyhedron, self)._return_type_shapes()


class Ellipsoid(_HPMCIntegrator):
    """Hard ellipsoid Monte Carlo.

    Perform hard particle Monte Carlo of ellipsoids defined by 3 half axes
    (see `shape`).

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.Ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape["A"] = dict(a=0.5, b=0.25, c=0.125);
        print('ellipsoids parameters (a,b,c) = ',
              mc.shape["A"]["a"],
              mc.shape["A"]["b"],
              mc.shape["A"]["c"])

    Depletants Example::

        mc = hpmc.integrate.Ellipsoid(seed=415236, d=0.3, a=0.4, nselect=1)
        mc.shape["A"] = dict(a=0.5, b=0.25, c=0.125);
        mc.shape["B"] = dict(a=0.05, b=0.05, c=0.05);
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``a`` (`float`, **required**) - half axis of ellipsoid in the *x*
              direction (distance units)
            * ``b`` (`float`, **required**) - half axis of ellipsoid in the *y*
              direction (distance units)
            * ``c`` (`float`, **required**) - half axis of ellipsoid in the *z*
              direction (distance units)
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoEllipsoid'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            a=float,
                                            b=float,
                                            c=float,
                                            ignore_statistics=False,
                                            len_keys=1))

        self._extend_typeparam([typeparam_shape])

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.5, 'c': 1}]
        """
        return super()._return_type_shapes()


class SphereUnion(_HPMCIntegrator):
    """Hard sphere union Monte Carlo.

    Perform hard particle Monte Carlo of unions of spheres (see `shape`).

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Note:

        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent spheres in the tree, different
        values of the number of spheres per leaf node may yield different
        performance. The capacity of leaf nodes is configurable.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.SphereUnion(seed=415236, d=0.3, a=0.4)
        sphere1 = dict(diameter=1)
        sphere2 = dict(diameter=2)
        mc.shape["A"] = dict(shapes=[sphere1, sphere2],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1])
        print('diameter of the first sphere = ',
              mc.shape["A"]["shapes"][0]["diameter"])
        print('center of the first sphere = ', mc.shape["A"]["positions"][0])

    Depletants Example::

        mc = hpmc.integrate.SphereUnion(seed=415236, d=0.3, a=0.4, nselect=1)
        mc.shape["A"] = dict(diameters=[1.0, 1.0],
                             centers=[(-0.25, 0.0, 0.0),
                                      (0.25, 0.0, 0.0)]);
        mc.shape["B"] = dict(diameters=[0.05], centers=[(0.0, 0.0, 0.0)]);
        mc.depletant_fugacity["B"] = 3.0

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (List[dict], **required**) -
              Shape parameters for each sphere in the union. See `Sphere.shape`
              for the accepted parameters.
            * ``positions`` (List[Tuple[float, float, float]], **required**) -
              Position of each sphere in the union.
            * ``orientations`` (List[Tuple[float, float, float, float]],\
                                **default:** None) -
              Orientation of each sphere in the union. When not None,
              ``orientations`` must have a length equal to that of
              ``positions``. When None (the default), ``orientations`` is
              initialized with all [1,0,0,0]'s.
            * ``overlap`` (List[bool], **default:** None) - Check for overlaps
              between constituent particles when ``overlap [i] & overlap[j]``
              is nonzero (``&`` is the bitwise AND operator). When not None,
              ``overlap`` must have a length equal to that of ``faces``. When
              None (the default), ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoSphereUnion'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            shapes=[
                                                dict(diameter=float,
                                                     ignore_statistics=False,
                                                     orientable=False)
                                            ],
                                            positions=[(float, float, float)],
                                            orientations=OnlyIf(
                                                to_type_converter(
                                                    [(float, float, float,
                                                      float)]),
                                                allow_none=True),
                                            capacity=4,
                                            overlap=OnlyIf(to_type_converter(
                                                [int]), allow_none=True),
                                            ignore_statistics=False,
                                            len_keys=1,
                                            _defaults={
                                                'orientations': None,
                                                'overlap': None
                                            }))
        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """List[dict]: Description of shapes in ``type_shapes`` format.

        Examples:
            The type will be 'SphereUnion' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'SphereUnion',
              'centers': [[0, 0, 0], [0, 0, 1]],
              'diameters': [1, 0.5]},
             {'type': 'SphereUnion',
              'centers': [[1, 2, 3], [4, 5, 6]],
              'diameters': [0.5, 1]}]
        """
        return super()._return_type_shapes()


class ConvexSpheropolyhedronUnion(_HPMCIntegrator):
    """Hard convex spheropolyhedron union Monte Carlo.

    Perform hard particle Monte Carlo of unions of convex sphereopolyhedra
    (see `shape`).

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Note:

        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent spheropolyhedra in the tree,
        different values of the number of spheropolyhedra per leaf node may
        yield different performance. The capacity of leaf nodes is configurable.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion(seed=27,
                                                              d=0.3,
                                                              a=0.4)
        cube_verts = [[-1,-1,-1],
                      [-1,-1,1],
                      [-1,1,1],
                      [-1,1,-1],
                      [1,-1,-1],
                      [1,-1,1],
                      [1,1,1],
                      [1,1,-1]]
        mc.shape["A"] = dict(shapes=[cube_verts, cube_verts],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1]);
        print('vertices of the first cube = ',
              mc.shape["A"]["shapes"][0]["vertices"])
        print('center of the first cube = ', mc.shape["A"]["positions"][0])
        print('orientation of the first cube = ',
              mc.shape_param["A"]["orientations"][0])

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (List[dict], **required**) -
              Shape parameters for each spheropolyhedron in the union. See
              `ConvexSpheropolyhedron.shape` for the accepted parameters.
            * ``positions`` (List[Tuple[float, float, float]], **required**) -
              Position of each spheropolyhedron in the union.
            * ``orientations`` (List[Tuple[float, float, float, float]],\
                                **default:** None) -
              Orientation of each spheropolyhedron in the union. When not None,
              ``orientations`` must have a length equal to that of
              ``positions``. When None (the default), ``orientations`` is
              initialized with all [1,0,0,0]'s.
            * ``overlap`` (List[bool], **default:** None) - Check for overlaps
              between constituent particles when ``overlap [i] & overlap[j]``
              is nonzero (``&`` is the bitwise AND operator). When not None,
              ``overlap`` must have a length equal to that of ``faces``. When
              None (the default), ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedronUnion'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(shapes=[
                dict(vertices=[(float, float, float)],
                     sweep_radius=0.0,
                     ignore_statistics=False)
            ],
                positions=[(float, float, float)],
                orientations=OnlyIf(to_type_converter(
                    [(float, float, float, float)]), allow_none=True),
                overlap=OnlyIf(to_type_converter([int]), allow_none=True),
                ignore_statistics=False,
                capacity=4,
                len_keys=1,
                _defaults={
                'orientations': None,
                'overlap': None
            }))

        self._add_typeparam(typeparam_shape)
        # meta data
        self.metadata_fields = ['capacity']


class FacetedEllipsoidUnion(_HPMCIntegrator):
    """Hard convex spheropolyhedron union Monte Carlo.

    Perform hard particle Monte Carlo of unions of faceted ellipsoids
    (see `shape`).

    Args:
        seed (int): Random number seed.

        d (float): Default maximum size of displacement trial moves
            (distance units).

        a (float): Default maximum size of rotation trial moves.

        move_ratio (float): Ratio of translation moves to rotation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Note:

        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent spheropolyhedra in the tree,
        different values of the number of spheropolyhedra per leaf node may
        yield different performance. The capacity of leaf nodes is configurable.

    Important:
        Assign a `shape` specification for each particle type in the
        `hoomd.State`.

    Example::

        mc = hpmc.integrate.FacetedEllipsoidUnion(seed=27, d=0.3, a=0.4)

        # make a prolate Janus ellipsoid
        # cut away -x halfspace
        normals = [(-1,0,0)]
        offsets = [0]
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5)

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],
                      [-.1,-.5,.5],
                      [-.1,.5,.5],
                      [-.1,.5,-.5],
                      [1,-.5,-.5],
                      [1,-.5,.5],
                      [1,.5,.5],
                      [1,.5,-.5]]

        faceted_ellipsoid1 = dict(normals=slab_normals,
                                  offsets=slab_offsets,
                                  vertices=slab_verts,
                                  a=1.0,
                                  b=0.5,
                                  c=0.5);
        faceted_ellipsoid2 = dict(normals=slab_normals,
                                  offsets=slab_offsets,
                                  vertices=slab_verts,
                                  a=0.5,
                                  b=1,
                                  c=1);

        mc.shape["A"] = dict(shapes=[faceted_ellipsoid1, faceted_ellipsoid2],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1]);

        print('offsets of the first faceted ellipsoid = ',
              mc.shape["A"]["shapes"][0]["offsets"])
        print('normals of the first faceted ellispoid = ',
              mc.shape["A"]["shapes"][0]["normals"])
        print('vertices of the first faceted ellipsoid = ',
              mc.shape["A"]["shapes"][0]["vertices"]

    Attributes:
        shape (TypeParameter[particle type, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (List[dict], **required**) -
              Shape parameters for each faceted ellipsoid in the union. See
              `ConvexFacetedEllipsoid.shape` for the accepted parameters.
            * ``positions`` (List[Tuple[float, float, float]], **required**) -
              Position of each faceted ellipsoid in the union.
            * ``orientations`` (List[Tuple[float, float, float, float]],\
                                **default:** None) -
              Orientation of each faceted ellipsoid in the union. When not None,
              ``orientations`` must have a length equal to that of
              ``positions``. When None (the default), ``orientations`` is
              initialized with all [1,0,0,0]'s.
            * ``overlap`` (List[bool], **default:** None) - Check for overlaps
              between constituent particles when ``overlap [i] & overlap[j]``
              is nonzero (``&`` is the bitwise AND operator). When not None,
              ``overlap`` must have a length equal to that of ``faces``. When
              None (the default), ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** False) - set to True
              to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoidUnion'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 move_ratio=0.5,
                 nselect=4):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(shapes=[
                dict(a=float,
                     b=float,
                     c=float,
                     normals=[(float, float, float)],
                     offsets=[float],
                     vertices=[(float, float, float)],
                     origin=tuple,
                     ignore_statistics=False)
            ],
                positions=[(float, float, float)],
                orientations=OnlyIf(to_type_converter(
                    [(float, float, float, float)]), allow_none=True),
                overlap=OnlyIf(to_type_converter([int]), allow_none=True),
                ignore_statistics=False,
                capacity=4,
                len_keys=1,
                _defaults={
                'orientations': None,
                'overlap': None
            }))
        self._add_typeparam(typeparam_shape)
