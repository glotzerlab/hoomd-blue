# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

from hoomd import _hoomd
from hoomd.parameterdicts import TypeParameterDict
from hoomd.parameterdicts import ParameterDict
from hoomd.typeparam import TypeParameter
from hoomd.hpmc import _hpmc
from hoomd.integrate import _BaseIntegrator
from hoomd.logger import Loggable
import hoomd
import json


# Helper method to inform about implicit depletants citation
# TODO: figure out where to call this
def cite_depletants():
    _citation = hoomd.cite.article(cite_key='glaser2015',
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
    R""" Base class HPMC integrator.

    :py:class:`_HPMCIntegrator` is the base class for all HPMC integrators. It
    provides common interface elements.  Users should not instantiate this class
    directly. Methods documented here are available to all hpmc integrators.

    .. rubric:: State data

    HPMC integrators can save and restore the following state information to gsd
    files:

        * Maximum trial move displacement *d*
        * Maximum trial rotation move *a*
        * Shape parameters for all types.

    State data are *not* written by default. You must explicitly request that
    state data for an mc integrator is written to a gsd file (see
    :py:meth:`hoomd.dump.GSD.dump_state`).

    .. code::

        mc = hoomd.hpmc.shape(...)
        gsd = hoomd.dump.gsd(...)
        gsd.dump_state(mc)

    State data are *not* restored by default. You must explicitly request that
    state data be restored when initializing the integrator.

    .. code::

        init.read_gsd(...)
        mc = hoomd.hpmc.shape(..., restore_state=True)

    See the *State data* section of the `HOOMD GSD schema
    <http://gsd.readthedocs.io/en/latest/schema-hoomd.html>`_ for details on GSD
    data chunk names and how the data are stored.
    """

    _cpp_cls = None

    def __init__(self, seed, d, a, move_ratio, nselect, deterministic):
        super().__init__()

        # Set base parameter dict for hpmc integrators
        self._param_dict = ParameterDict(dict(seed=int, move_ratio=float,
                                         nselect=int, deterministic=bool)
                                         )
        self._param_dict.update(dict(seed=int(seed),
                                     move_ratio=float(move_ratio),
                                     nselect=int(nselect),
                                     deterministic=bool(deterministic))
                                )

        # Set standard typeparameters for hpmc integrators
        typeparam_d = TypeParameter('d', type_kind='particle_types',
                                    param_dict=TypeParameterDict(float(d),
                                                                 len_keys=1)
                                    )
        typeparam_a = TypeParameter('a', type_kind='particle_types',
                                    param_dict=TypeParameterDict(float(a),
                                                                 len_keys=1)
                                    )

        typeparam_fugacity = TypeParameter('depletant_fugacity',
                                           type_kind='particle_types',
                                           param_dict=TypeParameterDict(
                                               0., len_keys=1)
                                           )

        typeparam_inter_matrix = TypeParameter('interaction_matrix',
                                               type_kind='particle_types',
                                               param_dict=TypeParameterDict(
                                                   True, len_keys=2)
                                               )

        self._extend_typeparam([typeparam_d, typeparam_a,
                                typeparam_fugacity, typeparam_inter_matrix])

    def attach(self, simulation):
        '''initialize the reflected c++ class'''
        sys_def = simulation.state._cpp_sys_def
        if simulation.device.mode == 'GPU':
            cl_c = _hoomd.CellListGPU(sys_def)
            self._cpp_obj = getattr(_hpmc,
                                    self._cpp_cls + 'GPU')(sys_def,
                                                           cl_c, self.seed)
        else:
            self._cpp_obj = getattr(_hpmc,
                                    self._cpp_cls)(sys_def, self.seed)
            cl_c = None

        super().attach(simulation)
        return [cl_c] if cl_c is not None else None

    # Set the external field
    def set_external(self, ext):
        self._cpp_obj.setExternalField(ext.cpp_compute)

    # Set the patch
    def set_PatchEnergyEvaluator(self, patch):
        self._cpp_obj.setPatchEnergy(patch.cpp_evaluator)

    # TODO need to validate somewhere that quaternions are normalized

    @property
    def type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Since this behaves differently for different types of shapes, the
        default behavior just raises an exception. Subclasses can override this
        to properly return.
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
        R""" Build an overlap map of the system

        Returns:
            List of tuples. True/false value of the i,j entry indicates
            overlap/non-overlap of the ith and jth particles (by tag)

        Note:
            :py:meth:`map_overlaps` does not support MPI parallel simulations.

        Example:
            mc = hpmc.integrate.shape(...)
            mc.shape_param.set(...)
            overlap_map = np.asarray(mc.map_overlaps())
        """

        if not self.is_attached:
            return None
        return self._cpp_obj.mapOverlaps()

    @Loggable.log
    def overlaps(self):
        R""" Count the number of overlaps.

        Returns:
            The number of overlaps in the current system configuration

        Example::

            mc = hpmc.integrate.Shape(..)
            mc.shape['A'] = dict(....)
            run(100)
            num_overlaps = mc.overlaps
        """
        if not self.is_attached:
            return None
        self._cpp_obj.communicate(True)
        return self._cpp_obj.countOverlaps(False)

    def test_overlap(self, type_i, type_j, rij, qi, qj, use_images=True,
                     exclude_self=False):
        R""" Test overlap between two particles.

        Args:
            type_i (str): Type of first particle
            type_j (str): Type of second particle
            rij (tuple): Separation vector **rj**-**ri** between the particle centers
            qi (tuple): Orientation quaternion of first particle
            qj (tuple): Orientation quaternion of second particle
            use_images (bool): If True, check for overlap between the periodic images of the particles by adding
                the image vector to the separation vector
            exclude_self (bool): If both **use_images** and **exclude_self** are true, exclude the primary image

        For two-dimensional shapes, pass the third dimension of **rij** as zero.

        Returns:
            True if the particles overlap.
        """
        self.update_forces()

        ti = hoomd.context.current.system_definition.getParticleData().getTypeByName(type_i)
        tj = hoomd.context.current.system_definition.getParticleData().getTypeByName(type_j)

        rij = hoomd.util.listify(rij)
        qi = hoomd.util.listify(qi)
        qj = hoomd.util.listify(qj)
        return self._cpp_obj.py_test_overlap(ti, tj, rij, qi, qj, use_images, exclude_self)

    @Loggable.log(flag='multi')
    def translate_moves(self):
        R""" Get the number of accepted and rejected translate moves.

        Returns:
            The number of accepted and rejected translate moves during the last
            :py:func:`hoomd.run()`.

        Example::

            mc = hpmc.integrate.Shape(..)
            mc.shape['A'] = dict(....)
            run(100)
            t_accept = mc.translate_acceptance

        """
        return self._cpp_obj.getCounters(1).translate

    @Loggable.log(flag='multi')
    def rotate_moves(self):
        R""" Get the number of accepted and reject rotation moves

        Returns:
            The number of accepted and rejected rotate moves during the last
            :py:func:`hoomd.run()`.

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param.set(....);
            run(100)
            t_accept = mc.get_rotate_acceptance();

        """
        return self._cpp_obj.getCounters(1).rotate

    @Loggable.log
    def mps(self):
        R""" Get the number of trial moves per second.

        Returns:
            The number of trial moves per second performed during the last :py:func:`hoomd.run()`.

        """
        return self._cpp_obj.getMPS()

    @property
    def counters(self):
        R""" Get all trial move counters.

        Returns:
            counter object which has ``translate``, ``rotate``,
            ``ovelap_checks``, and ``overlap_errors`` attributes. The attributes
            ``translate`` and ``rotate`` are tuples of the accepted and rejected
            respective trial move while ``overlap_checks`` and
            ``overlap_errors`` are integers.
        """
        return self._cpp_obj.getCounters(1)


class Sphere(_HPMCIntegrator):
    R""" HPMC integration for spheres (2D/3D).

    Args:
        seed (int): Random number seed
        d (float): Maximum move displacement, Scalar to set for all types, or a
            dict containing {type:size} to set by type.
        a (float): Maximum rotation move to set for all types, or a dict
            containing {type:size} to set by type.
        move_ratio (float, only used with **orientable=True**): Ratio of
            translation moves to rotation moves. (added in version 2.3)
        nselect (int): The number of trial moves to perform in each cell.

    Hard particle Monte Carlo integration method for spheres.

    Sphere parameters:

    * *diameter* (**required**) - diameter of the sphere (distance units)
    * *orientable* (**default: False**) - set to True for spheres with
      orientation (added in version 2.3)
    * *ignore_statistics* (**default: False**) - set to True to disable ignore
      for statistics tracking

    Examples::

        mc = hpmc.integrate.sphere(seed=415236, d=0.3)
        mc.shape_param.set('A', diameter=1.0)
        mc.shape_param.set('B', diameter=2.0)
        mc.shape_param.set('C', diameter=1.0, orientable=True)
        print('diameter = ', mc.shape_param['A'].diameter)

    Depletants Example::

        mc = hpmc.integrate.sphere(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=8)
        mc.shape_param.set('A', diameter=1.0)
        mc.shape_param.set('B', diameter=.1)
        mc.set_fugacity('B',fugacity=3.0)
    """
    _cpp_cls = 'IntegratorHPMCMonoSphere'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameter=float,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)

    @Loggable.log(flag='multi')
    def type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Examples:
            The types will be 'Sphere' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'Sphere', 'diameter': 1}, {'type': 'Sphere', 'diameter': 2}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super()._return_type_shapes()


class convex_polygon(_HPMCIntegrator):
    R""" HPMC integration for convex polygons (2D).

    Args:
        seed (int): Random number seed
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Note:
        For concave polygons, use :py:class:`simple_polygon`.

    Convex polygon parameters:

    * *vertices* (**required**) - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units)

        * Vertices **MUST** be specified in a *counter-clockwise* order.
        * The origin **MUST** be contained within the vertices.
        * Points inside the polygon **MUST NOT** be included.
        * The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't put the origin right next to an edge).

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.convex_polygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoConvexPolygon(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUConvexPolygon(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)

        self.initialize_shape_params()
        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(convex_polygon, self)._return_type_shapes()


class convex_spheropolygon(_HPMCIntegrator):
    R""" HPMC integration for convex spheropolygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Spheropolygon parameters:

    * *vertices* (**required**) - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units)

        * The origin **MUST** be contained within the shape.
        * The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't put the origin right next to an edge).

    * *sweep_radius* (**default: 0.0**) - the radius of the sphere swept around the edges of the polygon (distance units) - **optional**
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Useful cases:

     * A 1-vertex spheropolygon is a disk.
     * A 2-vertex spheropolygon is a spherocylinder.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.convex_spheropolygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], sweep_radius=0.1, ignore_statistics=False);
        mc.shape_param.set('A', vertices=[(0,0)], sweep_radius=0.5, ignore_statistics=True);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoSpheropolygon(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUSpheropolygon(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0.1,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(convex_spheropolygon, self)._return_type_shapes()


class simple_polygon(_HPMCIntegrator):
    R""" HPMC integration for simple polygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Note:
        For simple polygons that are not concave, use :py:class:`convex_polygon`, it will execute much faster than
        :py:class:`simple_polygon`.

    Simple polygon parameters:

    * *vertices* (**required**) - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units)

        * Vertices **MUST** be specified in a *counter-clockwise* order.
        * The polygon may be concave, but edges must not cross.
        * The origin doesn't necessarily need to be inside the shape.
        * The origin centered circle that encloses all vertices should be of minimal size for optimal performance.

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.simple_polygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(0, 0.5), (-0.5, -0.5), (0, 0), (0.5, -0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoSimplePolygon(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUSimplePolygon(
                hoomd.context.current.system_definition, cl_c, seed)

        # set parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(simple_polygon, self)._return_type_shapes()


class polyhedron(_HPMCIntegrator):
    R""" HPMC integration for general polyhedra (3D).

    This shape uses an internal OBB tree for fast collision queries.
    Depending on the number of constituent spheres in the tree, different values of the number of
    spheres per leaf node may yield different optimal performance.
    The capacity of leaf nodes is configurable.

    Only triangle meshes and spheres are supported. The mesh must be free of self-intersections.

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Polyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        * The origin **MUST** strictly be contained in the generally nonconvex volume defined by the vertices and faces
        * The (0,0,0) centered sphere that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't translate the shape such that (0,0,0) right next to a face).

    * *faces* (**required**) - a list of vertex indices for every face

        * For visualization purposes, the faces **MUST** be defined with a counterclockwise winding order to produce an outward normal.

    * *sweep_radius* (**default: 0.0**) - rounding radius applied to polyhedron
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    * *capacity* (**default: 4**) - set to the maximum number of particles per leaf node for better performance

        * .. versionadded:: 2.2

    * *origin* (**default: (0,0,0)**) - a point strictly inside the shape, needed for correctness of overlap checks

        * .. versionadded:: 2.2

    * *hull_only* (**default: True**) - if True, only consider intersections between hull polygons

        * .. versionadded:: 2.2

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
                 (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],\
        faces = [[0, 2, 6], [6, 4, 0], [5, 0, 4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], [3,6,2], \
                 [3,7,6], [3,1,5], [3,5,7]]
        print('vertices = ', mc.shape_param['A'].vertices)
        print('faces = ', mc.shape_param['A'].faces)

    Depletants Example::

        mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        cube_verts = [(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
                     (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)];
        cube_faces = [[0, 2, 6], [6, 4, 0], [5, 0, 4], [5,1,0], [5,4,6], [5,6,7], [3,2,0], [3,0,1], [3,6,2], \
                     [3,7,6], [3,1,5], [3,5,7]]
        tetra_verts = [(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)];
        tetra_faces = [[0, 1, 2], [3, 0, 2], [3, 2, 1], [3,1,0]];
        mc.shape_param.set('A', vertices = cube_verts, faces = cube_faces);
        mc.shape_param.set('B', vertices = tetra_verts, faces = tetra_faces, origin = (0,0,0));
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoPolyhedron(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUPolyhedron(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Mesh', 'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]],
              'indices': [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(polyhedron, self)._return_type_shapes()


class convex_polyhedron(_HPMCIntegrator):
    R""" HPMC integration for convex polyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): (Override the automatic choice for the number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Convex polyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        * The origin **MUST** be contained within the vertices.
        * The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't put the origin right next to a face).

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    Depletants Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        mc.shape_param.set('B', vertices=[(0.05, 0.05, 0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (-0.05, -0.05, 0.05)]);
        mc.set_fugacity('B',fugacity=3.0)
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoConvexPolyhedron(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUConvexPolyhedron(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        if nselect is not None:
            self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'ConvexPolyhedron', 'rounding_radius': 0,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(convex_polyhedron, self)._return_type_shapes()


class faceted_ellipsoid(_HPMCIntegrator):
    R""" HPMC integration for faceted ellipsoids (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    A faceted ellipsoid is an ellipsoid intersected with a convex polyhedron defined through
    halfspaces. The equation defining each halfspace is given by:

    .. math::
        n_i\cdot r + b_i \le 0

    where :math:`n_i` is the face normal, and :math:`b_i` is  the offset.

    Warning:
        The origin must be chosen so as to lie **inside the shape**, or the overlap check will not work.
        This condition is not checked.

    Faceted ellipsoid parameters:

    * *normals* (**required**) - list of (x,y,z) tuples defining the facet normals (distance units)
    * *offsets* (**required**) - list of offsets (distance unit^2)
    * *a* (**required**) - first half axis of ellipsoid
    * *b* (**required**) - second half axis of ellipsoid
    * *c* (**required**) - third half axis of ellipsoid
    * *vertices* (**required**) - list of vertices for intersection polyhedron
    * *origin* (**required**) - origin vector
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        Planes must not be coplanar.

    Note:
        The half-space intersection of the normals has to match the convex polyhedron defined by
        the vertices (if non-empty), currently the half-space intersection is **not** calculated automatically.
        For simple intersections with planes that do not intersect within the sphere, the vertices
        list can be left empty.

    Example::

        mc = hpmc.integrate.faceted_ellipsoid(seed=415236, d=0.3, a=0.4)

        # half-space intersection
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5)

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],[-.1,-.5,.5],[-.1,.5,.5],[-.1,.5,-.5], [1,-.5,-.5],[1,-.5,.5],[1,.5,.5],[1,.5,-.5]]

        mc.shape_param.set('A', normals=slab_normals, offsets=slab_offsets, vertices=slab_verts,a=1.0, b=0.5, c=0.5);
        print('a = {}, b = {}, c = {}', mc.shape_param['A'].a,mc.shape_param['A'].b,mc.shape_param['A'].c)

    Depletants Example::

        mc = hpmc.integrate.faceted_ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],a=1.0, b=0.5, c=0.25);
        # depletant sphere
        mc.shape_param.set('B', normals=[],a=0.1,b=0.1,c=0.1);
        mc.set_fugacity('B',fugacity=3.0)
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoFacetedEllipsoid(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUFacetedEllipsoid(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()


class faceted_sphere(faceted_ellipsoid):
    R""" HPMC integration for faceted spheres (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    A faceted sphere is a sphere intersected with halfspaces. The equation defining each halfspace is given by:

    .. math::
        n_i\cdot r + b_i \le 0

    where :math:`n_i` is the face normal, and :math:`b_i` is  the offset.

    Warning:
        The origin must be chosen so as to lie **inside the shape**, or the overlap check will not work.
        This condition is not checked.

    Faceted sphere parameters:

    * *normals* (**required**) - list of (x,y,z) tuples defining the facet normals (distance units)
    * *offsets* (**required**) - list of offsets (distance unit^2)
    * *diameter* (**required**) - diameter of sphere
    * *vertices* (**required**) - list of vertices for intersection polyhedron
    * *origin* (**required**) - origin vector
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        Planes must not be coplanar.

    Note:
        The half-space intersection of the normals has to match the convex polyhedron defined by
        the vertices (if non-empty), currently the half-space intersection is **not** calculated automatically.
        For simple intersections with planes that do not intersect within the sphere, the vertices
        list can be left empty.

    Example::
        # half-space intersection
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5)

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],[-.1,-.5,.5],[-.1,.5,.5],[-.1,.5,-.5], [.5,-.5,-.5],[.5,-.5,.5],[.5,.5,.5],[.5,.5,-.5]]

        mc = hpmc.integrate.faceted_sphere(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', normals=slab_normals,offsets=slab_offsets, vertices=slab_verts,diameter=1.0);
        print('diameter = ', mc.shape_param['A'].diameter)

    Depletants Example::

        mc = hpmc.integrate.faceted_sphere(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
        mc.shape_param.set('B', normals=[],diameter=0.1);
        mc.set_fugacity('B',fugacity=3.0)
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        super(faceted_sphere, self).__init__(seed=seed, d=d, a=a, move_ratio=move_ratio,
                                             nselect=nselect, restore_state=restore_state)


class sphinx(_HPMCIntegrator):
    R""" HPMC integration for sphinx particles (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Sphinx particles are dimpled spheres (spheres with 'positive' and 'negative' volumes).

    Sphinx parameters:

    * *diameters* - diameters of spheres (positive OR negative real numbers)
    * *centers* - centers of spheres in local coordinate frame
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Quick Example::

        mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
        print('diameters = ', mc.shape_param['A'].diameters)

    Depletants Example::

        mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,-.25])
        mc.shape_param.set('B', centers=[(0,0,0)], diameters=[.15])
        mc.set_fugacity('B',fugacity=3.0)
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoSphinx(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")

            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUSphinx(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        if nselect is not None:
            self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()


class convex_spheropolyhedron(_HPMCIntegrator):
    R""" HPMC integration for spheropolyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    A spheropolyhedron can also represent spheres (0 or 1 vertices), and spherocylinders (2 vertices).

    Spheropolyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        - The origin **MUST** be contained within the vertices.
        - The origin centered sphere that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't put the origin right next to a face).
        - A sphere can be represented by specifying zero vertices (i.e. vertices=[]) and a non-zero radius R
        - Two vertices and a non-zero radius R define a prolate spherocylinder.

    * *sweep_radius* (**default: 0.0**) - the radius of the sphere swept around the edges of the polygon (distance units) - **optional**
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)
        mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1, ignore_statistics=True);

    Depletants example::

        mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1);
        mc.set_fugacity('B',fugacity=3.0)
    """

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoSpheropolyhedron(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUSpheropolyhedron(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        if nselect is not None:
            self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'ConvexPolyhedron', 'rounding_radius': 0.1,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(convex_spheropolyhedron, self)._return_type_shapes()


class ellipsoid(_HPMCIntegrator):
    R""" HPMC integration for ellipsoids (2D/3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Ellipsoid parameters:

    * *a* (**required**) - principle axis a of the ellipsoid (radius in the x direction) (distance units)
    * *b* (**required**) - principle axis b of the ellipsoid (radius in the y direction) (distance units)
    * *c* (**required**) - principle axis c of the ellipsoid (radius in the z direction) (distance units)
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking

    Example::

        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        print('ellipsoids parameters (a,b,c) = ', mc.shape_param['A'].a, mc.shape_param['A'].b, mc.shape_param['A'].c)

    Depletants Example::

        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        mc.shape_param.set('B', a=0.05, b=0.05, c=0.05);
        mc.set_fugacity('B',fugacity=3.0)
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoEllipsoid(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUEllipsoid(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)

        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:

            >>> mc.get_type_shapes()
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.5, 'c': 1}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(ellipsoid, self)._return_type_shapes()


class sphere_union(_HPMCIntegrator):
    R""" HPMC integration for unions of spheres (3D).

    This shape uses an internal OBB tree for fast collision queries.
    Depending on the number of constituent spheres in the tree, different values of the number of
    spheres per leaf node may yield different optimal performance.
    The capacity of leaf nodes is configurable.

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        capacity (int): Set to the number of constituent spheres per leaf node. (added in version 2.2)
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`_HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)

    Sphere union parameters:

    * *diameters* (**required**) - list of diameters of the spheres (distance units).
    * *centers* (**required**) - list of centers of constituent spheres in particle coordinates.
    * *overlap* (**default: 1 for all spheres**) - only check overlap between constituent particles for which *overlap [i] & overlap[j]* is !=0, where '&' is the bitwise AND operator.

        * .. versionadded:: 2.1

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking.
    * *capacity* (**default: 4**) - set to the maximum number of particles per leaf node for better performance
        * .. versionadded:: 2.2

    Example::

        mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
        print('diameter of the first sphere = ', mc.shape_param['A'].members[0].diameter)
        print('center of the first sphere = ', mc.shape_param['A'].centers[0])

    Depletants Example::

        mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
        mc.shape_param.set('B', diameters=[0.05], centers=[(0.0, 0.0, 0.0)]);
        mc.set_fugacity('B',fugacity=3.0)
    """

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, restore_state=False):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoSphereUnion(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUSphereUnion(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        if restore_state:
            self.restore_state()


class convex_spheropolyhedron_union(_HPMCIntegrator):
    R""" HPMC integration for unions of convex polyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        capacity (int): Set to the number of constituent convex polyhedra per leaf node

    .. versionadded:: 2.2

    Convex polyhedron union parameters:

    * *vertices* (**required**) - list of vertex lists of the polyhedra in particle coordinates.
    * *centers* (**required**) - list of centers of constituent polyhedra in particle coordinates.
    * *orientations* (**required**) - list of orientations of constituent polyhedra.
    * *overlap* (**default: 1 for all particles**) - only check overlap between constituent particles for which *overlap [i] & overlap[j]* is !=0, where '&' is the bitwise AND operator.
    * *sweep_radii* (**default: 0 for all particle**) - radii of spheres sweeping out each constituent polyhedron

        * .. versionadded:: 2.4

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking.

    Example::

        mc = hpmc.integrate.convex_spheropolyhedron_union(seed=27, d=0.3, a=0.4)
        cube_verts = [[-1,-1,-1],[-1,-1,1],[-1,1,1],[-1,1,-1],
                     [1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1]]
        mc.shape_param.set('A', vertices=[cube_verts, cube_verts],
                                centers=[[-1,0,0],[1,0,0]],orientations=[[1,0,0,0],[1,0,0,0]]);
        print('vertices of the first cube = ', mc.shape_param['A'].members[0].vertices)
        print('center of the first cube = ', mc.shape_param['A'].centers[0])
        print('orientation of the first cube = ', mc.shape_param['A'].orientations[0])
    """

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoConvexPolyhedronUnion(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUConvexPolyhedronUnion(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        # meta data
        self.metadata_fields = ['capacity']


class faceted_ellipsoid_union(_HPMCIntegrator):
    R""" HPMC integration for unions of faceted ellipsoids (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        capacity (int): Set to the number of constituent convex polyhedra per leaf node

    .. versionadded:: 2.5

    See :py:class:`faceted_ellipsoid` for a detailed explanation of the constituent particle parameters.

    Faceted ellipsiod union parameters:

    * *normals* (**required**) - list of list of (x,y,z) tuples defining the facet normals (distance units)
    * *offsets* (**required**) - list of list of offsets (distance unit^2)
    * *axes* (**required**) - list of half axes, tuple of three per constituent ellipsoid
    * *vertices* (**required**) - list of list list of vertices for intersection polyhedron
    * *origin* (**required**) - list of origin vectors

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking.

    Example::

        mc = hpmc.integrate.faceted_ellipsoid_union(seed=27, d=0.3, a=0.4)

        # make a prolate Janus ellipsoid
        # cut away -x halfspace
        normals = [(-1,0,0)]
        offsets = [0]

        mc.shape_param.set('A', normals=[normals, normals],
                                offsets=[offsets, offsets],
                                vertices=[[], []],
                                axes=[(.5,.5,2),(.5,.5,2)],
                                centers=[[0,0,0],[0,0,0]],
                                orientations=[[1,0,0,0],[0,0,0,-1]]);

        print('offsets of the first faceted ellipsoid = ', mc.shape_param['A'].members[0].normals)
        print('normals of the first faceted ellispoid = ', mc.shape_param['A'].members[0].offsets)
        print('vertices of the first faceted ellipsoid = ', mc.shape_param['A'].members[0].vertices)
    """

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4):

        # initialize base class
        _HPMCIntegrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self._cpp_obj = _hpmc.IntegratorHPMCMonoFacetedEllipsoidUnion(
                hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.overwriteCompute(cl_c, "auto_cl2")
            self._cpp_obj = _hpmc.IntegratorHPMCMonoGPUFacetedEllipsoidUnion(
                hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self._cpp_obj, d)
        setA(self._cpp_obj, a)
        self._cpp_obj.setMoveRatio(move_ratio)
        self._cpp_obj.setNSelect(nselect)

        hoomd.context.current.system.setIntegrator(self._cpp_obj)
        self.initialize_shape_params()

        # meta data
        self.metadata_fields = ['capacity']
