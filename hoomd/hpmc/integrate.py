# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.parameterdicts import TypeParameterDict, AttachedTypeParameterDict
from hoomd.parameterdicts import RequiredArg
from hoomd.typeparam import TypeParameter
from hoomd.hpmc import _hpmc
from hoomd.integrate import _integrator
import hoomd
import sys
import json


class interaction_matrix:
    R""" Define pairwise interaction matrix

    All shapes use :py:class:`interaction_matrix` to define the interaction matrix between different
    pairs of particles indexed by type. The set of pair coefficients is a symmetric
    matrix defined over all possible pairs of particle types.

    By default, all elements of the interaction matrix are 1, that means that overlaps
    are checked between all pairs of types. To disable overlap checking for a specific
    type pair, set the coefficient for that pair to 0.

    Access the interaction matrix with a saved integrator object like so::

        from hoomd import hpmc

        mc = hpmc.integrate.some_shape(arguments...)
        mv.overlap_checks.set('A', 'A', enable=False)
        mc.overlap_checks.set('A', 'B', enable=True)
        mc.overlap_checks.set('B', 'B', enable=False)

    .. versionadded:: 2.1
    """

    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};

    ## \internal
    # \brief Return a compact representation of the pair coefficients
    def get_metadata(self):
        # return list for easy serialization
        l = []
        for (a,b) in self.values:
            item = dict()
            item['typei'] = a
            item['typej'] = b
            item['enable'] = self.values[(a,b)]
            l.append(item)
        return l

    ## \var values
    # \internal
    # \brief Contains the matrix of set values in a dictionary

    def set(self, a, b, enable):
        R""" Sets parameters for one type pair.

        Args:
            a (str): First particle type in the pair (or a list of type names)
            b (str): Second particle type in the pair (or a list of type names)
            enable: Set to True to enable overlap checks for this pair, False otherwise

        By default, all interaction matrix elements are set to 'True'.

        It is not an error, to specify matrix elements for particle types that do not exist in the simulation.

        There is no need to specify matrix elements for both pairs 'A', 'B' and 'B', 'A'. Specifying
        only one is sufficient.

        To set the same elements between many particle types, provide a list of type names instead of a single
        one. All pairs between the two lists will be set to the same parameters.

        Examples::

            mc.overlap_checks.set('A', 'A', False);
            mc.overlap_checks.set('B', 'B', False);
            mc.overlap_checks.set('A', 'B', True);
            mc.overlap_checks.set(['A', 'B', 'C', 'D'], 'F', True);
            mc.overlap_checks.set(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], False);


        """

        # listify the inputs
        a = hoomd.util.listify(a)
        b = hoomd.util.listify(b)

        for ai in a:
            for bi in b:
                self.set_single(ai, bi, enable);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, a, b, enable):
        a = str(a);
        b = str(b);

        # create the pair if it hasn't been created it
        if (not (a,b) in self.values) and (not (b,a) in self.values):
            self.values[(a,b)] = bool(enable);
        else:
            # Find the pair to update
            if (a,b) in self.values:
                cur_pair = (a,b);
            elif (b,a) in self.values:
                cur_pair = (b,a);
            else:
                hoomd.context.current.device.cpp_msg.error("Bug detected in integrate.interaction_matrix(). Please report\n");
                raise RuntimeError("Error setting matrix elements");

            self.values[cur_pair] = bool(enable)

    ## \internal
    # \brief Try to get a single pair coefficient
    # \detail
    # \param a First name in the type pair
    # \param b Second name in the type pair
    def get(self,a,b):
        if (a,b) in self.values:
            cur_pair = (a,b);
        elif (b,a) in self.values:
            cur_pair = (b,a);
        else:
            return None

        return self.values[cur_pair];

# Helper method to inform about implicit depletants citation
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


class HPMCIntegrator(_integrator):
    R""" Base class HPMC integrator.

    :py:class:`HPMCIntegrator` is the base class for all HPMC integrators. It
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
        self._param_dict.update(dict(seed=seed, move_ratio=move_ratio,
                                     nselect=nselect,
                                     deterministic=deterministic)
                                )

        # Set standard typeparameters for hpmc integrators
        typeparam_d = TypeParameter('d', type_kind='particle_types',
                                    param_dict=TypeParameterDict(d, len_keys=1)
                                    )
        typeparam_a = TypeParameter('a', type_kind='particle_types',
                                    param_dict=TypeParameterDict(a, len_keys=1)
                                    )
        typeparam_fugacity = TypeParameter('depletant_fugacity',
                                           type_kind='particle_types',
                                           param_dict=TypeParameterDict(
                                               float(0), len_keys=1)
                                           )

        self._extend_typeparam([typeparam_d, typeparam_a, typeparam_fugacity])

        self.overlap_checks = interaction_matrix()


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

        # set the non type specfic parameters
        self._apply_param_dict()

        # Deal with type specific properties
        self._apply_typeparam_dict(self._cpp_obj, simulation)

        return [cl_c] if cl_c is not None else None

    # Set the external field
    def set_external(self, ext):
        self._cpp_obj.setExternalField(ext.cpp_compute)

    # Set the patch
    def set_PatchEnergyEvaluator(self, patch):
        self._cpp_obj.setPatchEnergy(patch.cpp_evaluator)

    # TODO need to validate somewhere that quaternions are normalized

    # Declare the GSD state schema.
    @classmethod
    def _gsd_state_name(cls):
        return "state/hpmc/" + str(cls.__name__) + "/"

    def restore_state(self):
        super(HPMCIntegrator, self).restore_state()

        # if restore state succeeds, all shape information is set
        # set the python level is_set flags to notify this
        for type in self.shape_param.keys():
            self.shape_param[type].is_set = True

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Since this behaves differently for different types of shapes, the
        default behavior just raises an exception. Subclasses can override this
        to properly return.
        """
        raise NotImplementedError(
            "You are using a shape type that is not implemented! "
            "If you want it, please modify the "
            "hoomd.hpmc.integrate.HPMCIntegrator.get_type_shapes function.")

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret

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

        self.update_forces()
        N = hoomd.context.current.system_definition.getParticleData().getMaximumTag() + 1
        overlap_map = self._cpp_obj.mapOverlaps()
        return list(zip(*[iter(overlap_map)] * N))

    def count_overlaps(self):
        R""" Count the number of overlaps.

        Returns:
            The number of overlaps in the current system configuration

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param.set(....);
            run(100)
            num_overlaps = mc.count_overlaps();
        """
        self.update_forces()
        self._cpp_obj.communicate(True)
        return self._cpp_obj.countOverlaps(hoomd.context.current.system.getCurrentTimeStep(), False)

    def test_overlap(self, type_i, type_j, rij, qi, qj, use_images=True, exclude_self=False):
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

    def get_translate_acceptance(self):
        R""" Get the average acceptance ratio for translate moves.

        Returns:
            The average translate accept ratio during the last :py:func:`hoomd.run()`.

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param.set(....);
            run(100)
            t_accept = mc.get_translate_acceptance();

        """
        counters = self._cpp_obj.getCounters(1)
        return counters.getTranslateAcceptance()

    def get_rotate_acceptance(self):
        R""" Get the average acceptance ratio for rotate moves.

        Returns:
            The average rotate accept ratio during the last :py:func:`hoomd.run()`.

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param.set(....);
            run(100)
            t_accept = mc.get_rotate_acceptance();

        """
        counters = self._cpp_obj.getCounters(1)
        return counters.getRotateAcceptance()

    def get_mps(self):
        R""" Get the number of trial moves per second.

        Returns:
            The number of trial moves per second performed during the last :py:func:`hoomd.run()`.

        """
        return self._cpp_obj.getMPS()

    def get_counters(self):
        R""" Get all trial move counters.

        Returns:
            A dictionary containing all trial moves counted during the last :py:func:`hoomd.run()`.

        The dictionary contains the entries:

        * *translate_accept_count* - count of the number of accepted translate moves
        * *translate_reject_count* - count of the number of rejected translate moves
        * *rotate_accept_count* - count of the number of accepted rotate moves
        * *rotate_reject_count* - count of the number of rejected rotate moves
        * *overlap_checks* - estimate of the number of overlap checks performed
        * *translate_acceptance* - Average translate acceptance ratio over the run
        * *rotate_acceptance* - Average rotate acceptance ratio over the run
        * *move_count* - Count of the number of trial moves during the run
        """
        counters = self._cpp_obj.getCounters(1)
        return dict(translate_accept_count=counters.translate_accept_count,
                    translate_reject_count=counters.translate_reject_count,
                    rotate_accept_count=counters.rotate_accept_count,
                    rotate_reject_count=counters.rotate_reject_count,
                    overlap_checks=counters.overlap_checks,
                    translate_acceptance=counters.getTranslateAcceptance(),
                    rotate_acceptance=counters.getRotateAcceptance(),
                    move_count=counters.getNMoves())


class Sphere(HPMCIntegrator):
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
                                            diameter=RequiredArg,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Examples:
            The types will be 'Sphere' regardless of dimensionality.

            >>> mc.get_type_shapes()
            [{'type': 'Sphere', 'diameter': 1}, {'type': 'Sphere', 'diameter': 2}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super()._return_type_shapes()


class ConvexPolygon(HPMCIntegrator):
    R""" HPMC integration for convex polygons (2D).

    Args:
        seed (int): Random number seed
        d (float): Maximum move displacement, Scalar to set for all types, or a 
          dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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
    _cpp_cls = 'IntegratorHPMCMonoConvexPolygon'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            ignore_statistics=False,
                                            sweep_radius=0,
                                            len_keys=1)
                                        )

        self._add_typeparam(typeparam_shape)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(ConvexPolygon, self)._return_type_shapes()


class ConvexSpheropolygon(HPMCIntegrator):
    R""" HPMC integration for convex spheropolygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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
    _cpp_cls = 'IntegratorHPMCMonoSpheropolygon'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            sweep_radius=RequiredArg,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )

        self._add_typeparam(typeparam_shape)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0.1,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(ConvexSpheropolygon, self)._return_type_shapes()


class SimplePolygon(HPMCIntegrator):
    R""" HPMC integration for simple polygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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

    _cpp_cls = 'IntegratorHPMCMonoSimplePolygon'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            ignore_statistics=False,
                                            sweep_radius=0,
                                            len_keys=1)
                                        )

        self._add_typeparam(typeparam_shape)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Polygon', 'rounding_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(SimplePolygon, self)._return_type_shapes()


class Polyhedron(HPMCIntegrator):
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
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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

    _cpp_cls = 'IntegratorHPMCMonoPolyhedron'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            faces=RequiredArg,
                                            sweep_radius=0.0,
                                            capacity=4,
                                            origin=(0, 0, 0),
                                            hull_only=True,
                                            overlap=None,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )

        self._add_typeparam(typeparam_shape)

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Mesh', 'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]],
              'indices': [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super(Polyhedron, self)._return_type_shapes()


class ConvexPolyhedron(HPMCIntegrator):
    R""" HPMC integration for convex polyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): (Override the automatic choice for the number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedron'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)

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
        return super(ConvexPolyhedron, self)._return_type_shapes()



class FacetedEllipsoid(HPMCIntegrator):
    R""" HPMC integration for faceted ellipsoids (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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
    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoid'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            normals=RequiredArg,
                                            offsets=RequiredArg,
                                            a=RequiredArg,
                                            b=RequiredArg,
                                            c=RequiredArg,
                                            vertices=RequiredArg,
                                            origin=RequiredArg,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)



class FacetedSphere(FacetedEllipsoid):
    R""" HPMC integration for faceted spheres (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, deterministic=False):

        super(FacetedSphere, self).__init__(seed=seed, d=d, a=a, move_ratio=move_ratio,
                                             nselect=nselect, deterministic=deterministic)


class Sphinx(HPMCIntegrator):
    R""" HPMC integration for sphinx particles (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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
    _cpp_cls = 'IntegratorHPMCMonoSphinx'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameters=RequiredArg,
                                            centers=RequiredArg,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)


class ConvexSpheropolyhedron(HPMCIntegrator):
    R""" HPMC integration for spheropolyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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

    _cpp_cls = 'IntegratorHPMCMonoSpheropolyhedron'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=RequiredArg,
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)


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
        return super(ConvexSpheropolyhedron, self)._return_type_shapes()


class Ellipsoid(HPMCIntegrator):
    R""" HPMC integration for ellipsoids (2D/3D).
​
    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
                             for a description of what state data restored. (added in version 2.2)
​
    Ellipsoid parameters:
​
    * *a* (**required**) - principle axis a of the ellipsoid (radius in the x direction) (distance units)
    * *b* (**required**) - principle axis b of the ellipsoid (radius in the y direction) (distance units)
    * *c* (**required**) - principle axis c of the ellipsoid (radius in the z direction) (distance units)
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
​
    Example::
​
        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        print('ellipsoids parameters (a,b,c) = ', mc.shape_param['A'].a, mc.shape_param['A'].b, mc.shape_param['A'].c)
​
    Depletants Example::
​
        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.set_param(nselect=1)
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        mc.shape_param.set('B', a=0.05, b=0.05, c=0.05);
        mc.set_fugacity('B',fugacity=3.0)
    """
    _cpp_cls = 'IntegratorHPMCMonoEllipsoid'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            a=RequiredArg,
                                            b=RequiredArg,
                                            c=RequiredArg,
                                            ignore_statistics=False,
                                            len_keys=1)
                                    )

        self._extend_typeparam([typeparam_shape])

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.
        Example:
            >>> mc.get_type_shapes()
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.5, 'c': 1}]
        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        return super()._return_type_shapes()


class SphereUnion(HPMCIntegrator):
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
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`HPMCIntegrator`
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

    _cpp_cls = 'IntegratorHPMCMonoSphereUnion'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            shapes=RequiredArg,
                                            orientations=RequiredArg,
                                            positions=RequiredArg,
                                            capacity=4,
                                            overlap=1,
                                            ignore_statistics=False,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)


class ConvexSpheropolyhedronUnion(HPMCIntegrator):
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

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedronUnion'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            shapes=RequiredArg,
                                            positions=RequiredArg,
                                            orientations=RequiredArg,
                                            overlap=1,
                                            ignore_statistics=False,
                                            capacity=4,
                                            len_keys=1)
                                        )
        self._add_typeparam(typeparam_shape)
        # meta data
        self.metadata_fields = ['capacity']


class FacetedEllipsoidUnion(HPMCIntegrator):
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

    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoidUnion'

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,
                 nselect=4, deterministic=False):

        # initialize base class
        super().__init__(seed, d, a, move_ratio, nselect, deterministic)

        typeparam_shape = TypeParameter('shape', type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            shapes=RequiredArg,
                                            positions=RequiredArg,
                                            orientations=RequiredArg,
                                            overlap=1,
                                            ignore_statistics=False,
                                            capacity=4,
                                            len_keys=1)
                                            )
        self._add_typeparam(typeparam_shape)
