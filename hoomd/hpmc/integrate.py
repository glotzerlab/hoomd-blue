# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.hpmc import _hpmc
from hoomd.hpmc import data
from hoomd.integrate import _integrator
import hoomd
import sys

def _get_sized_entry(base, max_n):
    """ Get a sized entry.

    HPMC has some classes and functions templated by maximum size. This convenience function returns a class given a
    base name and a maximum value for n.

    Args:
        base (string): Base name of the class.
        max_n (int): Maximum size needed

    Returns:
        The selected class with a maximum n greater than or equal to *max_n*.
    """

    # inspect _hpmc.__dict__ for base class name + integer suffix
    sizes=[]
    for cls in _hpmc.__dict__:
        if cls.startswith(base):
            # append only suffixes that convert to ints
            try:
                sizes.append(int(cls.split(base)[1]))
            except:
                pass
    sizes = sorted(sizes)

    if max_n > sizes[-1]:
        raise ValueError("Maximum value must be less than or equal to {0}".format(sizes[-1]));

    # Find the smallest size that fits size
    for size in sizes:
        if max_n <= size:
            selected_size = size;
            break;

    return _hpmc.__dict__["{0}{1}".format(base, size)];

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

            interaction_matrix.set('A', 'A', False);
            interaction_matrix.set('B', 'B', False);
            interaction_matrix.set('A', 'B', True);
            interaction_matrix.set(['A', 'B', 'C', 'D'], 'F', True);
            interaction_matrix.set(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], False);


        """
        hoomd.util.print_status_line();

        # listify the inputs
        if isinstance(a, str):
            a = [a];
        if isinstance(b, str):
            b = [b];

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
                hoomd.context.msg.error("Bug detected in integrate.interaction_matrix(). Please report\n");
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


class mode_hpmc(_integrator):
    R""" Base class HPMC integrator.

    :py:class:`mode_hpmc` is the base class for all HPMC integrators. It provides common interface elements.
    Users should not instantiate this class directly. Methods documented here are available to all hpmc
    integrators.
    """

    ## \internal
    # \brief Initialize an empty integrator
    #
    # \post the member shape_param is created
    def __init__(self, implicit):
        _integrator.__init__(self);
        self.implicit=implicit

        # setup the shape parameters
        self.shape_param = data.param_dict(self); # must call initialize_shape_params() after the cpp_integrator is created.

        # setup interaction matrix
        self.overlap_checks = interaction_matrix()

        #initialize list to check implicit params
        if self.implicit:
            self.implicit_params=list()

    ## Set the external field
    def set_external(self, ext):
        self.cpp_integrator.setExternalField(ext.cpp_compute);

    def get_metadata(self):
        data = super(mode_hpmc, self).get_metadata()
        data['d'] = self.get_d()
        data['a'] = self.get_a()
        data['move_ratio'] = self.get_move_ratio()
        data['nselect'] = self.get_nselect()
        shape_dict = {};
        for key in self.shape_param.keys():
            shape_dict[key] = self.shape_param[key].get_metadata();
        data['shape_param'] = shape_dict;
        data['overlap_checks'] = self.overlap_checks.get_metadata()
        return data

    ## \internal
    # \brief Updates the integrators in the reflected c++ class
    #
    # hpmc doesn't use forces, but we use the method to update shape parameters
    def update_forces(self):
        self.check_initialization();

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_names = [];
        for i in range(0,ntypes):
            type_names.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));
        # make sure all params have been set at least once.
        for name in type_names:
            # build a dict of the params to pass to proces_param
            if not self.shape_param[name].is_set:
                hoomd.context.msg.error("Particle type {} has not been set!\n".format(name));
                raise RuntimeError("Error running integrator");

        # backwards compatibility
        if not hasattr(self,'has_printed_warning'):
            self.has_printed_warning = False

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_names = [ hoomd.context.current.system_definition.getParticleData().getNameByType(i) for i in range(0,ntypes) ];
        first_warning = False
        for (i,type_i) in enumerate(type_names):
            if hasattr(self.shape_param[type_i],'ignore_overlaps') and self.shape_param[type_i].ignore_overlaps is not None:
                if not self.has_printed_warning and not first_warning:
                    hoomd.context.msg.warning("ignore_overlaps is deprecated. Use mc.overlap_checks.set() instead.\n")
                    first_warning = True
                for (j, type_j) in enumerate(type_names):
                    if hasattr(self.shape_param[type_j],'ignore_overlaps') and self.shape_param[type_j].ignore_overlaps is not None:
                        enable = not (self.shape_param[type_i].ignore_overlaps and self.shape_param[type_j].ignore_overlaps)
                        if not self.has_printed_warning:
                            hoomd.context.msg.warning("Setting overlap checks for type pair ({}, {}) to {}\n".format(type_i,type_j, enable))

                        hoomd.util.quiet_status()
                        self.overlap_checks.set(type_i, type_j, enable)
                        hoomd.util.unquiet_status()

        self.has_printed_warning = True

        # setup new interaction matrix elements to default
        for i in range(0,ntypes):
            type_name_i = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            for j in range(0,ntypes):
                type_name_j = hoomd.context.current.system_definition.getParticleData().getNameByType(j);
                if self.overlap_checks.get(type_name_i, type_name_j) is None: # only add new pairs
                    # by default, perform overlap checks
                    hoomd.util.quiet_status()
                    self.overlap_checks.set(str(type_name_i), str(type_name_j), True)
                    hoomd.util.unquiet_status()

        # set overlap matrix on C++ side
        for (i,type_i) in enumerate(type_names):
            for (j,type_j) in enumerate(type_names):
                check = self.overlap_checks.get(type_i, type_j)
                if check is None:
                    hoomd.context.msg.error("Interaction matrix element ({},{}) not set!\n".format(type_i, type_j))
                    raise RuntimeError("Error running integrator");
                self.cpp_integrator.setOverlapChecks(i,j,check)

        # check that particle orientations are normalized
        if not self.cpp_integrator.checkParticleOrientations():
           hoomd.context.msg.warning("Particle orientations are not normalized\n");

        #make sure all the required parameters for implicit depletant simulations have been supplied
        if self.implicit:
            self.check_implicit_params()

    def setup_pos_writer(self, pos, colors={}):
        R""" Set pos_writer definitions for specified shape parameters.

        Args:
            pos (:py:class:`hoomd.deprecated.dump.pos`): pos writer to setup
            colors (dict): dictionary of type name to color mappings

        :py:meth:`setup_pos_writer` uses the shape_param settings to specify the shape definitions (via set_def)
        to the provided pos file writer. This overrides any previous values specified to
        :py:meth:`hoomd.deprecated.dump.pos.set_def`.

        *colors* allows you to set per-type colors for particles. Specify colors as strings in the injavis format. When
        colors is not specified for a type, all colors default to ``005984FF``.

        Examples::

            mc = hpmc.integrate.shape(...);
            mc.shape_param.set(....);
            pos = pos_writer.dumpy.pos("dump.pos", period=100);
            mc.setup_pos_writer(pos, colors=dict(A='005984FF'));
        """
        self.check_initialization();

        # param_list = self.required_params;
        # # check that the force parameters are valid
        # if not self.shape_param.verify(param_list):
        #    hoomd.context.msg.error("Not all shape parameters are set\n");
        #    raise RuntimeError("Error setting up pos writer");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the params to pass to proces_param
            # param_dict = {};
            # for name in param_list:
            #     param_dict[name] = self.shape_param.get(type_list[i], name);

            color = colors.setdefault(type_list[i], '005984FF');
            shapedef = self.format_param_pos(self.shape_param[type_list[i]]);
            pos.set_def(type_list[i], shapedef + ' ' + color)

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation

        Since this behaves differently for different types of shapes, the default behavior just raises an exception. Subclasses can override this to properly return
        """
        raise NotImplementedError("You are using a shape type that is not implemented! If you want it, please modify the hoomd.hpmc.integrate.mode_hpmc.get_type_shapes function")

    def initialize_shape_params(self):
        shape_param_type = None;
        # have to have a few extra checks becuase the sized class don't actually exist yet.
        if isinstance(self, convex_polyhedron):
            shape_param_type = data.convex_polyhedron_params.get_sized_class(self.max_verts);
        elif isinstance(self, convex_spheropolyhedron):
            shape_param_type = data.convex_spheropolyhedron_params.get_sized_class(self.max_verts);
        elif isinstance(self, sphere_union):
            shape_param_type = data.sphere_union_params.get_sized_class(self.max_members);
        else:
            shape_param_type = data.__dict__[self.__class__.__name__ + "_params"]; # using the naming convention for convenience.

        # setup the coefficient options
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        for i in range(0,ntypes):
            type_name = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            if not type_name in self.shape_param.keys(): # only add new keys
                self.shape_param.update({ type_name: shape_param_type(self, i) });

        # setup the interaction matrix elements
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        for i in range(0,ntypes):
            type_name_i = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            for j in range(0,ntypes):
                type_name_j = hoomd.context.current.system_definition.getParticleData().getNameByType(j);
                if self.overlap_checks.get(type_name_i, type_name_j) is None: # only add new pairs
                    # by default, perform overlap checks
                    hoomd.util.quiet_status()
                    self.overlap_checks.set(type_name_i, type_name_j, True)
                    hoomd.util.unquiet_status()

    def set_params(self,
                   d=None,
                   a=None,
                   move_ratio=None,
                   nselect=None,
                   nR=None,
                   depletant_type=None,
                   ntrial=None):
        R""" Changes parameters of an existing integration mode.

        Args:
            d (float): (if set) Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
            a (float): (if set) Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
            move_ratio (float): (if set) New value for the move ratio.
            nselect (int): (if set) New value for the number of particles to select for trial moves in one cell.
            nR (int): (if set) **Implicit depletants only**: Number density of implicit depletants in free volume.
            depletant_type (str): (if set) **Implicit depletants only**: Particle type to use as implicit depletant.
            ntrial (int): (if set) **Implicit depletants only**: Number of re-insertion attempts per overlapping depletant.
        """

        hoomd.util.print_status_line();
        # check that proper initialization has occured
        if self.cpp_integrator == None:
            hoomd.context.msg.error("Bug in hoomd_script: cpp_integrator not set, please report\n");
            raise RuntimeError('Error updating forces');

        # change the parameters
        if d is not None:
            if isinstance(d, dict):
                for t,t_d in d.items():
                    self.cpp_integrator.setD(t_d,hoomd.context.current.system_definition.getParticleData().getTypeByName(t))
            else:
                for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes()):
                    self.cpp_integrator.setD(d,i);

        if a is not None:
            if isinstance(a, dict):
                for t,t_a in a.items():
                    self.cpp_integrator.setA(t_a,hoomd.context.current.system_definition.getParticleData().getTypeByName(t))
            else:
                for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes()):
                    self.cpp_integrator.setA(a,i);

        if move_ratio is not None:
            self.cpp_integrator.setMoveRatio(move_ratio);

        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        if self.implicit:
            if nR is not None:
                self.implicit_params.append('nR')
                self.cpp_integrator.setDepletantDensity(nR)
            if depletant_type is not None:
                self.implicit_params.append('depletant_type')
                itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(depletant_type)
                self.cpp_integrator.setDepletantType(itype)
            if ntrial is not None:
                self.implicit_params.append('ntrial')
                self.cpp_integrator.setNTrial(ntrial)
        elif any([p is not None for p in [nR,depletant_type,ntrial]]):
            hoomd.context.msg.warning("Implicit depletant parameters not supported by this integrator.\n")

    def map_overlaps(self):
        R""" Build an overlap map of the system

        Returns:
            List of tuples. True/false value of the i,j entry indicates overlap/non-overlap of the ith and jth particles (by tag)

        Example:
            mc = hpmc.integrate.shape(...)
            mc.shape_param.set(...)
            overlap_map = np.asarray(mc.map_overlaps())
        """

        self.update_forces()
        N = hoomd.context.current.system_definition.getParticleData().getMaximumTag() + 1;
        overlap_map = self.cpp_integrator.mapOverlaps();
        return list(zip(*[iter(overlap_map)]*N))


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
        self.cpp_integrator.communicate(True);
        return self.cpp_integrator.countOverlaps(hoomd.context.current.system.getCurrentTimeStep(), False);

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
        counters = self.cpp_integrator.getCounters(1);
        return counters.getTranslateAcceptance();

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
        counters = self.cpp_integrator.getCounters(1);
        return counters.getRotateAcceptance();

    def get_mps(self):
        R""" Get the number of trial moves per second.

        Returns:
            The number of trial moves per second performed during the last :py:func:`hoomd.run()`.

        """
        return self.cpp_integrator.getMPS();

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
        counters = self.cpp_integrator.getCounters(1);
        return dict(translate_accept_count=counters.translate_accept_count,
                    translate_reject_count=counters.translate_reject_count,
                    rotate_accept_count=counters.rotate_accept_count,
                    rotate_reject_count=counters.rotate_reject_count,
                    overlap_checks=counters.overlap_checks,
                    translate_acceptance=counters.getTranslateAcceptance(),
                    rotate_acceptance=counters.getRotateAcceptance(),
                    move_count=counters.getNMoves());

    def get_d(self,type=None):
        R""" Get the maximum trial displacement.

        Args:
            type (str): Type name to query.

        Returns:
            The current value of the 'd' parameter of the integrator.

        """
        if type is None:
            return self.cpp_integrator.getD(0);
        else:
            return self.cpp_integrator.getD(hoomd.context.current.system_definition.getParticleData().getTypeByName(type));

    def get_a(self,type=None):
        R""" Get the maximum trial rotation.

        Args:
            type (str): Type name to query.

        Returns:
            The current value of the 'a' parameter of the integrator.

        """
        if type is None:
            return self.cpp_integrator.getA(0);
        else:
            return self.cpp_integrator.getA(hoomd.context.current.system_definition.getParticleData().getTypeByName(type));

    def get_move_ratio(self):
        R""" Get the current probability of attempting translation moves.

        Returns: The current value of the 'move_ratio' parameter of the integrator.

        """
        return self.cpp_integrator.getMoveRatio();

    def get_nselect(self):
        R""" Get nselect parameter.

        Returns:
            The current value of the 'nselect' parameter of the integrator.

        """
        return self.cpp_integrator.getNSelect();

    def get_ntrial(self):
        R""" Get ntrial parameter.

        Returns:
            The current value of the 'ntrial' parameter of the integrator.

        """
        if not self.implicit:
            hoomd.context.msg.warning("ntrial only available in simulations with non-interacting depletants. Returning 0.\n")
            return 0;

        return self.cpp_integrator.getNTrial();

    def get_configurational_bias_ratio(self):
        R""" Get the average ratio of configurational bias attempts to depletant insertion moves.

        Returns:
            The average configurational bias ratio during the last :py:func:`hoomd.run()`.

        Example::

            mc = hpmc.integrate.shape(..,implicit=True);
            mc.shape_param.set(....);
            run(100)
            cb_ratio = mc.get_configurational_bias_ratio();

        """
        if not self.implicit:
            hoomd.context.msg.warning("Quantity only available in simulations with non-interacting depletants. Returning 0.\n")
            return 0;

        counters = self.cpp_integrator.getImplicitCounters(1);
        return counters.getConfigurationalBiasRatio();

    ## Check that the required implicit depletant parameters have been supplied
    # \returns Nothing
    #
    def check_implicit_params(self):
        for p in self.implicit_required_params:
            if not p in self.implicit_params:
                raise RuntimeError("Implicit depletant integrator is missing required parameter '%s.'"%(p))

## Helper methods to set rotation and translation moves by type
def setD(cpp_integrator,d):
    if isinstance(d, dict):
        for t,t_d in d.items():
            cpp_integrator.setD(t_d,hoomd.context.current.system_definition.getParticleData().getTypeByName(t))
    else:
        for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes()):
            cpp_integrator.setD(d,i);

def setA(cpp_integrator,a):
    if isinstance(a, dict):
        for t,t_a in a.items():
            cpp_integrator.setA(t_a,hoomd.context.current.system_definition.getParticleData().getTypeByName(t))
    else:
        for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes()):
            cpp_integrator.setA(a,i);

class sphere(mode_hpmc):
    R""" HPMC integration for spheres (2D/3D).

    Args:
        seed (int): Random number seed
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.

    Hard particle Monte Carlo integration method for spheres.

    Sphere parameters:

    * *diameter* (**required**) - diameter of the sphere (distance units)
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Examples::

        mc = hpmc.integrate.sphere(seed=415236)
        mc = hpmc.integrate.sphere(seed=415236, d=0.3)
        mc.shape_param.set('A', diameter=1.0)
        mc.shape_param.set('B', diameter=2.0)
        print('diameter = ', mc.shape_param['A'].diameter)

    Depletants Example::

        mc = hpmc.integrate.sphere(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set('A', diameter=1.0)
        mc.shape_param.set('B', diameter=.1)
    """

    def __init__(self, seed, d=0.1, nselect=4, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSphere(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSphere(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSphere(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSphere(hoomd.context.current.system_definition, cl_c, seed);

        # set the default parameters
        setD(self.cpp_integrator,d);
        self.cpp_integrator.setMoveRatio(1.0)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        d = param.diameter;
        return 'sphere {0}'.format(d);

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation
        Returns:
            A list of dictionaries, one for each particle type in the system. Currently assumes that all 3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        dim = hoomd.context.current.system_definition.getNDimensions()

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.shape_param.get(typename)
            # Need to add logic to figure out whether this is 2D or not
            if dim == 3:
                result.append(dict(type='Sphere',
                                   diameter=shape.diameter))
            else:
                result.append(dict(type='Disk',
                                   diameter=shape.diameter))

        return result


class convex_polygon(mode_hpmc):
    R""" HPMC integration for convex polygons (2D).

    Args:
        seed (int): Random number seed
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.

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
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.convex_polygon(seed=415236)
        mc = hpmc.integrate.convex_polygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self, False);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoConvexPolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUConvexPolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        self.initialize_shape_params();

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        verts = param.vertices;
        shape_def = 'poly3d {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} 0 '.format(*v);

        return shape_def

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation
        Returns:
            A list of dictionaries, one for each particle type in the system. Currently assumes that all 3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.shape_param.get(typename)
            result.append(dict(type='Polygon',
                                   rounding_radius=0,
                                   vertices=shape.vertices))

        return result


class convex_spheropolygon(mode_hpmc):
    R""" HPMC integration for convex spheropolygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.

    Spheropolygon parameters:

    * *vertices* (**required**) - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units)

        * The origin **MUST** be contained within the shape.
        * The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
          don't put the origin right next to an edge).

    * *sweep_radius* (**default: 0.0**) - the radius of the sphere swept around the edges of the polygon (distance units) - **optional**
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Useful cases:

     * A 1-vertex spheropolygon is a disk.
     * A 2-vertex spheropolygon is a spherocylinder.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.convex_spheropolygon(seed=415236)
        mc = hpmc.integrate.convex_spheropolygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], sweep_radius=0.1, ignore_statistics=False);
        mc.shape_param.set('A', vertices=[(0,0)], sweep_radius=0.5, ignore_statistics=True);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,False);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoSpheropolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSpheropolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string
        verts = param.vertices;
        R = float(param.sweep_radius);

        if len(verts) == 1:
            shape_def = 'ellipsoid {0} {0} {0} '.format(R);

        else:
            shape_def = 'spoly3d {0} {1} '.format(R, len(verts));

            for v in verts:
                shape_def += '{0} {1} 0 '.format(*v);

        return shape_def

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation
        Returns:
            A list of dictionaries, one for each particle type in the system. Currently assumes that all 3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.shape_param.get(typename)
            result.append(dict(type='ConvexPolyhedron',
                                   rounding_radius=shape.sweep_radius,
                                   vertices=shape.vertices))
        return result


class simple_polygon(mode_hpmc):
    R""" HPMC integration for simple polygons (2D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.

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
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Examples::

        mc = hpmc.integrate.simple_polygon(seed=415236)
        mc = hpmc.integrate.simple_polygon(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(0, 0.5), (-0.5, -0.5), (0, 0), (0.5, -0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,False);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoSimplePolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSimplePolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        verts = param.vertices;
        shape_def = 'poly3d {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} 0 '.format(*v);

        return shape_def

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation
        Returns:
            A list of dictionaries, one for each particle type in the system. Currently assumes that all 3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.shape_param.get(typename)
            result.append(dict(type='Polygon',
                                   rounding_radius=0,
                                   vertices=shape.vertices))

        return result

class polyhedron(mode_hpmc):
    R""" HPMC integration for general polyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.

    Polyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        * The origin **MUST** strictly be contained in the generally nonconvex volume defined by the vertices and faces
        * The origin centered circle that encloses all verticies should be of minimal size for optimal performance (e.g.
          don't put the origin right next to a face).

    * *faces* (**required**) - a list of vertex indices for every face
    * *sweep_radius* (**default: 0.0**) - rounding radius applied to polyhedron
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.polyhedron(seed=415236)
        mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
            (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],\
            faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)]);
        print('vertices = ', mc.shape_param['A'].vertices)
        print('faces = ', mc.shape_param['A'].faces)

    Depletants Example::

        mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=3,depletant_type='B')
        faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)];
        mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
            (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)], faces = faces);
        mc.shape_param.set('B', vertices=[(-0.05, -0.05, -0.05), (-0.05, -0.05, 0.05), (-0.05, 0.05, -0.05), (-0.05, 0.05, 0.05), \
            (0.05, -0.05, -0.05), (0.05, -0.05, 0.05), (0.05, 0.05, -0.05), (0.05, 0.05, 0.05)], faces = faces);
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitPolyhedron(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoPolyhedron(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUPolyhedron(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUPolyhedron(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop

        verts = param.vertices;
        # represent by convex hull, pos doesn't support non-convex shapes yet
        shape_def = 'polyV {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} {2} '.format(*v);

        faces = param.faces;
        shape_def += '{0} '.format(len(faces))
        for f in faces:
            shape_def += '{0} '.format(len(f));
            for vi in f:
                shape_def += '{0} '.format(vi)

        return shape_def

class convex_polyhedron(mode_hpmc):
    R""" HPMC integration for convex polyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): (Override the automatic choice for the number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.
        max_verts (int): Set the maximum number of vertices in a polyhedron.

    Convex polyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        * The origin **MUST** be contained within the vertices.
        * The origin centered circle that encloses all verticies should be of minimal size for optimal performance (e.g.
          don't put the origin right next to a face).

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236)
        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)

    Depletants Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=3,depletant_type='B')
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        mc.shape_param.set('B', vertices=[(0.05, 0.05, 0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (-0.05, -0.05, 0.05)]);
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False, max_verts=8):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if implicit:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitGPUConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoGPUConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.max_verts = max_verts;
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

        # meta data
        self.metadata_fields = ['max_verts']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        verts = param.vertices;
        shape_def = 'poly3d {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} {2} '.format(*v);

        return shape_def

    def get_type_shapes(self):
        """ Get all the types of shapes in the current simulation
        Returns:
            A list of dictionaries, one for each particle type in the system. Currently assumes that all 3D shapes are convex.
        """
        result = []

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();

        for i in range(ntypes):
            typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            shape = self.shape_param.get(typename)
            dim = hoomd.context.current.system_definition.getNDimensions()
            # Currently can't trivially pull the radius for nonspherical shapes
            result.append(dict(type='ConvexPolyhedron',
                                   rounding_radius=0,
                                   vertices=shape.vertices))

        return result

class faceted_sphere(mode_hpmc):
    R""" HPMC integration for faceted spheres (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.

    A faceted sphere is a sphere interesected with halfspaces. The equation defining each halfspace is given by:

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
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        Planes must not be coplanar.

    Example::

        mc = hpmc.integrate.faceted_sphere(seed=415236)
        mc = hpmc.integrate.faceted_sphere(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
        print('diameter = ', mc.shape_param['A'].diameter)

    Depletants Example::

        mc = hpmc.integrate.pathcy_sphere(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=3,depletant_type='B')
        mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
        mc.shape_param.set('B', normals=[],diameter=0.1);
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitFacetedSphere(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoFacetedSphere(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUFacetedSphere(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUFacetedSphere(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        vertices = param.vertices;
        d = param.diameter;
        if vertices is not None:
            # build up shape_def string in a loop
            shape_def = 'polySphere {0} {1} '.format(d/2.0,len(vertices));

            v = []
            for v in vertices:
                shape_def += '{0} {1} {2} '.format(*v)

            return shape_def
        else:
            raise RuntimeError("No vertices supplied.")

class sphinx(mode_hpmc):
    R""" HPMC integration for sphinx particles (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.

    Sphinx particles are dimpled spheres (spheres with 'positive' and 'negative' volumes).

    Sphinx parameters:

    * *diameters* - diameters of spheres (positive OR negative real numbers)
    * *centers* - centers of spheres in local coordinate frame
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Quick Example::

        mc = hpmc.integrate.sphinx(seed=415236)
        mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
        print('diameters = ', mc.shape_param['A'].diameters)

    Depletants Example::

        mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=3,depletant_type='B')
        mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,-.25])
        mc.shape_param.set('B', centers=[(0,0,0)], diameters=[.15])
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSphinx(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSphinx(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")

            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSphinx(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSphinx(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        centers = param.centers;
        diameters = param.diameters;
        circumsphere_d = param.diameter

        colors = param.colors
        if colors is None:
            # default
            colors = ["005984ff" for c in centers]

        # build up shape_def string in a loop
        shape_def = 'sphinx 0 {0} {1} '.format(circumsphere_d, len(centers));

        # for every plane, construct four bounding vertices
        for (d,c,col) in zip(diameters, centers, colors):
            shape_def += '{0} {1} {2} {3} {4} '.format(d/2.0,c[0],c[1],c[2], col);

        return shape_def

class convex_spheropolyhedron(mode_hpmc):
    R""" HPMC integration for spheropolyhedra (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.
        max_verts (int): Set the maximum number of vertices in a polyhedron.

    A sperholpolyhedron can also represent spheres (0 or 1 vertices), and spherocylinders (2 vertices).

    Spheropolyhedron parameters:

    * *vertices* (**required**) - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units)

        - The origin **MUST** be contained within the vertices.
        - The origin centered sphere that encloses all verticies should be of minimal size for optimal performance (e.g.
          don't put the origin right next to a face).
        - A sphere can be represented by specifying zero vertices (i.e. vertices=[]) and a non-zero radius R
        - Two vertices and a non-zero radius R define a prolate spherocylinder.

    * *sweep_radius* (**default: 0.0**) - the radius of the sphere swept around the edges of the polygon (distance units) - **optional**
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example::

        mc = hpmc.integrate.convex_spheropolyhedron(seed=415236)
        mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4)
        mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape_param['A'].vertices)
        mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1, ignore_statistics=True);

    Depletants example::

        mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nR=3,depletant_type='SphericalDepletant')
        mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1);
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False, max_verts=8):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoGPUSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitGPUSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.max_verts = max_verts
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

        # meta data
        self.metadata_fields = ['max_verts']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        verts = param.vertices;
        R = float(param.sweep_radius);
        # Allow spheres to be represented for zero or one verts for maximum compatibility.
        if len(verts) <= 1:
            # draw spherocylinder to avoid having to handle orientation output differently
            d = R * 2.0;
            return 'cyl {0} 0'.format(d);
        # else draw spheropolyhedron
        # build up shape_def string in a loop
        shape_def = 'spoly3d {0} {1} '.format(R, len(verts));

        for v in verts:
            shape_def += '{0} {1} {2} '.format(*v);

        return shape_def

class ellipsoid(mode_hpmc):
    R""" HPMC integration for ellipsoids (2D/3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.

    Ellipsoid parameters:

    * *a* (**required**) - principle axis a of the ellipsoid (radius in the x direction) (distance units)
    * *b* (**required**) - principle axis b of the ellipsoid (radius in the y direction) (distance units)
    * *c* (**required**) - principle axis c of the ellipsoid (radius in the z direction) (distance units)
    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Example::

        mc = hpmc.integrate.ellipsoid(seed=415236)
        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        print('ellipsoids parameters (a,b,c) = ', mc.shape_param['A'].a, mc.shape_param['A'].b, mc.shape_param['A'].c)

    Depletants Example::

        mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=50,depletant_type='B')
        mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
        mc.shape_param.set('B', a=0.05, b=0.05, c=0.05);
    """
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitEllipsoid(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoEllipsoid(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUEllipsoid(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUEllipsoid(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)

        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        return 'ellipsoid {0} {1} {2}'.format(param.a, param.b, param.c);

class sphere_union(mode_hpmc):
    R""" HPMC integration for unions of spheres (3D).

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): The number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.
        max_members (int): Set the maximum number of members in the sphere union
            * .. versionadded:: 2.1

    Sphere union parameters:

    * *diameters* (**required**) - list of diameters of the spheres (distance units).
    * *centers* (**required**) - list of centers of constituent spheres in particle coordinates.
    * *overlap* (**default: 1 for all spheres**) - only check overlap between constituent particles for which *overlap [i] & overlap[j]* is !=0, where '&' is the bitwise AND operator.

        * .. versionadded:: 2.1

    * *ignore_statistics* (**default: False**) - set to True to disable ignore for statistics tracking.
    * *ignore_overlaps* (**default: False**) - set to True to disable overlap checks between this and other types with *ignore_overlaps=True*

        * .. deprecated:: 2.1
             Replaced by :py:class:`interaction_matrix`.

    Example::

        mc = hpmc.integrate.sphere_union(seed=415236)
        mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4)
        mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
        print('diameter of the first sphere = ', mc.shape_param['A'].members[0].diameter)
        print('center of the first sphere = ', mc.shape_param['A'].centers[0])

    Depletants Example::

        mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4, implicit=True)
        mc.set_param(nselect=1,nR=50,depletant_type='B')
        mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
        mc.shape_param.set('B', diameters=[0.05], centers=[(0.0, 0.0, 0.0)]);
    """

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=4, implicit=False, max_members=8):
        hoomd.util.print_status_line();

        # initialize base class
        mode_hpmc.__init__(self,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitSphereUnion', max_members)(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoSphereUnion', max_members)(hoomd.context.current.system_definition, seed)
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoGPUSphereUnion', max_members)(hoomd.context.current.system_definition, cl_c, seed)
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitGPUSphereUnion', max_members)(hoomd.context.current.system_definition, cl_c, seed)

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.max_members = max_members;
        self.initialize_shape_params();

        # meta data
        self.metadata_fields = ['max_members']

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        diameters = [m.diameter for m in param.members]
        centers = param.centers
        colors = param.colors
        N = len(diameters);
        shape_def = 'sphere_union {0} '.format(N);
        if param.colors is None:
            # default
            colors = ["ff5984ff" for c in centers]


        for d,p,c in zip(diameters, centers, colors):
            shape_def += '{0} '.format(d);
            shape_def += '{0} {1} {2} '.format(*p);
            shape_def += '{0} '.format(c);
            # No need to use stored value for member sphere orientations

        return shape_def
