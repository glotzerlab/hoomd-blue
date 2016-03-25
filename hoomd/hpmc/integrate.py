## \package hpmc.integrate
# \brief HPMC integration modes

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

    # Hack - predefine the possible sizes. This could be possibly better by determining the sizes by inspecting
    # _hpmc.__dict__. But this works for now, it just requires updating if the C++ module is changed.
    sizes=[8,16,32,64,128]

    if max_n > sizes[-1]:
        raise ValueError("Maximum value must be less than or equal to {0}".format(sizes[-1]));

    # Find the smallest size that fits size
    for size in sizes:
        if max_n <= size:
            selected_size = size;
            break;

    return _hpmc.__dict__["{0}{1}".format(base, size)];

## Base class hpmc integrator
#
# _mode_hpmc is the base class for all HPMC integrators. It provides common interface elements. Users should not
# instantiate this class directly. Methods documented here are available to all hpmc integrators.
class _mode_hpmc(_integrator):
    ## \internal
    # \brief Initialize an empty integrator
    #
    # \post the member shape_param is created
    def __init__(self,fl_flag,implicit):
        _integrator.__init__(self);
        self.fl_flag=fl_flag
        self.implicit=implicit

        # setup the shape parameters
        self.shape_param = data.param_dict(self); # must call initialize_shape_params() after the cpp_integrator is created.

        #initialize list to check fl params
        if self.fl_flag:
            self.fl_params=list()

        #initialize list to check fl params
        if self.implicit:
            self.implicit_params=list()


    ## Set the external field
    def set_external(self, ext):
        self.cpp_integrator.setExternalField(ext.cpp_compute);

    def get_metadata(self):
        data = super(_mode_hpmc, self).get_metadata()
        data['d'] = self.get_d()
        data['a'] = self.get_a()
        data['move_ratio'] = self.get_move_ratio()
        data['nselect'] = self.get_nselect()
        shape_dict = {};
        for key in self.shape_param.keys():
            shape_dict[key] = self.shape_param[key].get_metadata();
        data['shape_param'] = shape_dict;
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

        # check that particle orientations are normalized
        if not self.cpp_integrator.checkParticleOrientations():
           hoomd.context.msg.error("Particle orientations are not normalized\n");
           raise RuntimeError("Error running integrator");

        #make sure all the required FL parameters have been supplied
        if self.fl_flag:
            self.check_fl_params()

        #make sure all the required parameters for implicit depletant simulations have been supplied
        if self.implicit:
            self.check_implicit_params()

    ## Set pos_writer definitions for specified shape parameters
    #
    # \param pos pos writer to setup
    # \param colors dictionary of type name to color mappings
    # \returns nothing
    #
    # setup_pos_writer uses the shape_param settings to specify the shape definitions (via set_def) to the provided
    # pos file. This overrides any previous values specified to set_def.
    #
    # \a colors allows you to set per-type colors for particles. Specify colors as strings in the injavis format. When
    # colors is not specified for a type, all colors default to 005984FF.
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(...);
    # mc.shape_param.set(....);
    # pos = pos_writer.dumpy.pos("dump.pos", period=100);
    # mc.setup_pos_writer(pos, colors=dict(A='005984FF'));
    # ~~~~~~~~~~~~
    #
    def setup_pos_writer(self, pos, colors={}):
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

    def initialize_shape_params(self):
        shape_param_type = None;
        # have to have a few extra checks becuase the sized class don't actually exist yet.
        if isinstance(self, convex_polyhedron):
            shape_param_type = data.convex_polyhedron_params.get_sized_class(self.max_verts);
        elif isinstance(self, convex_spheropolyhedron):
            shape_param_type = data.convex_spheropolyhedron_params.get_sized_class(self.max_verts);
        else:
            shape_param_type = data.__dict__[self.__class__.__name__ + "_params"]; # using the naming convention for convenience.
        # setup the coefficient options
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        for i in range(0,ntypes):
            type_name = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
            if not type_name in self.shape_param.keys(): # only add new keys
                self.shape_param.update({ type_name: shape_param_type(self, i) });


    ## Changes parameters of an existing integration mode
    # \param d (if set) Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a (if set) Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio (if set) New value for the move ratio
    # \param nselect (if set) New value for the number of particles to select for trial moves in one cell
    # \param ln_gamma (if set) **Frenkel-Ladd only**: new value for ln_gamma
    # \param q_factor (if set) **Frenkel-Ladd only**: new value for q_factor
    # \param r0 (if set) **Frenkel-Ladd only**: new value for r0
    # \param q0 (if set) **Frenkel-Ladd only**: new value for q0
    # \param drift_period (if set) **Frenkel-Ladd only**: new value for drift_period
    # \param nR (if set) **Implicit depletants only**: Number density of implicit depletants in free volume
    # \param depletant_type (if set) **Implicit depletants only**: Particle type to use as implicit depletant
    # \param ntrial (if set) **Implicit depletants only**: Number of re-insertion attempts per overlapping depletant
    # \param use_bvh (if set) No longer used.
    #
    # \returns nothing
    #
    def set_params(self,
                   d=None,
                   a=None,
                   move_ratio=None,
                   nselect=None,
                   ln_gamma=None,
                   q_factor=None,
                   r0=None,
                   q0=None,
                   drift_period=None,
                   use_bvh=None,
                   nR=None,
                   depletant_type=None,
                   ntrial=None):
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

        if self.fl_flag:
            if ln_gamma is not None:
                  self.fl_params.append('ln_gamma')
                  self.cpp_integrator.setLnGamma(ln_gamma)
            if q_factor is not None:
                  self.fl_params.append('q_factor')
                  self.cpp_integrator.setQFactor(q_factor)
            if r0 is not None:
                  self.fl_params.append('r0')
                  self.cpp_integrator.setR0([tuple(r) for r in r0])
            if q0 is not None:
                  self.fl_params.append('q0')
                  self.cpp_integrator.setQ0([tuple(q) for q in q0])
            if drift_period is not None:
                  self.fl_params.append('drift_period')
                  self.cpp_integrator.setDriftPeriod(drift_period)
        elif any([p is not None for p in [ln_gamma,q_factor,r0,q0,drift_period]]):
            raise RuntimeError("FL integration parameters specified for non-FL integrator")

        if use_bvh is not None:
            hoomd.context.msg.warning("use_bvh is no longer needed, HPMC always run in BVH mode.")
            hoomd.context.msg.warning("use_bvh may be removed in a future version.")
            # TODO: remove use_bvh some day - this message was added 8/4/2014

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

    ## Count the number of overlaps
    #
    # \returns The number of overlaps in the current system configuration
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param.set(....);
    # run(100)
    # num_overlaps = mc.count_overlaps();
    # ~~~~~~~~~~~~
    #
    def count_overlaps(self):
        self.update_forces()
        self.cpp_integrator.communicate(True);
        return self.cpp_integrator.countOverlaps(hoomd.context.current.system.getCurrentTimeStep(), False);

    ## Get the average acceptance ratio for translate moves
    #
    # \returns The average translate accept ratio during the last run()
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param.set(....);
    # run(100)
    # t_accept = mc.get_translate_acceptance();
    # ~~~~~~~~~~~~
    #
    def get_translate_acceptance(self):
        counters = self.cpp_integrator.getCounters(1);
        return counters.getTranslateAcceptance();

    ## Get the average acceptance ratio for rotate moves
    #
    # \returns The average rotate accept ratio during the last run()
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param.set(....);
    # run(100)
    # t_accept = mc.get_rotate_acceptance();
    # ~~~~~~~~~~~~
    #
    def get_rotate_acceptance(self):
        counters = self.cpp_integrator.getCounters(1);
        return counters.getRotateAcceptance();

    ## Get the number of trial moves per second
    #
    # \returns The number of trial moves per second performed during the last run() command
    #
    def get_mps(self):
        return self.cpp_integrator.getMPS();

    ## Get all trial move counters
    #
    # \returns A dictionary containing all trial moves counted during the last run
    #
    # The dictionary contains the entries
    # * *translate_accept_count* - count of the number of accepted translate moves
    # * *translate_reject_count* - count of the number of rejected translate moves
    # * *rotate_accept_count* - count of the number of accepted rotate moves
    # * *rotate_reject_count* - count of the number of rejected rotate moves
    # * *overlap_checks* - estimate of the number of overlap checks performed
    # * *translate_acceptance* - Average translate acceptance ratio over the run
    # * *rotate_acceptance* - Average rotate acceptance ratio over the run
    # * *move_count* - Count of the number of trial moves during the run
    #
    def get_counters(self):
        counters = self.cpp_integrator.getCounters(1);
        return dict(translate_accept_count=counters.translate_accept_count,
                    translate_reject_count=counters.translate_reject_count,
                    rotate_accept_count=counters.rotate_accept_count,
                    rotate_reject_count=counters.rotate_reject_count,
                    overlap_checks=counters.overlap_checks,
                    translate_acceptance=counters.getTranslateAcceptance(),
                    rotate_acceptance=counters.getRotateAcceptance(),
                    move_count=counters.getNMoves());

    ## Get the maximum trial displacement
    # \param type Type name to query
    #
    # \returns The current value of the 'd' parameter of the integrator
    #
    def get_d(self,type=None):
        if type is None:
            return self.cpp_integrator.getD(0);
        else:
            return self.cpp_integrator.getD(hoomd.context.current.system_definition.getParticleData().getTypeByName(type));

    ## Get the maximum trial rotation
    # \param type Type name to query
    #
    # \returns The current value of the 'a' parameter of the integrator
    #
    def get_a(self,type=None):
        if type is None:
            return self.cpp_integrator.getA(0);
        else:
            return self.cpp_integrator.getA(hoomd.context.current.system_definition.getParticleData().getTypeByName(type));

    ## Get the current probability of attempting translation moves (versus rotation moves)
    #
    # \returns The current value of the 'move_ratio' parameter of the integrator
    #
    def get_move_ratio(self):
        return self.cpp_integrator.getMoveRatio();

    ## Get nselect parameter
    #
    # \returns The current value of the 'nselect' parameter of the integrator
    #
    def get_nselect(self):
        return self.cpp_integrator.getNSelect();

    ## Get ntrial parameter
    #
    # \returns The current value of the 'ntrial' parameter of the integrator
    #
    def get_ntrial(self):
        if not self.implicit:
            hoomd.context.msg.warning("ntrial only available in simulations with non-interacting depletants. Returning 0.\n")
            return 0;

        return self.cpp_integrator.getNTrial();

    ## Get the average ratio of configurational bias attempts to depletant insertion moves
    #
    # \note Only applies to simulations with non-interacting depletants
    #
    # \returns The average configurational bias ratio during the last run()
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..,implicit=True);
    # mc.shape_param.set(....);
    # run(100)
    # cb_ratio = mc.get_configurational_bias_ratio();
    # ~~~~~~~~~~~~
    #
    def get_configurational_bias_ratio(self):
        if not self.implicit:
            hoomd.context.msg.warning("Quantity only available in simulations with non-interacting depletants. Returning 0.\n")
            return 0;

        counters = self.cpp_integrator.getImplicitCounters(1);
        return counters.getConfigurationalBiasRatio();

    ## Reset integrator state
    #
    # If the integrator maintains any internal state, this method resets it.
    # This is useful for integrators which maintain running averages (e.g. Frenkel Ladd)
    # \returns Nothing
    #
    def reset_state(self):
        if self.fl_flag:
            self.cpp_integrator.resetState(hoomd.context.current.system.getCurrentTimeStep())
        else:
            hoomd.context.msg.warning("This integrator does not maintain state. Are you using the integrator you think you are?\n");

    ## Check the the required FL parameters have been supplied
    # \returns Nothing
    #
    def check_fl_params(self):
        for p in self.fl_required_params:
            if not p in self.fl_params:
                raise RuntimeError("FL Integrator is missing required parameter '%s.'"%(p))

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

## HPMC integration for spheres (2D/3D)
#
# Hard particle Monte Carlo integration method for spheres.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Sphere parameters:**
# - *diameter* - diameter of the sphere (distance units) - **required**
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \note The largest diameter in a simulation should be 1 for optimal performance.
#
class sphere(_mode_hpmc):
    ## Specifies the hpmc integration mode for spheres
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar for all types or dict by type
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236)
    # mc = hpmc.integrate.sphere(seed=415236, d=0.3)
    # mc.shape_param.set('A', diameter=1.0)
    # mc.shape_param.set('B', diameter=2.0)
    # print('diameter = ', mc.shape_param['A'].diameter)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # mc = hpmc.integrate.sphere(seed=415236, d=0.3, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,r0=rs,drift_period)
    # mc.shape_param.set('B', diameter=1.0)
    #
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=8,nR=3,depletant_type='B')
    # mc.shape_param.set('A', diameter=1.0)
    # mc.shape_param.set('B', diameter=.1)
    def __init__(self, seed, d=0.1, nselect=None, fl_flag=False, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if (fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSphere(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSphere(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSphere(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSphere(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSphere(hoomd.context.current.system_definition, cl_c, seed);

        # set the default parameters
        setD(self.cpp_integrator,d);
        self.cpp_integrator.setMoveRatio(1.0)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        self.initialize_shape_params();

        if fl_flag:
          self.fl_required_params=['r0','ln_gamma']
          self.cpp_integrator.setQFactor(0)
          self.cpp_integrator.setQ0(hoomd.context.current.system_definition.getParticleData().getN()*[(1.0,0.0,0.0,0.0)])
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        d = param.diameter;
        return 'sphere {0}'.format(d);

## \internal
# \brief HPMC integration for spherocylinders
#
# \warning The spherocylinder is not implemented. Do not use this method.
# \param fl_flag Flag for enabling Frenkel-Ladd Integration
#
class spherocylinder(_mode_hpmc):
    ## Specifies the hpmc integration mode
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5,fl_flag=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag);

        # initialize the reflected c++ class
        #if not hoomd.context.exec_conf.isCUDAEnabled():
        #    self.cpp_integrator = _hpmc.IntegratorHPMCSphere(hoomd.context.current.system_definition, neighbor_list.cpp_nlist, seed);
        #else:
        #    self.cpp_integrator = _hpmc.IntegratorHPMCGPUSphere(hoomd.context.current.system_definition, neighbor_list.cpp_nlist, seed);

        #No GPU implementation yet...
        if (fl_flag):
          self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSpherocylinder(hoomd.context.current.system_definition, neighbor_list.cpp_nlist, seed);
        else:
          self.cpp_integrator = _hpmc.IntegratorHPMCMonoSpherocylinder(hoomd.context.current.system_definition, neighbor_list.cpp_nlist, seed);

        # Set the appropriate neighbor list range
        neighbor_list.subscribe(lambda: self.cpp_integrator.getMaxDiameter()+2*self.d);

        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

## HPMC integration for convex polygons (2D)
#
# Hard particle Monte Carlo integration method for convex polygons.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Convex polygon parameters:**
# - *vertices* - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units) - **required**
#     - Vertices **MUST** be specified in a *counter-clockwise* order.
#     - The origin **MUST** be contained within the vertices.
#     - Points inside the polygon **MUST NOT** be included.
#     - The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
#       don't put the origin right next to an edge).
#     - The enclosing circle of the largest polygon should be diameter ~1 for optimal performance.
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class convex_polygon(_mode_hpmc):
    ## Specifies the hpmc integration mode for convex polygons
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_polygon(seed=415236)
    # mc = hpmc.integrate.convex_polygon(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.convex_polygon(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_polygon(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]);
    # mc.shape_param.set('B', vertices=[(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)]);
    #
    # \note Implicit depletants are not yet supported for this 2D shape
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag, implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLConvexPolygon(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitConvexPolygon(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoConvexPolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUConvexPolygon(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUConvexPolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        self.initialize_shape_params();

        # setup the coefficient options
        if fl_flag:
            self.required_params+=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        verts = param.vertices;
        shape_def = 'poly3d {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} 0 '.format(*v);

        return shape_def

## HPMC integration for convex spheropolygons
#
# Hard particle Monte Carlo integration method for convex spheropolygons.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# Useful cases:
#  * A 1-vertex spheropolygon is a disk.
#  * A 2-vertex spheropolygon is a spherocylinder.
#
# **Spheropolygon parameters:**
# - *vertices* - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units) - **required**
#     - The origin **MUST** be contained within the vertices.
#     - The origin centered circle that encloses all vertices should be of minimal size for optimal performance (e.g.
#       don't put the origin right next to an edge).
#     - The enclosing circle of the largest spheropolygon should be diameter ~1 for optimal performance.
# - *sweep_radius* - the radius of the sphere swept around the edges of the polygon (distance units) - **optional** (default: 0.0)
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class convex_spheropolygon(_mode_hpmc):
    ## Specifies the hpmc integration mode for convex spheropolygons
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_spheropolygon(seed=415236)
    # mc = hpmc.integrate.convex_spheropolygon(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], sweep_radius=0.1, ignore_statistics=False);
    # mc.shape_param.set('A', vertices=[(0,0)], sweep_radius=0.5, ignore_statistics=True);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.convex_spheropolygon(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], sweep_radius=0.1, ignore_statistics=False);
    # mc.shape_param.set('A', vertices=[(0,0)], sweep_radius=0.5, ignore_statistics=True);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_spheropolygon(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], sweep_radius=0.1, ignore_statistics=False);
    # mc.shape_param.set('B', vertices=[(0,0)], sweep_radius=0.1);
    #
    # \note Implicit depletants are not yet supported for this 2D shape
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSpheropolygon(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSpheropolygon(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSpheropolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSpheropolygon(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSpheropolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

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

## HPMC integration for simple polygons (2D)
#
# Hard particle Monte Carlo integration method for simple polygons.
#
# For simple polygons that are not concave, use integrate.convex_polygon, it will execute much faster than
# integrate.simple_polygon.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Simple polygon parameters:**
# - *vertices* - vertices of the polygon as is a list of (x,y) tuples of numbers (distance units) - **required**
#     - Vertices **MUST** be specified in a *counter-clockwise* order.
#     - The polygon may be concave, but edges must not cross.
#     - The origin doesn't necessarily need to be inside the shape.
#     - The origin centered circle that encloses all vertices should be of minimal size for optimal performance.
#     - The enclosing circle of the largest polygon should be diameter ~1 for optimal performance.
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class simple_polygon(_mode_hpmc):
    ## Specifies the hpmc integration mode for simple polygons
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.simple_polygon(seed=415236)
    # mc = hpmc.integrate.simple_polygon(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', vertices=[(0, 0.5), (-0.5, -0.5), (0, 0), (0.5, -0.5)]);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.simple_polygon(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', vertices=[(0, 0.5), (-0.5, -0.5), (0, 0), (0.5, -0.5)]);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.simple_polygon(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    # mc.shape_param.set('B', vertices=[(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)])
    #
    # \note Implicit depletants are not yet supported for this 2D shape
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSimplePolygon(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSimplyPolygon(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSimplePolygon(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSimplePolygon(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSimplePolygon(hoomd.context.current.system_definition, cl_c, seed);

        # set parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        verts = param.vertices;
        shape_def = 'poly3d {0} '.format(len(verts));

        for v in verts:
            shape_def += '{0} {1} 0 '.format(*v);

        return shape_def

## HPMC integration for general polyhedra
#
# Hard particle Monte Carlo integration method for general polyhedra.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Polyhedron parameters:**
# - *vertices* - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units) - **required**
#     - The origin **MUST** strictly be contained in the generally nonconvex volume defined by the vertices and faces
#     - The origin centered circle that encloses all verticies should be of minimal size for optimal performance (e.g.
#       don't put the origin right next to an edge).
#     - The enclosing circle of the largest polyhedron should be diameter 1 for optimal performance.
# - *faces* - a list of vertex indices for every face
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# violated.
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class polyhedron(_mode_hpmc):
    ## Specifies the hpmc integration mode for convex polyhedra
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.polyhedron(seed=415236)
    # mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
    #     (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],\
    #     faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)]);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # print('faces = ', mc.shape_param['A'].faces)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
    #     (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)],\
    #     faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)]);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.polyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # faces = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)];
    # mc.shape_param.set('A', vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), \
    #     (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)], faces = faces);
    # mc.shape_param.set('B', vertices=[(-0.05, -0.05, -0.05), (-0.05, -0.05, 0.05), (-0.05, 0.05, -0.05), (-0.05, 0.05, 0.05), \
    #     (0.05, -0.05, -0.05), (0.05, -0.05, 0.05), (0.05, 0.05, -0.05), (0.05, 0.05, 0.05)], faces = faces);
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False, implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLPolyhedron(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitPolyhedron(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoPolyhedron(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUPolyhedron(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUPolyhedron(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
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

## HPMC integration for convex polyhedra
#
# Hard particle Monte Carlo integration method for convex polyhedra.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Convex polyhedron parameters:**
# - *vertices* - vertices of the polyhedron as is a list of (x,y,z) tuples of numbers (distance units) - **required**
#     - The origin **MUST** be contained within the vertices.
#     - The origin centered circle that encloses all verticies should be of minimal size for optimal performance (e.g.
#       don't put the origin right next to an edge).
#     - The enclosing circle of the largest polyhedron should be diameter 1 for optimal performance.
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class convex_polyhedron(_mode_hpmc):
    ## Specifies the hpmc integration mode for convex polyhedra
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    # \param max_verts Set the maximum number of vertices in a polyhedron.
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_polyhedron(seed=415236)
    # mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # mc.shape_param.set('B', vertices=[(0.05, 0.05, 0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (-0.05, -0.05, 0.05)]);
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False, max_verts=64):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMono_FLConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoConvexPolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU.");
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

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
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

## HPMC integration for faceted spheres
#
# Hard particle Monte Carlo integration method for spheres intersected by halfspaces
#
# The equation defining each halfspace is given by
# \f$ n_i\cdot r + b_i \le 0 \f$,
# where \f$ n_i \f$ is the face normal, and \f$ b_i \f$ the offset.
#
# The origin must be chosen so as to lie **inside the shape**, or the overlap check will not work.
# This condition is not checked.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Faceted sphere parameters:**
# - *normals* - list of (x,y,z) tuples defining the facet normals (distance units) - **required**
# - *offsets* - list of offsets (distance unit^2) -- **required**
# - *diameter* - diameter of sphere
# - *vertices* - list of vertices for intersection polyhedron - **required**
# - *origin* - origin vector -- **required**
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \warn It is assumed that none of the input planes are coplanar
#
class faceted_sphere(_mode_hpmc):
    ## Specifies the hpmc integration mode for faceted spheres
    # \param seed Random number seed
    # \param d Maximum move displacement
    # \param a Maximum rotation move
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.faceted_sphere(seed=415236)
    # mc = hpmc.integrate.faceted_sphere(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
    # print('diameter = ', mc.shape_param['A'].diameter)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.faceted_sphere(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.pathcy_sphere(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', normals=[(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)],diameter=1.0);
    # mc.shape_param.set('B', normals=[],diameter=0.1);

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLFacetedSphere(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitFacetedSphere(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoFacetedSphere(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUFacetedSphere(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUFacetedSphere(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
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

## HPMC integration for sphinx particles
#
# Hard particle Monte Carlo integration method for dimpled spheres (spheres with 'positive' and 'negative' volumes)
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Sphinx parameters:**
# - *diameters* - diameters of spheres (positive OR negative real numbers)
# - *centers* - centers of spheres in local coordinate frame
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
class sphinx(_mode_hpmc):
    ## Specifies the hpmc integration mode for faceted spheres
    # \param seed Random number seed
    # \param d Maximum move displacement
    # \param a Maximum rotation move
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphinx(seed=415236)
    # mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
    # print('diameters = ', mc.shape_param['A'].diameters)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphinx(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=3,depletant_type='B')
    # mc.shape_param.set('A', centers=[(0,0,0),(1,0,0)], diameters=[1,-.25])
    # mc.shape_param.set('B', centers=[(0,0,0)], diameters=[.15])
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSphinx(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSphinx(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSphinx(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
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

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        centers = param.centers;
        diameters = param.diameters;
        circumsphere_d = param.diameter

        if param.colors is None:
            # default
            colors = ["005984ff" for c in centers]

        # build up shape_def string in a loop
        shape_def = 'sphinx 0 {0} {1} '.format(circumsphere_d, len(centers));

        # for every plane, construct four bounding vertices
        for (d,c,col) in zip(diameters, centers, colors):
            shape_def += '{0} {1} {2} {3} {4} '.format(d/2.0,c[0],c[1],c[2], col);

        return shape_def

#
## HPMC integration for spheropolyhedra
#
# Hard particle Monte Carlo integration method for spheres, spherocylinders, and spheropolyhedra.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Spheropolyhedron parameters:**
# - *vertices* - vertices of the polyhedron as a list of (x,y,z) tuples of numbers (distance units) - **required**
#     - The origin **MUST** be contained within the vertices.
#     - The origin centered sphere that encloses all verticies should be of minimal size for optimal performance (e.g.
#       don't put the origin right next to an edge).
#     - The enclosing sphere of the largest polyhedron should be diameter ~1 for optimal performance.
#     - A sphere can be represented by specifying zero vertices (i.e. vertices=[]) and a non-zero radius R
#     - Two vertices and a non-zero radius R define a prolate spherocylinder, but pos file output is not supported.
# - *sweep_radius* - the radius of the sphere swept around the surface of the polyhedron - **optional** (default: 0.0)
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#              (default: False)
#
# \warning Currently, hpmc does not check that all requirements are met. Undefined behavior will result if they are
# violated.
#
class convex_spheropolyhedron(_mode_hpmc):
    ## Specifies the hpmc integration mode for convex spheropolyhedra
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    # \param max_verts Set the maximum number of vertices in a spheropolyhedron.
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_spheropolyhedron(seed=415236)
    # mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4)
    # mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # print('vertices = ', mc.shape_param['A'].vertices)
    # mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1, ignore_statistics=True);
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1, ignore_statistics=True);
    # ~~~~~~~~~~~~~
    # \par Depletants example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.convex_spheropolyhedron(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nR=3,depletant_type='SphericalDepletant')
    # mc.shape_param['tetrahedron'].set(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
    # mc.shape_param['SphericalDepletant'].set(vertices=[], sweep_radius=0.1);
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False, max_verts=64):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMono_FLSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoImplicitSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _get_sized_entry('IntegratorHPMCMonoSpheropolyhedron', max_verts)(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
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

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
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

## HPMC integration for ellipsoids (2D/3D)
#
# Hard particle Monte Carlo integration method for ellipsoids.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **Ellipsoid parameters:**
# - *a* - principle axis a of the ellipsoid (radius in the x direction) (distance units) - **required**
# - *b* - principle axis b of the ellipsoid (radius in the b direction) (distance units) - **required**
# - *c* - principle axis c of the ellipsoid (radius in the c direction) (distance units) - **required**
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# \note The largest radius should be 0.5 for optimal performance
#
class ellipsoid(_mode_hpmc):
    ## Specifies the hpmc integration mode for ellipsoid shapes
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.ellipsoid(seed=415236)
    # mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
    # print('ellipsoids parameters (a,b,c) = ', mc.shape_param['A'].a, mc.shape_param['A'].b, mc.shape_param['A'].c)
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
    # mc.shape_param.set('A', a=0.5, b=0.25, c=0.125, ignore_statistics=True);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.ellipsoid(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=50,depletant_type='B')
    # mc.shape_param.set('A', a=0.5, b=0.25, c=0.125);
    # mc.shape_param.set('B', a=0.5, b=0.25, c=0.125);
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLEllipsoid(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitEllipsoid(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoEllipsoid(hoomd.context.current.system_definition, seed);
        else:
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUEllipsoid(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUEllipsoid(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)

        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        return 'ellipsoid {0} {1} {2}'.format(param['a'], param['b'], param['c']);

## HPMC integration for sphere_union
#
# Hard particle Monte Carlo integration method for particles composed of union of multiple spheres.
#
# Required parameters must be set for each particle type before the simulation is started with run(). To set shape
# parameters, save the integrator in a variable (e.g. mc) then use
# \link param::set mc.shape_param.set() \endlink to set parameters.
# ~~~~~~~~~~~~
# mc = hpmc.integrate.shape(...)
# mc.shape_param.set('A', param=1.0)
# ~~~~~~~~~~~~
#
# **SphereUnion parameters:**
# - *diameters* - list of diameters of the spheres (distance units) - **required**
# - *centers* - list of centers of constituent spheres in particle coordinates - **required**
# - *ignore_overlaps* - set to True to disable overlap checks between this and other types with *ignore_overlaps*=True - **optional** (default: False)
# - *ignore_statistics* - set to True to disable ignore for statistics tracking **optional** (default: False)
#
# Multishape is the name of the internal C++ naming scheme. The user-facing interface is called sphere_union
class sphere_union(_mode_hpmc):
    ## Specifies the hpmc integration mode for a union of spheres
    # \param seed Random number seed
    # \param d Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param a Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type
    # \param move_ratio ratio of translation moves to rotation moves
    # \param nselect (if set) Override the automatic choice for the number of trial moves to perform in each cell
    # \param fl_flag Flag for enabling Frenkel-Ladd Integration
    # \param implicit Flag to enable implicit depletants
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere_union(seed=415236)
    # mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4)
    # mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
    # print('diameter of the first sphere = ', mc.shape_param['A'].members[0].diameter)
    # print('center of the first sphere = ', mc.shape_param['A'].centers[0])
    # ~~~~~~~~~~~~~
    # \par FL Example:
    # ~~~~~~~~~~~~~
    # rs = [...] # Einstein crystal coordinate position
    # qs = [...] # Einstein crystal coordinate orientation
    # mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4, fl_flag=True)
    # mc.set_param(ln_gamma=2.0,q_factor=10.0,r0=rs,q0=qs,drift_period)
    # mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
    # ~~~~~~~~~~~~~
    # \par Depletants Example:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere_union(seed=415236, d=0.3, a=0.4, implicit=True)
    # mc.set_param(nselect=1,nR=50,depletant_type='B')
    # mc.shape_param.set('A', diameters=[1.0, 1.0], centers=[(-0.25, 0.0, 0.0), (0.25, 0.0, 0.0)]);
    # mc.shape_param.set('B', diameters=[1.0], centers=[(0.0, 0.0, 0.0)]);

    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.5, nselect=None,fl_flag=False,implicit=False):
        hoomd.util.print_status_line();

        # initialize base class
        _mode_hpmc.__init__(self,fl_flag,implicit);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(fl_flag):
                self.cpp_integrator = _hpmc.IntegratorHPMCMono_FLSphereUnion(hoomd.context.current.system_definition, seed);
            elif(implicit):
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitSphereUnion(hoomd.context.current.system_definition, seed)
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoSphereUnion(hoomd.context.current.system_definition, seed);
        else:
            if (fl_flag):
                raise RuntimeError("Frenkel Ladd calculations are not implemented for the GPU at this time")
            cl_c = _hoomd.CellListGPU(hoomd.context.current.system_definition);
            hoomd.context.current.system.addCompute(cl_c, "auto_cl2")
            if not implicit:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoGPUSphereUnion(hoomd.context.current.system_definition, cl_c, seed);
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoImplicitGPUSphereUnion(hoomd.context.current.system_definition, cl_c, seed);

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if fl_flag:
            self.fl_required_params=['r0','q0','ln_gamma','q_factor']
        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

    # \internal
    # \brief Format shape parameters for pos file output
    def format_param_pos(self, param):
        # build up shape_def string in a loop
        diameters = [member.radius*2.0 for member in param.members];
        centers = param.centers;
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
