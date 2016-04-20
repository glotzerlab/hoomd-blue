## \package hpmc.update
# \brief NPT box Updater for HPMC

from . import _hpmc
from . import integrate
from . import compute
from hoomd import _hoomd

import math

from hoomd.update import _updater
import hoomd

## Apply box updates to sample the NPT ensemble
#
# Every \a period steps, a lattice vector is rescaled or sheared with Metropolis acceptance criteria.
# Most trial box changes require updating every particle position at least once and checking the whole system
# for overlaps. This will slow down simulations a lot if run frequently, but box angles are slow to equilibrate.
#
# Pressure inputs to update.npt are defined as \f$ \beta P \f$. Conversions from a specific definition of reduced
# pressure \f$ P^* \f$ are left for the user to perform.
#
class npt(_updater):
    ## Initialize the box updater.
    #
    # \param mc HPMC integrator object for system on which to apply box updates
    # \param P \f$ \beta P \f$. Apply your chosen reduced pressure convention externally.
    # \param dLx maximum change of the box length in the first lattice vector direction (in distance units)
    # \param dLy maximum change of the box length in the second lattice vector direction (in distance units)
    # \param dLz maximum change of the box length in the third lattice vector direction (in distance units)
    # \param dxy maximum change of the X-Y tilt factor (dimensionless)
    # \param dxz maximum change of the X-Z tilt factor (dimensionless)
    # \param dyz maximum change of the Y-Z tilt factor as a function of time (dimensionless)
    # \param move_ratio ratio of trial volume change attempts to trial box shearing attempts
    # \param reduce Maximum number of lattice vectors of shear to allow before applying lattice reduction.
    #            Shear of +/- 0.5 cannot be lattice reduced, so set to a value <= 0.5 to disable (default 0).
    #            Note that due to precision errors, lattice reduction may introduce small overlaps which can be resolved,
    #            but which temporarily break detailed balance. Automatic lattice reduction is not supported with MPI.
    # \param isotropic Set to true to link dLx, dLy, and dLz. The dLy and dLz parameters are then ignored. To change the box
    #            aspect ratio, either disable the updater or set the isotropic parameter to False during the call to
    #            the hoomd update.box_resize()
    # \param seed random number seed for MC box changes
    # \param period The box size will be updated every \a period steps
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236, d=0.3)
    # npt = hpmc.update.npt(mc, P=1.0, dLx=0.1, dLy=0.1, dxy=0.1)
    # ~~~~~~~~~~~~~
    def __init__(self, mc, P, seed, dLx=0.0, dLy=0.0, dLz=0.0, dxy=0.0, dxz=0.0, dyz=0.0, move_ratio=0.5, reduce=0.0, isotropic=False, period=1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        if not isinstance(mc, integrate.mode_hpmc):
            hoomd.context.msg.warning("update.npt: Must have a handle to an HPMC integrator.\n");
            return;
        if dLx == 0.0 and dLy == 0.0 and dLz == 0.0 and dxy == 0.0 and dxz == 0.0 and dyz == 0.0:
            hoomd.context.msg.warning("update.npt: All move size parameters are 0\n");

        self.P = hoomd.variant._setup_variant_input(P);
        self.dLx = float(dLx);
        self.dLy = float(dLy);
        self.dLz = float(dLz);
        self.dxy = float(dxy);
        self.dxz = float(dxz);
        self.dyz = float(dyz);
        self.reduce = float(reduce);
        self.isotropic = bool(isotropic);
        self.move_ratio = float(move_ratio);

        # create the c++ mirror class
        self.cpp_updater = _hpmc.UpdaterBoxNPT(hoomd.context.current.system_definition,
                                               mc.cpp_integrator,
                                               self.P.cpp_variant,
                                               self.dLx,
                                               self.dLy,
                                               self.dLz,
                                               self.dxy,
                                               self.dxz,
                                               self.dyz,
                                               self.move_ratio,
                                               self.reduce,
                                               self.isotropic,
                                               int(seed),
                                               );

        if period is None:
            self.cpp_updater.update(hoomd.context.current.system.getCurrentTimeStep());
        else:
            self.setupUpdater(period);

    ## Change npt parameters
    #
    # \param P \f$ \beta P \f$. Apply your chosen reduced pressure convention externally.
    # \param dLx (if set) new maximum change of the box length in the first lattice vector direction (in distance units)
    # \param dLy (if set) new maximum change of the box length in the second lattice vector direction (in distance units)
    # \param dLz (if set) new maximum change of the box length in the third lattice vector direction (in distance units)
    # \param dxy (if set) new maximum change of the X-Y tilt factor (dimensionless)
    # \param dxz (if set) new maximum change of the X-Z tilt factor (dimensionless)
    # \param dyz (if set) new maximum change of the Y-Z tilt factor as a function of time (dimensionless)
    # \param move_ratio ratio of trial volume change attempts to trial box shearing attempts
    # \param reduce Maximum number of lattice vectors of shear to allow before applying lattice reduction.
    #            Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable (default 0)
    #            Note that due to precision errors, lattice reduction may introduce small overlaps which can be resolved,
    #            but which temporarily break detailed balance.
    # \param isotropic Set to true to link dLx, dLy, and dLz. The dLy and dLz parameters are then ignored.
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # box_update = hpmc.update.npt(mc, P=10., dLx = 0.01, period = 10)
    # box_update.set_params(P=20.)
    # \endcode
    #
    # \returns None. Returns early if sanity check fails
    #
    def set_params(self, P=None, dLx=None, dLy = None, dLz = None, dxy=None, dxz=None, dyz=None, move_ratio=None, reduce=None, isotropic=None):
        hoomd.util.print_status_line();
        self.check_initialization();

        noop = True;

        if P is not None:
            self.P = hoomd.variant._setup_variant_input(P)
            noop = False;
        if dLx is not None:
            self.dLx = float(dLx)
            noop = False;
        if dLy is not None:
            self.dLy = float(dLy)
            noop = False;
        if dLz is not None:
            self.dLz = float(dLz)
            noop = False;
        if dxy is not None:
            self.dxy = float(dxy)
            noop = False;
        if dxz is not None:
            self.dxz = float(dxz)
            noop = False;
        if dyz is not None:
            self.dyz = float(dyz)
            noop = False;
        if move_ratio is not None:
            self.move_ratio = float(move_ratio)
            noop = False;
        if reduce is not None:
            self.reduce = float(reduce)
            noop = False;
        if isotropic is not None:
            self.isotropic = bool(isotropic)
            noop = False;
        if noop:
            hoomd.context.msg.warning("update.npt: No parameters changed\n");
            return;

        self.cpp_updater.setParams( self.P.cpp_variant,
                                    self.dLx,
                                    self.dLy,
                                    self.dLz,
                                    self.dxy,
                                    self.dxz,
                                    self.dyz,
                                    self.move_ratio,
                                    self.reduce,
                                    self.isotropic);

    ## Get npt parameters
    # \param timestep Timestep at which to evaluate variants (or the current step if None)
    #
    # \returns Dictionary of parameters values at the current or indicated timestep.
    # The dictionary contains the keys (P, dLx, dLy, dLz, dxy, dxz, dyz, move_ratio, reduce, isotropic), which mirror the same parameters to
    # set_params()
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, period = 10)
    # run(100)
    # params = box_update.get_params(1000)
    # P = params['P']
    # params = box_update.get_params()
    # dLx = params['dLx']
    # ~~~~~~~~~~~~
    #
    def get_params(self, timestep=None):
        if timestep is None:
            timestep = get_step()
        P = self.cpp_updater.getP()
        dLx = self.cpp_updater.getdLx()
        dLy = self.cpp_updater.getdLy()
        dLz = self.cpp_updater.getdLz()
        dxy = self.cpp_updater.getdxy()
        dxz = self.cpp_updater.getdxz()
        dyz = self.cpp_updater.getdyz()
        move_ratio = self.cpp_updater.getMoveRatio()
        reduce = self.cpp_updater.getReduce()
        isotropic = self.cpp_updater.getIsotropic()
        ret_val = dict(
                  P=P.getValue(timestep),
                  dLx=dLx,
                  dLy=dLy,
                  dLz=dLz,
                  dxy=dxy,
                  dxz=dxz,
                  dyz=dyz,
                  move_ratio=move_ratio,
                  reduce=reduce,
                  isotropic=isotropic
                  )
        return ret_val

    ## Get pressure parameter
    # \param timestep Timestep at which to evaluate variants (or the current step if None)
    #
    # \returns pressure value at the current or indicated timestep
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, period = 10)
    # run(100)
    # P_now = box_update.get_P()
    # P_future = box_update.get_P(1000)
    # ~~~~~~~~~~~~
    #
    def get_P(self, timestep=None):
        if timestep is None:
            timestep = get_step()
        P = self.cpp_updater.getP()
        return P.getValue(timestep)

    ## Get dLx parameter
    #
    # \returns max trial dLx change
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, period = 10)
    # run(100)
    # dLx_now = box_update.get_dLx()
    # ~~~~~~~~~~~~
    #
    def get_dLx(self):
        dLx = self.cpp_updater.getdLx()
        return dLx

    ## Get dLy parameter
    #
    # \returns max trial dLy change
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLy = 0.01, period = 10)
    # run(100)
    # dLy_now = box_update.get_dLy()
    # ~~~~~~~~~~~~
    #
    def get_dLy(self):
        dLy = self.cpp_updater.getdLy()
        return dLy

    ## Get dLz parameter
    #
    # \returns max trial dLz change
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLz = 0.01, period = 10)
    # run(100)
    # dLz_now = box_update.get_dLz()
    # ~~~~~~~~~~~~
    #
    def get_dLz(self):
        dLz = self.cpp_updater.getdLz()
        return dLz

    ## Get dxy parameter
    #
    # \returns max trial dxy change
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, dLy=0.01, dxy=0.01 period = 10)
    # run(100)
    # dxy_now = box_update.get_dxy()
    # ~~~~~~~~~~~~
    #
    def get_dxy(self):
        dxy = self.cpp_updater.getdxy()
        return dxy

    ## Get dxz parameter
    #
    # \returns max trial dxz change
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, dLy=0.01, dxz=0.01 period = 10)
    # run(100)
    # dxz_now = box_update.get_dxz()
    # ~~~~~~~~~~~~
    #
    def get_dxz(self):
        dxz = self.cpp_updater.getdxz()
        return dxz

    ## Get dyz parameter
    #
    # \returns max trial dyz change at the current or indicated timestep
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, dLy=0.01, dyz=0.01 period = 10)
    # run(100)
    # dyz_now = box_update.get_dyz()
    # ~~~~~~~~~~~~
    #
    def get_dyz(self):
        dyz = self.cpp_updater.getdyz()
        return dyz

    ## Get move_ratio parameter
    #
    # \returns fraction of box moves to attempt as volume changes versus box shearing
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # P = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 1e2)])
    # box_update = hpmc.update.npt(mc, P=P, dLx = 0.01, period = 10)
    # run(100)
    # ratio_now = box_update.get_move_ratio()
    # ~~~~~~~~~~~~
    #
    def get_move_ratio(self):
        move_ratio = self.cpp_updater.getMoveRatio()
        return move_ratio

    ## Get the average acceptance ratio for volume changing moves
    #
    # \returns The average volume change acceptance for the last run
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # box_update = hpmc.update.npt(mc, P=10., dLx = 0.01, period = 10)
    # run(100)
    # v_accept = box_update.get_volume_acceptance()
    # ~~~~~~~~~~~~
    #
    def get_volume_acceptance(self):
        counters = self.cpp_updater.getCounters(1);
        return counters.getVolumeAcceptance();

    ## Get the average acceptance ratio for shear changing moves
    #
    # \returns The average shear change acceptance for the last run
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.shape(..);
    # mc.shape_param[name].set(....);
    # box_update = hpmc.update.npt(mc, P=10., dLx = 0.01, dxy=0.01 period = 10)
    # run(100)
    # v_accept = box_update.get_shear_acceptance()
    # ~~~~~~~~~~~~
    #
    def get_shear_acceptance(self):
        counters = self.cpp_updater.getCounters(1);
        return counters.getShearAcceptance();

    ## Enables the updater
    # \returns nothing
    # \b Examples:
    # ~~~~~~~~~~~~
    # npt_updater.set_params(isotropic=True)
    # run(1e5)
    # npt_updater.disable()
    # update.box_resize(dLy = 10)
    # npt_updater.enable()
    # run(1e5)
    # ~~~~~~~~~~~~
    # See updater base class documentation for more information
    def enable(self):
        self.cpp_updater.computeAspectRatios();
        _updater.enable(self);

## Apply wall updates with a user-provided python callback
#
# Every \a period steps, a walls update move is tried with probability \a move_ratio. This update move is provided by the \a py_updater callback.
# The python callback must be a function of the timestep of the simulation. It must actually update the compute.wall mirror class, and therefore the c++ class.
# Then, update.wall only accepts an update move provided by the python callback if it maintains confinement conditions associated with all walls. Otherwise,
# it reverts back to a non-updated copy of the walls.
#
# Once initialized, the update provides the following log quantities that can be logged via analyze.log:
# **hpmc_wall_acceptance_ratio** -- the acceptance ratio for wall update moves
# \b Example:
# \code
# mc = hpmc.integrate.sphere(seed = 415236);
# ext_wall = hpmc.compute.wall(mc);
# ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
# def perturb(timestep):
#   r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
#   ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
# wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
# log = analyze.log(quantities=['hpmc_wall_acceptance_ratio'], period=100, filename='log.dat', overwrite=True);
# \endcode
class wall(_updater):
    ## Specifies the wall updater
    #
    # \param mc MC integrator (don't specify a new integrator, wall will continue to use the old one)
    # \param walls the compute.wall class instance to be updated
    # \param py_updater the python callback that performs the update moves. This must be a python method that is a function of the timestep of the simulation.
    # It must actually update the compute.wall mirror class, and therefore the c++ class.
    # \param move_ratio the probability with which an update move is attempted
    # \param seed the seed of the pseudo-random number generator that determines whether or not an update move is attempted
    # \param period the number of timesteps between update move attempt attempts
    #
    # \b Example:
    # \code
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # def perturb(timestep):
    #   r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
    #   ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
    # wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
    # \endcode
    def __init__(self, mc, walls, py_updater, move_ratio, seed, period=1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        cls = None;
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.UpdaterExternalFieldWallSphere;
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = integrate._get_sized_entry('UpdaterExternalFieldWallConvexPolyhedron', mc.max_verts);
        else:
            hoomd.context.msg.error("update.wall: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.wall");

        self.cpp_updater = cls(hoomd.context.current.system_definition, mc.cpp_integrator, walls.cpp_compute, py_updater, move_ratio, seed);
        self.setupUpdater(period);

    ## Get the number of accepted wall update moves
    #
    # \param mode integer that specifies the type of count to return. If mode!=0, return absolute quantities. If mode=0, return quantities relative to the start of the run.
    # DEFAULTS to 0.
    #
    # \returns the number of accepted wall update moves
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # def perturb(timestep):
    #   r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
    #   ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
    # wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
    # run(100);
    # acc_count = wall_updater.get_accepted_count(mode = 0);
    # ~~~~~~~~~~~~
    #
    def get_accepted_count(self, mode=0):
        return self.cpp_updater.getAcceptedCount(mode);

    ## Get the number of attempted wall update moves
    #
    # \param mode integer that specifies the type of count to return. If mode!=0, return absolute quantities. If mode=0, return quantities relative to the start of the run.
    # DEFAULTS to 0.
    #
    # \returns the number of attempted wall update moves
    #
    # \par Quick Example
    # ~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed = 415236);
    # ext_wall = hpmc.compute.wall(mc);
    # ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
    # def perturb(timestep):
    #   r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
    #   ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
    # wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
    # run(100);
    # tot_count = wall_updater.get_total_count(mode = 0);
    # ~~~~~~~~~~~~
    #
    def get_total_count(self, mode=0):
        return self.cpp_updater.getTotalCount(mode);

## Insert and remove particles in the muVT ensemble
#
# The muVT (or grand-canonical) ensemble simulates a system at constant fugacity.
#
# Gibbs ensemble simulations are also supported, where particles and volume are swapped between two or more
# boxes.  Every box correspond to one MPI partition, and can therefore run on multiple ranks.
# \sa hoomd_script.comm and the --nrank command line option for how to split a MPI task into partitions.
#
# \note Multiple Gibbs ensembles are also supported in a single parallel job, with the ngibbs option
# to update.muvt(), where the number of partitions can be a multiple of ngibbs.
#
class muvt(_updater):
    ## Specifies the muVT ensemble
    # \param mc MC integrator (don't specify a new integrator, muvt will continue to use the old one)
    # \param period Number of timesteps between histogram evaluations
    # \param transfer_types List of type names that are being transfered from/to the reservoir or between boxes (if *None*, all types)
    # \param seed The seed of the pseudo-random number generator (Needs to be the same across partitions of the same Gibbs ensemble)
    # \param ngibbs The number of partitions to use in Gibbs ensemble simulations (if == 1, perform grand canonical muVT)
    #
    # \par Quick Examples:
    # ~~~~~~~~~~~~~
    # mc = hpmc.integrate.sphere(seed=415236)
    # update.muvt(mc=mc, period)
    # ~~~~~~~~~~~~~
    def __init__(self, mc, period=1, transfer_types=None,seed=48123,ngibbs=1):
        hoomd.util.print_status_line();

        if not isinstance(mc, integrate.mode_hpmc):
            hoomd.context.msg.warning("update.muvt: Must have a handle to an HPMC integrator.\n");
            return;

        self.mc = mc

        # initialize base class
        _updater.__init__(self);

        if ngibbs > 1:
            self.gibbs = True;
        else:
            self.gibbs = False;

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # by default, transfer all types
        if transfer_types is None:
            transfer_types = type_list

        cls = None;
        if self.mc.implicit is True:
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.UpdaterMuVTImplicitSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.UpdaterMuVTImplicitConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.UpdaterMuVTImplicitSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('UpdaterMuVTImplicitConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('UpdaterMuVTImplicitSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.UpdaterMuVTImplicitEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.UpdaterMuVTImplicitSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.UpdaterMuVTImplicitFacetedSphere;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.UpdaterMuVTImplicitSphereUnion;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.UpdaterMuVTImplicitPolyhedron;
            else:
                hoomd.context.msg.error("update.muvt: Unsupported integrator.\n");
                raise RuntimeError("Error initializing update.muvt");
        else:
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.UpdaterMuVTSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.UpdaterMuVTConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.UpdaterMuVTSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('UpdaterMuVTConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('UpdaterMuVTSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.UpdaterMuVTEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.UpdaterMuVTSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.UpdaterMuVTFacetedSphere;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.UpdaterMuVTSphereUnion;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.UpdaterMuVTPolyhedron;
            else:
                hoomd.context.msg.error("update.muvt: Unsupported integrator.\n");
                raise RuntimeError("Error initializing update.muvt");

        if self.mc.implicit:
            self.cpp_updater = cls(hoomd.context.current.system_definition,
                                   mc.cpp_integrator,
                                   int(seed),
                                   ngibbs);
        else:
            self.cpp_updater = cls(hoomd.context.current.system_definition,
                                   mc.cpp_integrator,
                                   int(seed),
                                   ngibbs);

        # register the muvt updater
        self.setupUpdater(period);

        # set the list of transfered types
        if not isinstance(transfer_types,list):
            hoomd.context.msg.error("update.muvt: Need list of types to transfer.\n");
            raise RuntimeError("Error initializing update.muvt");

        cpp_transfer_types = _hoomd.std_vector_uint();
        for t in transfer_types:
            if t not in type_list:
                hoomd.context.msg.error("Trying to transfer unknown type " + str(t) + "\n");
                raise RuntimeError("Error setting muVT parameters");
            else:
                type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(t);

            cpp_transfer_types.append(type_id)

        self.cpp_updater.setTransferTypes(cpp_transfer_types)

    ## Change muVT fugacities
    #
    # \param type Particle type to set parameters for
    # \param fugacity Fugacity of this particle type (dimension of volume^-1)
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # muvt = hpmc.update.muvt(mc, period = 10)
    # muvt.set_fugacity(type='A',fugacity=1.23)
    # variant = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 4.56)])
    # muvt.set_fugacity(type='A', fugacity=variant)
    # \endcode
    #
    # \returns None. Returns early if sanity check fails
    #
    def set_fugacity(self, type, fugacity):
        hoomd.util.print_status_line();
        self.check_initialization();

        if self.gibbs:
            raise RuntimeError("Gibbs ensemble does not support setting the fugacity.\n");

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if type not in type_list:
            hoomd.context.msg.error("Trying to set fugacity for unknown type " + str(type) + "\n");
            raise RuntimeError("Error setting muVT parameters");
        else:
            type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        fugacity_variant = hoomd.variant._setup_variant_input(fugacity);
        self.cpp_updater.setFugacity(type_id, fugacity_variant.cpp_variant);

    ## Set muVT parameters
    #
    # \param dV (if set) Set volume rescaling factor (dimensionless)
    # \param move_ratio (if set) Set the ratio between volume and exchange/transfer moves (applies to Gibbs ensemble)
    # \param transfer_ratio (if set) Set the ratio between transfer and exchange moves
    #
    # To change the parameters of an existing updater, you must have saved it when it was specified.
    # \code
    # muvt = hpmc.update.muvt(mc, period = 10)
    # muvt.set_params(dV=0.1)
    # muvt.set_params(n_trial=2)
    # muvt.set_params(move_ratio=0.05)
    # \endcode
    #
    def set_params(self, dV=None, move_ratio=None, transfer_ratio=None):
        hoomd.util.print_status_line();
        self.check_initialization();

        if move_ratio is not None:
            if not self.gibbs:
                hoomd.context.msg.warning("Move ratio only used in Gibbs ensemble.\n");
            self.cpp_updater.setMoveRatio(float(move_ratio))

        if dV is not None:
            if not self.gibbs:
                hoomd.context.msg.warning("Parameter dV only available for Gibbs ensemble.\n");
            self.cpp_updater.setMaxVolumeRescale(float(dV))
        if transfer_ratio is not None:
            self.cpp_updater.setTransferRatio(float(transfer_ratio))

## remove the center of mass drift from a system restrained on a lattice.
#
# The command hpmc.update.remove_drift sets up an updater that removes the center of mass
# drift of a system every period timesteps
class remove_drift(_updater):
    # \param mc MC integrator (don't specify a new integrator later, lattice will continue to use the old one)
    # \param external_lattice lattice field where the lattice is defined.
    # \param period the period to call the updater
    # \b Example:
    # \code
    # mc = hpmc.integrate.convex_polyhedron(seed=seed);
    # mc.shape_param.set("A", vertices=verts)
    # mc.set_params(d=0.005, a=0.005)
    # lattice = hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
    # remove_drift = update.remove_drift(mc=mc, external_lattice=lattice, period=1000);
    # \endcode
    def __init__(self, mc, external_lattice, period=1):
        hoomd.util.print_status_line();
        #initiliaze base class
        _updater.__init__(self);
        cls = None;
        if not context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.RemoveDriftUpdaterSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.RemoveDriftUpdaterConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.RemoveDriftUpdaterSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = integrate._get_sized_entry('RemoveDriftUpdaterConvexPolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = integrate._get_sized_entry('RemoveDriftUpdaterSpheropolyhedron', mc.max_verts);
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.RemoveDriftUpdaterEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.RemoveDriftUpdaterSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.RemoveDriftUpdaterFacetedSphere;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.RemoveDriftUpdaterPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.RemoveDriftUpdaterSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.RemoveDriftUpdaterSphereUnion;
            else:
                hoomd.context.msg.error("update.remove_drift: Unsupported integrator.\n");
                raise RuntimeError("Error initializing update.remove_drift");
        else:
            raise RuntimeError("update.remove_drift: Error! GPU not implemented.");
            # if isinstance(mc, integrate.sphere):
            #     cls = _hpmc.RemoveDriftUpdaterGPUSphere;
            # elif isinstance(mc, integrate.convex_polygon):
            #     cls = _hpmc.RemoveDriftUpdaterGPUConvexPolygon;
            # elif isinstance(mc, integrate.simple_polygon):
            #     cls = _hpmc.RemoveDriftUpdaterGPUSimplePolygon;
            # elif isinstance(mc, integrate.convex_polyhedron):
            #     cls = integrate._get_sized_entry('RemoveDriftUpdaterGPUConvexPolyhedron', mc.max_verts);
            # elif isinstance(mc, integrate.convex_spheropolyhedron):
            #     cls = integrate._get_sized_entry('RemoveDriftUpdaterGPUSpheropolyhedron',mc.max_verts);
            # elif isinstance(mc, integrate.ellipsoid):
            #     cls = _hpmc.RemoveDriftUpdaterGPUEllipsoid;
            # elif isinstance(mc, integrate.convex_spheropolygon):
            #     cls =_hpmc.RemoveDriftUpdaterGPUSpheropolygon;
            # elif isinstance(mc, integrate.faceted_sphere):
            #     cls =_hpmc.RemoveDriftUpdaterGPUFacetedSphere;
            # elif isinstance(mc, integrate.polyhedron):
            #     cls =_hpmc.RemoveDriftUpdaterGPUPolyhedron;
            # elif isinstance(mc, integrate.sphinx):
            #     cls =_hpmc.RemoveDriftUpdaterGPUSphinx;
            # elif isinstance(mc, integrate.sphere_union):
            #     cls =_hpmc.RemoveDriftUpdaterGPUSphereUnion;
            # else:
            #     hoomd.context.msg.error("update.remove_drift: Unsupported integrator.\n");
            #     raise RuntimeError("Error initializing update.remove_drift");

        self.cpp_updater = cls(hoomd.context.current.system_definition, external_lattice.cpp_compute, mc.cpp_integrator);
        self.setupUpdater(period);
