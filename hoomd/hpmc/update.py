# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
R""" HPMC updaters.

Defines new ensembles and updaters to HPMC specifc data structures

"""

from . import _hpmc
from . import integrate
from . import compute
from hoomd import _hoomd

import math

from hoomd.update import _updater
import hoomd

class boxmc(_updater):
    R""" Apply box updates to sample isobaric and related ensembles.

    Args:

        mc (:py:mod:`hoomd.hpmc.integrate`): HPMC integrator object for system on which to apply box updates
        betaP (:py:class:`float` or :py:mod:`hoomd.variant`): :math:`\frac{p}{k_{\mathrm{B}}T}`. (units of inverse area in 2D or
                                                    inverse volume in 3D) Apply your chosen reduced pressure convention
                                                    externally.
        seed (int): random number seed for MC box changes

    One or more Monte Carlo move types are applied to evolve the simulation box. By default, no moves are applied.
    Activate desired move types using the following methods with a non-zero weight:

    - :py:meth:`aspect` - box aspect ratio moves
    - :py:meth:`length` - change box lengths independently
    - :py:meth:`shear` - shear the box
    - :py:meth:`volume` - scale the box lengths uniformly
    - :py:meth:`ln_volume` - scale the box lengths uniformly with logarithmic increments

    Pressure inputs to update.boxmc are defined as :math:`\beta P`. Conversions from a specific definition of reduced
    pressure :math:`P^*` are left for the user to perform.

    Note:
        All *delta* and *weight* values for all move types default to 0.

    Example::

        mc = hpmc.integrate.sphere(seed=415236, d=0.3)
        boxMC = hpmc.update.boxmc(mc, betaP=1.0, seed=9876)
        boxMC.set_betap(2.0)
        boxMC.ln_volume(delta=0.01, weight=2.0)
        boxMC.length(delta=(0.1,0.1,0.1), weight=4.0)
        run(30) # perform approximately 10 volume moves and 20 length moves

    """
    def __init__(self, mc, betaP, seed):
        hoomd.util.print_status_line();
        # initialize base class
        _updater.__init__(self);

        # Updater gets called at every timestep. Whether to perform a move is determined independently
        # according to frequency parameter.
        period = 1

        if not isinstance(mc, integrate.mode_hpmc):
            hoomd.context.msg.warning("update.boxmc: Must have a handle to an HPMC integrator.\n");
            return;

        self.betaP = hoomd.variant._setup_variant_input(betaP);

        self.seed = int(seed)

        # create the c++ mirror class
        self.cpp_updater = _hpmc.UpdaterBoxMC(hoomd.context.current.system_definition,
                                               mc.cpp_integrator,
                                               self.betaP.cpp_variant,
                                               1,
                                               self.seed,
                                               );
        self.setupUpdater(period);

        self.volume_delta = 0.0;
        self.volume_weight = 0.0;
        self.ln_volume_delta = 0.0;
        self.ln_volume_weight = 0.0;
        self.length_delta = [0.0, 0.0, 0.0];
        self.length_weight = 0.0;
        self.shear_delta = [0.0, 0.0, 0.0];
        self.shear_weight = 0.0;
        self.shear_reduce = 0.0;
        self.aspect_delta = 0.0;
        self.aspect_weight = 0.0;

        self.metadata_fields = ['betaP',
                                 'seed',
                                 'volume_delta',
                                 'volume_weight',
                                 'ln_volume_delta',
                                 'ln_volume_weight',
                                 'length_delta',
                                 'length_weight',
                                 'shear_delta',
                                 'shear_weight',
                                 'shear_reduce',
                                 'aspect_delta',
                                 'aspect_weight']

    def set_betap(self, betaP):
        R""" Update the pressure set point for Metropolis Monte Carlo volume updates.

        Args:
            betaP (float) or (:py:mod:`hoomd.variant`): :math:`\frac{p}{k_{\mathrm{B}}T}`. (units of inverse area in 2D or
                inverse volume in 3D) Apply your chosen reduced pressure convention
                externally.
        """
        self.betaP = hoomd.variant._setup_variant_input(betaP)
        self.cpp_updater.setP(self.betaP.cpp_variant)

    def volume(self, delta=None, weight=None):
        R""" Enable/disable isobaric volume move and set parameters.

        Args:
            delta (float): maximum change of the box area (2D) or volume (3D).
            weight (float): relative weight of this box move type relative to other box move types. 0 disables this move type.

        Sample the isobaric distribution of box volumes by rescaling the box.

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            box_update.volume(delta=0.01)
            box_update.volume(delta=0.01, weight=2)
            box_update.volume(delta=0.01, weight=0.15)

        Returns:
            A :py:class:`dict` with the current values of *delta* and *weight*.

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if weight is not None:
            self.volume_weight = float(weight)

        if delta is not None:
            self.volume_delta = float(delta)

        self.cpp_updater.volume(self.volume_delta, self.volume_weight);
        return {'delta': self.volume_delta, 'weight': self.volume_weight};

    def ln_volume(self, delta=None, weight=None):
        R""" Enable/disable isobaric volume move and set parameters.

        Args:
            delta (float): maximum change of **ln(V)** (where V is box area (2D) or volume (3D)).
            weight (float): relative weight of this box move type relative to other box move types. 0 disables this move type.

        Sample the isobaric distribution of box volumes by rescaling the box.

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            box_update.ln_volume(delta=0.001)
            box_update.ln_volume(delta=0.001, weight=2)
            box_update.ln_volume(delta=0.001, weight=0.15)

        Returns:
            A :py:class:`dict` with the current values of *delta* and *weight*.

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if weight is not None:
            self.ln_volume_weight = float(weight)

        if delta is not None:
            self.ln_volume_delta = float(delta)

        self.cpp_updater.ln_volume(self.ln_volume_delta, self.ln_volume_weight);
        return {'delta': self.ln_volume_delta, 'weight': self.ln_volume_weight};

    def length(self, delta=None, weight=None):
        R""" Enable/disable isobaric box dimension move and set parameters.

        Args:
            delta (:py:class:`float` or :py:class:`tuple`): maximum change of the box thickness for each pair of parallel planes
                                               connected by the corresponding box edges. I.e. maximum change of
                                               HOOMD-blue box parameters Lx, Ly, Lz. A single float *x* is equivalent to
                                               (*x*, *x*, *x*).
            weight (float): relative weight of this box move type relative to other box move types. 0 disables this
                            move type.

        Sample the isobaric distribution of box dimensions by rescaling the plane-to-plane distance of box faces,
        Lx, Ly, Lz (see :ref:`boxdim`).

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            box_update.length(delta=(0.01, 0.01, 0.0)) # 2D box changes
            box_update.length(delta=(0.01, 0.01, 0.01), weight=2)
            box_update.length(delta=0.01, weight=2)
            box_update.length(delta=(0.10, 0.01, 0.01), weight=0.15) # sample Lx more aggressively

        Returns:
            A :py:class:`dict` with the current values of *delta* and *weight*.

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if weight is not None:
            self.length_weight = float(weight)

        if delta is not None:
            if isinstance(delta, float) or isinstance(delta, int):
                self.length_delta = [float(delta)] * 3
            else:
                self.length_delta = [ float(d) for d in delta ]

        self.cpp_updater.length(   self.length_delta[0], self.length_delta[1],
                                        self.length_delta[2], self.length_weight);
        return {'delta': self.length_delta, 'weight': self.length_weight};

    def shear(self,  delta=None, weight=None, reduce=None):
        R""" Enable/disable box shear moves and set parameters.

        Args:
            delta (tuple): maximum change of the box tilt factor xy, xz, yz.
            reduce (float): Maximum number of lattice vectors of shear to allow before applying lattice reduction.
                    Shear of +/- 0.5 cannot be lattice reduced, so set to a value < 0.5 to disable (default 0)
                    Note that due to precision errors, lattice reduction may introduce small overlaps which can be
                    resolved, but which temporarily break detailed balance.
            weight (float): relative weight of this box move type relative to other box move types. 0 disables this
                            move type.

        Sample the distribution of box shear by adjusting the HOOMD-blue tilt factor parameters xy, xz, and yz.
        (see :ref:`boxdim`).

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            box_update.shear(delta=(0.01, 0.00, 0.0)) # 2D box changes
            box_update.shear(delta=(0.01, 0.01, 0.01), weight=2)
            box_update.shear(delta=(0.10, 0.01, 0.01), weight=0.15) # sample xy more aggressively

        Returns:
            A :py:class:`dict` with the current values of *delta*, *weight*, and *reduce*.

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if weight is not None:
            self.shear_weight = float(weight)

        if reduce is not None:
            self.shear_reduce = float(reduce)

        if delta is not None:
            if isinstance(delta, float) or isinstance(delta, int):
                self.shear_delta = [float(delta)] * 3
            else:
                self.shear_delta = [ float(d) for d in delta ]

        self.cpp_updater.shear(    self.shear_delta[0], self.shear_delta[1],
                                        self.shear_delta[2], self.shear_reduce,
                                        self.shear_weight);
        return {'delta': self.shear_delta, 'weight': self.shear_weight, 'reduce': self.shear_reduce}

    def aspect(self, delta=None, weight=None):
        R""" Enable/disable aspect ratio move and set parameters.

        Args:
            delta (float): maximum relative change of aspect ratio
            weight (float): relative weight of this box move type relative to other box move types. 0 disables this
                            move type.

        Rescale aspect ratio along a randomly chosen dimension.

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            box_update.aspect(delta=0.01)
            box_update.aspect(delta=0.01, weight=2)
            box_update.aspect(delta=0.01, weight=0.15)

        Returns:
            A :py:class:`dict` with the current values of *delta*, and *weight*.

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if weight is not None:
            self.aspect_weight = float(weight)

        if delta is not None:
            self.aspect_delta = float(delta)

        self.cpp_updater.aspect(self.aspect_delta, self.aspect_weight);
        return {'delta': self.aspect_delta, 'weight': self.aspect_weight}

    def get_volume_acceptance(self):
        R""" Get the average acceptance ratio for volume changing moves.

        Returns:
            The average volume change acceptance for the last run

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            box_update = hpmc.update.boxmc(mc, betaP=10, seed=1)
            run(100)
            v_accept = box_update.get_volume_acceptance()

        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getVolumeAcceptance();

    def get_ln_volume_acceptance(self):
        R""" Get the average acceptance ratio for log(V) changing moves.

        Returns:
            The average volume change acceptance for the last run

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            box_update = hpmc.update.boxmc(mc, betaP=10, seed=1)
            run(100)
            v_accept = box_update.get_ln_volume_acceptance()

        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getLogVolumeAcceptance();

    def get_shear_acceptance(self):
        R"""  Get the average acceptance ratio for shear changing moves.

        Returns:
           The average shear change acceptance for the last run

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            box_update = hpmc.update.boxmc(mc, betaP=10, seed=1)
            run(100)
            s_accept = box_update.get_shear_acceptance()

        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getShearAcceptance();
        counters = self.cpp_updater.getCounters(1);
        return counters.getShearAcceptance();

    def get_aspect_acceptance(self):
        R"""  Get the average acceptance ratio for aspect changing moves.

        Returns:
            The average aspect change acceptance for the last run

        Example::

            mc = hpmc.integrate.shape(..);
            mc_shape_param[name].set(....);
            box_update = hpmc.update.boxmc(mc, betaP=10, seed=1)
            run(100)
            a_accept = box_update.get_aspect_acceptance()

        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getAspectAcceptance();
        counters = self.cpp_updater.getCounters(1);
        return counters.getAspectAcceptance();

    def enable(self):
        R""" Enables the updater.

        Example::

            box_updater.set_params(isotropic=True)
            run(1e5)
            box_updater.disable()
            update.box_resize(dLy = 10)
            box_updater.enable()
            run(1e5)

        See updater base class documentation for more information
        """
        self.cpp_updater.computeAspectRatios();
        _updater.enable(self);

class wall(_updater):
    R""" Apply wall updates with a user-provided python callback.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        walls (:py:class:`hoomd.hpmc.field.wall`): the wall class instance to be updated
        py_updater (`callable`): the python callback that performs the update moves. This must be a python method that is a function of the timestep of the simulation.
               It must actually update the :py:class:`hoomd.hpmc.field.wall`) managed object.
        move_ratio (float): the probability with which an update move is attempted
        seed (int): the seed of the pseudo-random number generator that determines whether or not an update move is attempted
        period (int): the number of timesteps between update move attempt attempts
               Every *period* steps, a walls update move is tried with probability *move_ratio*. This update move is provided by the *py_updater* callback.
               Then, update.wall only accepts an update move provided by the python callback if it maintains confinement conditions associated with all walls. Otherwise,
               it reverts back to a non-updated copy of the walls.

    Once initialized, the update provides the following log quantities that can be logged via :py:class:`hoomd.analyze.log`:

    * **hpmc_wall_acceptance_ratio** - the acceptance ratio for wall update moves

    Example::

        mc = hpmc.integrate.sphere(seed = 415236);
        ext_wall = hpmc.compute.wall(mc);
        ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
        def perturb(timestep):
          r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
          ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
        wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
        log = analyze.log(quantities=['hpmc_wall_acceptance_ratio'], period=100, filename='log.dat', overwrite=True);

    Example::

        mc = hpmc.integrate.sphere(seed = 415236);
        ext_wall = hpmc.compute.wall(mc);
        ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
        def perturb(timestep):
          r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
          ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
        wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);

    """
    def __init__(self, mc, walls, py_updater, move_ratio, seed, period=1):
        hoomd.util.print_status_line();

        # initialize base class
        _updater.__init__(self);

        cls = None;
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.UpdaterExternalFieldWallSphere;
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = _hpmc.UpdaterExternalFieldWallConvexPolyhedron;
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = _hpmc.UpdaterExternalFieldWallSpheropolyhedron;
        else:
            hoomd.context.msg.error("update.wall: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.wall");

        self.cpp_updater = cls(hoomd.context.current.system_definition, mc.cpp_integrator, walls.cpp_compute, py_updater, move_ratio, seed);
        self.setupUpdater(period);

    def get_accepted_count(self, mode=0):
        R""" Get the number of accepted wall update moves.

        Args:
            mode (int): specify the type of count to return. If mode!=0, return absolute quantities. If mode=0, return quantities relative to the start of the run.
                        DEFAULTS to 0.

        Returns:
           the number of accepted wall update moves

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            def perturb(timestep):
              r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
              ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
            wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
            run(100);
            acc_count = wall_updater.get_accepted_count(mode = 0);
        """
        return self.cpp_updater.getAcceptedCount(mode);

    def get_total_count(self, mode=0):
        R""" Get the number of attempted wall update moves.

        Args:
            mode (int): specify the type of count to return. If mode!=0, return absolute quantities. If mode=0, return quantities relative to the start of the run.
                        DEFAULTS to 0.

        Returns:
           the number of attempted wall update moves

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0], inside = True);
            def perturb(timestep):
              r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0, param = "rsq"));
              ext_wall.set_sphere_wall(index = 0, radius = 1.5*r, origin = [0, 0, 0], inside = True);
            wall_updater = hpmc.update.wall(mc, ext_wall, perturb, move_ratio = 0.5, seed = 27, period = 50);
            run(100);
            tot_count = wall_updater.get_total_count(mode = 0);

        """
        return self.cpp_updater.getTotalCount(mode);

class muvt(_updater):
    R""" Insert and remove particles in the muVT ensemble.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        seed (int): The seed of the pseudo-random number generator (Needs to be the same across partitions of the same Gibbs ensemble)
        period (int): Number of timesteps between histogram evaluations.
        transfer_types (list): List of type names that are being transferred from/to the reservoir or between boxes (if *None*, all types)
        ngibbs (int): The number of partitions to use in Gibbs ensemble simulations (if == 1, perform grand canonical muVT)

    The muVT (or grand-canonical) ensemble simulates a system at constant fugacity.

    Gibbs ensemble simulations are also supported, where particles and volume are swapped between two or more
    boxes.  Every box correspond to one MPI partition, and can therefore run on multiple ranks.
    See :py:mod:`hoomd.comm` and the --nrank command line option for how to split a MPI task into partitions.

    Note:
        Multiple Gibbs ensembles are also supported in a single parallel job, with the ngibbs option
        to update.muvt(), where the number of partitions can be a multiple of ngibbs.

    Example::

        mc = hpmc.integrate.sphere(seed=415236)
        update.muvt(mc=mc, period)

    """
    def __init__(self, mc, seed, period=1, transfer_types=None,ngibbs=1):
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
            if self.mc.depletant_mode == 'overlap_regions':
                if isinstance(mc, integrate.sphere):
                    cls = _hpmc.UpdaterMuVTImplicitSphere;
                elif isinstance(mc, integrate.convex_polygon):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolygon;
                elif isinstance(mc, integrate.simple_polygon):
                    cls = _hpmc.UpdaterMuVTImplicitSimplePolygon;
                elif isinstance(mc, integrate.convex_polyhedron):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolyhedron;
                elif isinstance(mc, integrate.convex_spheropolyhedron):
                    cls = _hpmc.UpdaterMuVTImplicitSpheropolyhedron;
                elif isinstance(mc, integrate.ellipsoid):
                    cls = _hpmc.UpdaterMuVTImplicitEllipsoid;
                elif isinstance(mc, integrate.convex_spheropolygon):
                    cls =_hpmc.UpdaterMuVTImplicitSpheropolygon;
                elif isinstance(mc, integrate.faceted_sphere):
                    cls =_hpmc.UpdaterMuVTImplicitFacetedEllipsoid;
                elif isinstance(mc, integrate.sphere_union):
                    cls = _hpmc.UpdaterMuVTImplicitSphereUnion;
                elif isinstance(mc, integrate.convex_polyhedron_union):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolyhedronUnion;
                elif isinstance(mc, integrate.faceted_ellipsoid_union):
                    cls = _hpmc.UpdaterMuVTImplicitFacetedEllipsoidUnion;
                elif isinstance(mc, integrate.polyhedron):
                    cls =_hpmc.UpdaterMuVTImplicitPolyhedron;
                elif isinstance(mc, integrate.sphinx):
                    cls =_hpmc.UpdaterMuVTImplicitSphinx;
                else:
                    hoomd.context.msg.error("update.muvt: Unsupported integrator.\n");
                    raise RuntimeError("Error initializing update.muvt");
            else:
                if isinstance(mc, integrate.sphere):
                    cls = _hpmc.UpdaterMuVTImplicitSphere;
                elif isinstance(mc, integrate.convex_polygon):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolygon;
                elif isinstance(mc, integrate.simple_polygon):
                    cls = _hpmc.UpdaterMuVTImplicitSimplePolygon;
                elif isinstance(mc, integrate.convex_polyhedron):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolyhedron;
                elif isinstance(mc, integrate.convex_spheropolyhedron):
                    cls = _hpmc.UpdaterMuVTImplicitSpheropolyhedron;
                elif isinstance(mc, integrate.ellipsoid):
                    cls = _hpmc.UpdaterMuVTImplicitEllipsoid;
                elif isinstance(mc, integrate.convex_spheropolygon):
                    cls =_hpmc.UpdaterMuVTImplicitSpheropolygon;
                elif isinstance(mc, integrate.faceted_sphere):
                    cls =_hpmc.UpdaterMuVTImplicitFacetedEllipsoid;
                elif isinstance(mc, integrate.sphere_union):
                    cls = _hpmc.UpdaterMuVTImplicitSphereUnion;
                elif isinstance(mc, integrate.convex_polyhedron_union):
                    cls = _hpmc.UpdaterMuVTImplicitConvexPolyhedronUnion;
                elif isinstance(mc, integrate.faceted_ellipsoid_union):
                    cls = _hpmc.UpdaterMuVTImplicitFacetedEllipsoidUnion;
                elif isinstance(mc, integrate.polyhedron):
                    cls =_hpmc.UpdaterMuVTImplicitPolyhedron;
                elif isinstance(mc, integrate.sphinx):
                    cls =_hpmc.UpdaterMuVTImplicitSphinx;
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
                cls = _hpmc.UpdaterMuVTConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.UpdaterMuVTSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.UpdaterMuVTEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.UpdaterMuVTSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.UpdaterMuVTFacetedEllipsoid;
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.UpdaterMuVTSphereUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls = _hpmc.UpdaterMuVTConvexPolyhedronUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.UpdaterMuVTFacetedEllipsoidUnion;
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

        # set the list of transferred types
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

    def set_fugacity(self, type, fugacity):
        R""" Change muVT fugacities.

        Args:
            type (str): Particle type to set parameters for
            fugacity (float): Fugacity of this particle type (dimension of volume^-1)

        Example::

            muvt = hpmc.update.muvt(mc, period=10)
            muvt.set_fugacity(type='A', fugacity=1.23)
            variant = hoomd.variant.linear_interp(points=[(0,1e1), (1e5, 4.56)])
            muvt.set_fugacity(type='A', fugacity=variant)

        """
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

    def set_params(self, dV=None, move_ratio=None, transfer_ratio=None):
        R""" Set muVT parameters.

        Args:
            dV (float): (if set) Set volume rescaling factor (dimensionless)
            move_ratio (float): (if set) Set the ratio between volume and exchange/transfer moves (applies to Gibbs ensemble)
            transfer_ratio (float): (if set) Set the ratio between transfer and exchange moves

        Example::

            muvt = hpmc.update.muvt(mc, period = 10)
            muvt.set_params(dV=0.1)
            muvt.set_params(n_trial=2)
            muvt.set_params(move_ratio=0.05)

        """
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

class remove_drift(_updater):
    R""" Remove the center of mass drift from a system restrained on a lattice.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        external_lattice (:py:class:`hoomd.hpmc.field.lattice_field`): lattice field where the lattice is defined.
        period (int): the period to call the updater

    The command hpmc.update.remove_drift sets up an updater that removes the center of mass
    drift of a system every period timesteps,

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=seed);
        mc.shape_param.set("A", vertices=verts)
        mc.set_params(d=0.005, a=0.005)
        lattice = hpmc.compute.lattice_field(mc=mc, position=fcc_lattice, k=1000.0);
        remove_drift = update.remove_drift(mc=mc, external_lattice=lattice, period=1000);

    """
    def __init__(self, mc, external_lattice, period=1):
        hoomd.util.print_status_line();
        #initialize base class
        _updater.__init__(self);
        cls = None;
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if isinstance(mc, integrate.sphere):
                cls = _hpmc.RemoveDriftUpdaterSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.RemoveDriftUpdaterConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.RemoveDriftUpdaterSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.RemoveDriftUpdaterConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.RemoveDriftUpdaterSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.RemoveDriftUpdaterEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.RemoveDriftUpdaterSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.RemoveDriftUpdaterFacetedEllipsoid;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.RemoveDriftUpdaterPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.RemoveDriftUpdaterSphinx;
            elif isinstance(mc, integrate.sphere_union):
                cls = _hpmc.RemoveDriftUpdaterSphereUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls = _hpmc.RemoveDriftUpdaterConvexPolyhedronUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.RemoveDriftUpdaterFacetedEllipsoidUnion;
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
            #     cls =_hpmc.RemoveDriftUpdaterGPUFacetedEllipsoid;
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

class shape_update(_updater):
    R"""
    Apply shape updates to the shape definitions defined in the integrator. This class should not be instaianted directly
    but the alchemy and elastic_shapes classes can be. Each updater defines a specific statistical ensemble. See the different
    updaters for documentaion on the specific acceptance criteria and examples.

    Right now the shape move will update the shape definitions for every type.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): HPMC integrator object for system on which to apply box updates
        move_ratio (:py:class:`float` or :py:mod:`hoomd.variant`): fraction of steps to run the updater.
        seed (int): random number seed for shape move generators
        period (int): the period to call the updater
        phase (int): When -1, start on the current time step. When >= 0, execute on steps where *(step + phase) % period == 0*.
        pretend (bool): When True the updater will not actually make update the shape definitions but moves will be proposed and
                        the acceptance statistics will be updated correctly
        pos (:py:mod:`hoomd.deprecated.dump.pos`): HOOMD POS analyzer object used to update the shape definitions on the fly
        setup_pos (bool): When True the updater will automatically update the POS analyzer if it is provided
        setup_callback (callable): will override the default pos callback. will be called everytime the pos file is written
        nselect (int): number of types to change every time the updater is called.
        nsweeps (int): number of times to change nselect types every time the updater is called.
        multi_phase (bool): when True MPI is enforced and shapes are updated together for two boxes.
        num_phase (int): how many boxes are simulated at the same time, now support 2 and 3.
    Note:
        Only one of the Monte Carlo move types are applied to evolve the particle shape definition. By default, no moves are applied.
        Activate desired move types using the following methods.

        - :py:meth:`python_shape_move` - supply a python call back that will take a list of parameters between 0 and 1 and return a shape param object.
        - :py:meth:`vertex_shape_move` - make changes to the the vertices of the shape definition. Currently only defined for convex polyhedra.
        - :py:meth:`constant_shape_move` - make a single move to a shape i.e. shape_old -> shape_new. Useful when pretend is set to True.
        - :py:meth:`elastic_shape_move` - scale and shear the particle definitions. Currently only defined for ellipsoids and convex polyhedra.

        See the documentation for the individual moves for more usage information.

    """
    _ids = [];
    def __init__(   self,
                    mc,
                    move_ratio,
                    seed,
                    period = 1,
                    phase=-1,
                    pretend=False,
                    pos=None,
                    setup_pos=True,
                    pos_callback=None,
                    nselect=1,
                    nsweeps=1,
                    multi_phase=False,
                    num_phase=2,
                    gsdid = None,
                    kappa_iq=0.0):
        _updater.__init__(self);

        cls = None;
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.UpdaterShapeSphere;
        elif isinstance(mc, integrate.convex_polygon):
            cls = _hpmc.UpdaterShapeConvexPolygon;
        elif isinstance(mc, integrate.simple_polygon):
            cls = _hpmc.UpdaterShapeSimplePolygon;
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = _hpmc.UpdaterShapeConvexPolyhedron;
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = _hpmc.UpdaterShapeSpheropolyhedron;
        elif isinstance(mc, integrate.ellipsoid):
            cls = _hpmc.UpdaterShapeEllipsoid;
        elif isinstance(mc, integrate.convex_spheropolygon):
            cls =_hpmc.UpdaterShapeSpheropolygon;
        elif isinstance(mc, integrate.patchy_sphere):
            cls =_hpmc.UpdaterShapePatchySphere;
        elif isinstance(mc, integrate.polyhedron):
            cls =_hpmc.UpdaterShapePolyhedron;
        elif isinstance(mc, integrate.sphinx):
            cls =_hpmc.UpdaterShapeSphinx;
        elif isinstance(mc, integrate.sphere_union):
            cls =_hpmc.UpdaterShapeSphereUnion;
        else:
            hoomd.context.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");
        self.cpp_updater = cls(hoomd.context.current.system_definition,
                                mc.cpp_integrator,
                                move_ratio,
                                seed,
                                nselect,
                                nsweeps,
                                pretend,
                                multi_phase,
                                num_phase,
                                kappa_iq);
        self.move_cpp = None;
        self.boltzmann_function = None;
        self.seed = seed;
        self.mc = mc;
        self.pos = pos;
        self._gsdid = None;
        # if gsdid in shape_update._ids:
        #     raise RuntimeError("gsdid already exists")
        if gsdid is not None:
            self._gsdid = gsdid;
            shape_update._ids.append(gsdid);

        if pos and setup_pos:
            if pos_callback is None:
                pos.set_info(self.pos_callback);
            else:
                pos.set_info(pos_callback);
        self.setupUpdater(period, phase);
        self.kappa_iq = kappa_iq

    def python_shape_move(self, callback, params, stepsize, param_ratio):
        R"""Enable python shape move and set parameters. All python shape moves must
        be callable object that take a single list of parameters between 0 and 1 as
        the call arguments and returns a shape parameter definition.

        Args:
            callback (callable): The python function that will be called each update.
            params (dict): dictionary of types and the corresponding list parameters ({'A' : [1.0], 'B': [0.0]})
            stepsize (float): step size in parameter space.
            param_ratio (float): average fraction of parameters to change each update

        Note:
            Parameters must be given for every particle type. Shape move should rescale the particle to have constnt
            volume if necessary/desired.

        Example::

            # example callback
            class convex_polyhedron_callback:
                def __init__(self, mc):
                    self.mc = mc;
                def __call__(self, params):
                    # do something with params and define verts
                    return self.mc.shape_class.make_param(vertices=verts);

            # now set up the updater
            shape_up = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)
            shape_up.python_shape_move( callback=convex_polyhedron_callback(mc), params={'A': [0.5, 0.5]}, stepsize=0.001, param_ratio=0.5)

        """
        hoomd.util.print_status_line();
        if(self.move_cpp):
            hoomd.context.msg.error("update.shape_update.python_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            move_cls = _hpmc.PythonShapeMoveSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            move_cls = _hpmc.PythonShapeMoveConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            move_cls = _hpmc.PythonShapeMoveSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = _hpmc.PythonShapeMoveConvexPolyhedron;
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            move_cls = _hpmc.PythonShapeMoveSpheropolyhedron;
        elif isinstance(self.mc, integrate.ellipsoid):
            move_cls = _hpmc.PythonShapeMoveEllipsoid;
        elif isinstance(self.mc, integrate.convex_spheropolygon):
            move_cls = _hpmc.PythonShapeMoveConvexSphereopolygon;
        elif isinstance(self.mc, integrate.polyhedron):
            move_cls = _hpmc.PythonShapeMovePolyhedron;
        elif isinstance(self.mc, integrate.sphinx):
            move_cls = _hpmc.PythonShapeMoveSphinx;
        elif isinstance(self.mc, integrate.sphere_union):
            move_cls = _hpmc.PythonShapeMoveSphereUnion;
        else:
            hoomd.context.msg.error("update.shape_update.python_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            hoomd.context.msg.error("update.shape_update.python_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        param_list = [];
        for i in range(ntypes):
            typ = hoomd.context.current.system_definition.getParticleData().getNameByType(i)
            param_list.append(self.mc.shape_class.ensure_list(params[typ]));


        if isinstance(stepsize, float) or isinstance(stepsize, int):
            stepsize_list = [float(stepsize) for i in range(ntypes) ];
        else:
            stepsize_list = self.mc.shape_class.ensure_list(stepsize);

        # TODO: Make this possible
        # Currently computing the moments of inertia for spheropolyhedra is not implemented
        # In order to prevent improper usage, we throw an error here. The use of this
        # updater with spheropolyhedra is currently enabled to allow the use of spherical
        # depletants
        if isinstance(self.mc, integrate.convex_spheropolyhedron):
            for i in range(ntypes):
                typename = hoomd.context.current.system_definition.getParticleData().getNameByType(i);
                shape = self.mc.shape_param.get(typename)
                if shape.sweep_radius != 0 and len(shape.vertices) != 0 and stepsize_list[i] != 0:
                    raise RuntimeError("Currently alchemical moves with integrate.convex_spheropolyhedron \
are only enabled for polyhedral and spherical particles.")

        self.move_cpp = move_cls(ntypes, callback, param_list, stepsize_list, float(param_ratio));
        self.cpp_updater.registerShapeMove(self.move_cpp);

    def vertex_shape_move(self, stepsize, param_ratio, volume=1.0, kappa_iq=0.0):
        R"""
        Enable vertex shape move and set parameters. Changes a particle shape by
        translating vertices and rescaling to have constant volume. The shape definition
        corresponds to the convex hull of the vertices.

        Args:
            volume (float): volume of the particles to hold constant
            stepsize (float): stepsize for each
            param_ratio (float): average fraction of vertices to change each update

        Example::

            shape_up = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)
            shape_up.vertex_shape_move( stepsize=0.001, param_ratio=0.25, volume=1.0)

        """
        hoomd.util.print_status_line();
        if(self.move_cpp):
            hoomd.context.msg.error("update.shape_update.vertex_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            pass;
        elif isinstance(self.mc, integrate.convex_polygon):
            pass;
        elif isinstance(self.mc, integrate.simple_polygon):
            pass;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = _hpmc.GeneralizedShapeMoveConvexPolyhedron;
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            pass;
        elif isinstance(self.mc, integrate.ellipsoid):
            pass;
        elif isinstance(self.mc, integrate.convex_spheropolygon):
            pass;
        elif isinstance(self.mc, integrate.patchy_sphere):
            pass;
        elif isinstance(self.mc, integrate.polyhedron):
            pass;
        elif isinstance(self.mc, integrate.sphinx):
            pass;
        elif isinstance(self.mc, integrate.sphere_union):
            pass;
        else:
            hoomd.context.msg.error("update.shape_update.vertex_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            hoomd.context.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        self.move_cpp = move_cls(ntypes, stepsize, param_ratio, volume);
        self.cpp_updater.registerShapeMove(self.move_cpp);

    def constant_shape_move(self, **shape_params):
        R"""
        Enable constant shape move and set parameters. Changes a particle shape by
        the same way every time the updater is called. This is useful for calculating
        a spcific transition probability and derived thermodynamic quantities.

        Args:
            shape_parameters: arguments required to define the reference shape. Depends on the
            integrator.

        Example::

            shape_up = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)
            # convex_polyhedron
            shape_up.constant_shape_move(vertices=verts)
            # ellipsoid
            shape_up.constant_shape_move(a=1, b=1, c=1);

        See Also:
            :py:mod:`hoomd.hpmc.data` for required shape parameters.

        """
        hoomd.util.print_status_line();
        if(self.move_cpp):
            hoomd.context.msg.error("update.shape_update.constant_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            move_cls = _hpmc.ConstantShapeMoveSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            move_cls = _hpmc.ConstantShapeMoveConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            move_cls = _hpmc.ConstantShapeMoveSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = _hpmc.ConstantShapeMoveConvexPolyhedron;
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            move_cls = _hpmc.ConstantShapeMoveSpheropolyhedron;
        elif isinstance(self.mc, integrate.ellipsoid):
            move_cls = _hpmc.ConstantShapeMoveEllipsoid;
        elif isinstance(self.mc, integrate.convex_spheropolygon):
            move_cls = _hpmc.ConstantShapeMoveConvexSphereopolygon;
        elif isinstance(self.mc, integrate.polyhedron):
            move_cls = _hpmc.ConstantShapeMovePolyhedron;
        elif isinstance(self.mc, integrate.sphinx):
            move_cls = _hpmc.ConstantShapeMoveSphinx;
        elif isinstance(self.mc, integrate.sphere_union):
            move_cls = _hpmc.ConstantShapeMoveSphereUnion;
        else:
            hoomd.context.msg.error("update.shape_update.constant_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            hoomd.context.msg.error("update.shape_update.constant_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        self.move_cpp = move_cls(ntypes, [self.mc.shape_class.make_param(**shape_params)]);
        self.cpp_updater.registerShapeMove(self.move_cpp);

    def elastic_shape_move(self, stepsize, param_ratio=0.5):
        R"""
        Enable scale and shear shape move and set parameters. Changes a particle shape by
        scaling the particle and shearing the particle.

        Args:
            stepsize (float): largest scaling/shearing factor used.
            move_ratio (float): fraction of scale to shear moves.

        Example::

            shape_up = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)
            shape_up.elastic_shape_move(stepsize=0.01)

        """
        hoomd.util.print_status_line();
        if(self.move_cpp):
            hoomd.context.msg.error("update.shape_update.elastic_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = _hpmc.ScaleShearShapeMoveConvexPolyhedron;
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            pass;
        elif isinstance(self.mc, integrate.ellipsoid):
            move_cls = _hpmc.ScaleShearShapeMoveEllipsoid;
        elif isinstance(self.mc, integrate.convex_spheropolygon):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveConvexSphereopolygon;
        elif isinstance(self.mc, integrate.polyhedron):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMovePolyhedron;
        elif isinstance(self.mc, integrate.sphinx):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveSphinx;
        elif isinstance(self.mc, integrate.sphere_union):
            pass;
            # move_cls = _hpmc.ScaleShearShapeMoveSphereUnion;
        else:
            hoomd.context.msg.error("update.shape_update.elastic_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            hoomd.context.msg.error("update.shape_update.elastic_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        self.move_cpp = move_cls(ntypes, stepsize, param_ratio);
        self.cpp_updater.registerShapeMove(self.move_cpp);

    def get_tuner(self, average = False, **kwargs):
        R""" Get a :py:mod:`hoomd.hpmc.util.tune` object set to tune the step size of the shape move.
        Args:
            average (bool): If set to true will set up the tuner to set all types together using averaged statistics.
            kwargs: keyword argments that will be passed to :py:mod:`hoomd.hpmc.util.tune`

        Example::

            shape_up = hpmc.update.elastic_shape(mc=mc, move_ratio=0.1, seed=3832765, stiffness=100.0, reference=dict(vertices=v), nselect=3)
            shape_up.elastic_shape_move(stepsize=0.1);
            tuner = shape_up.get_tuner(average=True); # average stats over all particle types.
            for _ in range(100):
                hoomd.run(1000, quiet=True);
                tuner.update();
                shape_up.reset_statistics(); # reset the shape stats

        """
        from . import util
        import numpy as np
        if not average:
            ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
            tunables = ["stepsize-{}".format(i) for i in range(ntypes)];
            tunable_map = {};
            for i in range(ntypes):
                tunable_map.update({'stepsize-{}'.format(i) : {
                                        'get': lambda typeid=i: getattr(self, 'get_step_size')(typeid),
                                        'acceptance': lambda typeid=i: getattr(self, 'get_move_acceptance')(typeid),
                                        'set': lambda x, name=hoomd.context.current.system_definition.getParticleData().getNameByType(i): getattr(self, 'set_params')(types=name, stepsize=x),
                                        'maximum': 0.5
                                        }})
        else:
            ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
            type_list = [hoomd.context.current.system_definition.getParticleData().getNameByType(i) for i in range(ntypes)]
            tunables=["stepsize"]
            tunable_map = {'stepsize' : {
                                    'get': lambda : getattr(self, 'get_step_size')(0),
                                    'acceptance': lambda : float(getattr(self, 'get_move_acceptance')(None)),
                                    'set': lambda x, name=type_list: getattr(self, 'set_params')(types=name, stepsize=x),
                                    'maximum': 0.5
                                    }};
        return util.tune(self, tunables=tunables, tunable_map=tunable_map, **kwargs);

    def get_total_count(self, typeid=None):
        R""" Get the total number of moves attempted by the updater
        Args:
            typeid (int): the typeid of the particle type. If None the sum over all types will be returned.
        Returns:
            The total number of moves attempted by the updater

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            total = shape_updater.get_total_count(0)

        """
        hoomd.util.print_status_line();
        if typeid is None:
            return sum([self.cpp_updater.getTotalCount(i) for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes())])
        else:
            return self.cpp_updater.getTotalCount(typeid);

    def get_accepted_count(self, typeid=None):
        R""" Get the total number of moves accepted by the updater
        Args:
            typeid (int): the typeid of the particle type. if None then the sum of all counts will be returned.
        Returns:
            The total number of moves accepted by the updater

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            accepted = shape_updater.get_accepted_count(0)
        """
        hoomd.util.print_status_line();
        if typeid is None:
            return sum([self.cpp_updater.getAcceptedCount(i) for i in range(hoomd.context.current.system_definition.getParticleData().getNTypes())])
        else:
            return self.cpp_updater.getAcceptedCount(typeid);

    def get_move_acceptance(self, typeid=0):
        R""" Get the acceptance ratio for a particle type
        Args:
            typeid (int): the typeid of the particle type
        Returns:
            The acceptance ratio for a particle type

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            ratio = shape_updater.get_move_acceptance(0)

        """
        hoomd.util.print_status_line();
        acc = 0.0;
        hoomd.util.quiet_status()
        if self.get_total_count(typeid):
            acc = float(self.get_accepted_count(typeid))/float(self.get_total_count(typeid));
        hoomd.util.unquiet_status()
        return acc;

    def get_step_size(self, typeid=0):
        R""" Get the shape move stepsize for a particle type

        Args:
            typeid (int): the typeid of the particle type
        Returns:
            The shape move stepsize for a particle type

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            stepsize = shape_updater.get_step_size(0)

        """
        hoomd.util.print_status_line();
        return self.cpp_updater.getStepSize(typeid);

    def reset_statistics(self):
        R""" Reset the acceptance statistics for the updater

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            shape_updater.reset_statistics()

        """
        hoomd.util.print_status_line();
        self.cpp_updater.resetStatistics();

    ## \internal
    # \brief default pos writer callback.
    # Declare the GSD state schema.
    def _gsd_state_name(self):
        if self._gsdid is None:
            raise RuntimeError("Must specify unique gsdid for gsd state.");

        return "state/hpmc/"+str(self.__class__.__name__)+str(self._gsdid)+"/";

    ## \internal
    # \brief default pos writer callback.
    def pos_callback(self, timestep):
        if self.pos:
            self.mc.setup_pos_writer(pos=self.pos);
        return "";

    def set_params(self, types, stepsize=None):
        R""" Reset the acceptance statistics for the updater
        Args:
            type (str): Particle type (string) or list of types
            stepsize (float): Shape move stepsize to set for each type

        Example::

            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            shape_updater.set_params('A', stepsize=0.01)
            shape_updater.set_params('B', stepsize=0.01)
            shape_updater.set_params(['A', 'B'], stepsize=0.01)
            run(100)

        """
        hoomd.util.print_status_line();
        if isinstance(types, str):
            types = [types];
        for name in types:
            typ = hoomd.context.current.system_definition.getParticleData().getTypeByName(name);
            if not stepsize is None:
                self.cpp_updater.setStepSize(typ, stepsize);


class alchemy(shape_update):
    R""" Apply shape updates to the shape definitions defined in the integrator.

    Args:
        params (dict): any of the other keyword arguments from :py:mod:`hoomd.hpmc.update.shape_update`

    Additional comments here. what enseble are we simulating etc.

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.5)
        alchem = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)

    """
    def __init__(   self,
                    **params):
        hoomd.util.print_status_line();
        # initialize base class
        shape_update.__init__(self, **params);
        boltzmann_cls = None;
        if isinstance(self.mc, integrate.sphere):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannConvexPolyhedron;
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannSpheropolyhedron;
        elif isinstance(self.mc, integrate.ellipsoid):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannEllipsoid;
        elif isinstance(self.mc, integrate.convex_spheropolygon):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSpheropolygon;
        elif isinstance(self.mc, integrate.patchy_sphere):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannPatchySphere;
        elif isinstance(self.mc, integrate.polyhedron):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannPolyhedron;
        elif isinstance(self.mc, integrate.sphinx):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSphinx;
        elif isinstance(self.mc, integrate.sphere_union):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSphereUnion;
        else:
            hoomd.context.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        self.boltzmann_function = boltzmann_cls();
        self.cpp_updater.registerLogBoltzmannFunction(self.boltzmann_function);

class elastic_shape(shape_update):
    R""" Apply shape updates to the shape definitions defined in the integrator.

    Args:
        reference (dict): dictionary of shape parameters. (same as `mc.shape_param.set(....)`)
        stiffness (float): stiffness of the particle spring
        params (dict): any of the other keyword arguments to be passed to :py:mod:`hoomd.hpmc.update.shape_update`

    Additional comments here. what enseble are we simulating etc.

    Note on move functions and acceptance criterion:
        explain how to write the function here.

    Example::

        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.5)
        elastic = hpmc.update.elastic_shape(mc, move_ratio=0.25, seed=9876, stiffness=10.0, reference=dict(vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]))
        # Add a shape move.
        elastic.elastic_shape_move(stepsize=0.1, move_ratio=0.5);
    """

    def __init__(   self,
                    stiffness,
                    reference,
                    stepsize,
                    param_ratio,
                    **params):
        hoomd.util.print_status_line();
        # initialize base class
        shape_update.__init__(self, **params); # mc, move_ratio, seed,
        if hoomd.context.exec_conf.isCUDAEnabled():
            hoomd.context.msg.warning("update.elastic_shape: GPU is not implemented defaulting to CPU implementation.\n");

        self.elastic_shape_move(stepsize, param_ratio);

        if isinstance(self.mc, integrate.convex_polyhedron):
            clss = _hpmc.ShapeSpringLogBoltzmannConvexPolyhedron;
        elif isinstance(self.mc, integrate.ellipsoid):
            clss = _hpmc.ShapeSpringLogBoltzmannEllipsoid
        else:
            hoomd.context.msg.error("update.elastic_shape: Unsupported integrator.\n");
            raise RuntimeError("Error initializing compute.elastic_shape");

        self.stiffness = hoomd.variant._setup_variant_input(stiffness);
        ref_shape = self.mc.shape_class.make_param(**reference);
        self.boltzmann_function = clss(self.stiffness.cpp_variant, ref_shape, self.move_cpp);
        self.cpp_updater.registerLogBoltzmannFunction(self.boltzmann_function);

    def set_stiffness(self, stiffness):
        R""" Update the stiffness set point for Metropolis Monte Carlo elastic shape updates.

        Args:
            stiffness (float) or (:py:mod:`hoomd.variant`): :math:`\frac{k}/{k_{\mathrm{B}}T}`.
        """
        self.stiffness = hoomd.variant._setup_variant_input(stiffness)
        self.boltzmann_function.setStiffness(self.stiffness.cpp_variant)

class clusters(_updater):
    R""" Equilibrate the system according to the geometric cluster algorithm (GCA).

    The GCA as described in Liu and Lujten (2004), http://doi.org/10.1103/PhysRevLett.92.035504 is used for hard shape,
    patch interactions and depletants.

    With depletants, Clusters are defined by a simple distance cut-off criterion. Two particles belong to the same cluster if
    the circumspheres of the depletant-excluded volumes overlap.

    Supported moves include pivot moves (point reflection), line reflections (pi rotation around an axis), and type swaps.
    Only the pivot move is rejection free. With anisotropic particles, the pivot move cannot be used because it would create a
    chiral mirror image of the particle, and only line reflections are employed. Line reflections are not rejection free because
    of periodic boundary conditions, as discussed in Sinkovits et al. (2012), http://doi.org/10.1063/1.3694271 .

    The type swap move works between two types of spherical particles and exchanges their identities.

    The :py:class:`clusters` updater support TBB execution on multiple CPU cores. See :doc:`installation` for more information on how
    to compile HOOMD with TBB support.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        seed (int): The seed of the pseudo-random number generator (Needs to be the same across partitions of the same Gibbs ensemble)
        period (int): Number of timesteps between histogram evaluations.

    Example::

        mc = hpmc.integrate.sphere(seed=415236)
        hpmc.update.clusters(mc=mc, seed=123)

    """
    def __init__(self, mc, seed, period=1):
        hoomd.util.print_status_line();

        if not isinstance(mc, integrate.mode_hpmc):
            hoomd.context.msg.warning("update.clusters: Must have a handle to an HPMC integrator.\n");
            return

        # initialize base class
        _updater.__init__(self);

        if not mc.implicit:
            if isinstance(mc, integrate.sphere):
               cls = _hpmc.UpdaterClustersSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.UpdaterClustersConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.UpdaterClustersSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.UpdaterClustersConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.UpdaterClustersSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.UpdaterClustersEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.UpdaterClustersSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.UpdaterClustersFacetedEllipsoid;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.UpdaterClustersSphereUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls =_hpmc.UpdaterClustersConvexPolyhedronUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls =_hpmc.UpdaterClustersFacetedEllipsoidUnion;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.UpdaterClustersPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.UpdaterClustersSphinx;
            else:
                raise RuntimeError("Unsupported integrator.\n");
        else:
            if isinstance(mc, integrate.sphere):
               cls = _hpmc.UpdaterClustersImplicitSphere;
            elif isinstance(mc, integrate.convex_polygon):
                cls = _hpmc.UpdaterClustersImplicitConvexPolygon;
            elif isinstance(mc, integrate.simple_polygon):
                cls = _hpmc.UpdaterClustersImplicitSimplePolygon;
            elif isinstance(mc, integrate.convex_polyhedron):
                cls = _hpmc.UpdaterClustersImplicitConvexPolyhedron;
            elif isinstance(mc, integrate.convex_spheropolyhedron):
                cls = _hpmc.UpdaterClustersImplicitSpheropolyhedron;
            elif isinstance(mc, integrate.ellipsoid):
                cls = _hpmc.UpdaterClustersImplicitEllipsoid;
            elif isinstance(mc, integrate.convex_spheropolygon):
                cls =_hpmc.UpdaterClustersImplicitSpheropolygon;
            elif isinstance(mc, integrate.faceted_sphere):
                cls =_hpmc.UpdaterClustersImplicitFacetedEllipsoid;
            elif isinstance(mc, integrate.sphere_union):
                cls =_hpmc.UpdaterClustersImplicitSphereUnion;
            elif isinstance(mc, integrate.convex_polyhedron_union):
                cls =_hpmc.UpdaterClustersImplicitConvexPolyhedronUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls =_hpmc.UpdaterClustersImplicitFacetedEllipsoidUnion;
            elif isinstance(mc, integrate.polyhedron):
                cls =_hpmc.UpdaterClustersImplicitPolyhedron;
            elif isinstance(mc, integrate.sphinx):
                cls =_hpmc.UpdaterClustersImplicitSphinx;
            else:
                raise RuntimeError("Unsupported integrator.\n");

        self.cpp_updater = cls(hoomd.context.current.system_definition, mc.cpp_integrator, int(seed))

        # register the clusters updater
        self.setupUpdater(period)

    def set_params(self, move_ratio=None, flip_probability=None, swap_move_ratio=None, delta_mu=None, swap_types=None):
        R""" Set options for the clusters moves.

        Args:
            move_ratio (float): Set the ratio between pivot and reflection moves (default 0.5)
            flip_probability (float): Set the probability for transforming an individual cluster (default 0.5)
            swap_move_ratio (float): Set the ratio between type swap moves and geometric moves (default 0.5)
            delta_mu (float): The chemical potential difference between types to be swapped
            swap_types (list): A pair of two types whose identities are swapped

        Note:
            When an argument is None, the value is left unchanged from its current state.

        Example::

            clusters = hpmc.update.clusters(mc, seed=123)
            clusters.set_params(move_ratio = 1.0)
            clusters.set_params(swap_types=['A','B'], delta_mu = -0.001)
        """

        hoomd.util.print_status_line();

        if move_ratio is not None:
            self.cpp_updater.setMoveRatio(float(move_ratio))

        if flip_probability is not None:
            self.cpp_updater.setFlipProbability(float(flip_probability))

        if swap_move_ratio is not None:
            self.cpp_updater.setSwapMoveRatio(float(swap_move_ratio))

        if delta_mu is not None:
            self.cpp_updater.setDeltaMu(float(delta_mu))

        if swap_types is not None:
            my_swap_types = tuple(swap_types)
            if len(my_swap_types) != 2:
                hoomd.context.msg.error("update.clusters: Need exactly two types for type swap.\n");
                raise RuntimeError("Error setting parameters in update.clusters");
            type_A = hoomd.context.current.system_definition.getParticleData().getTypeByName(my_swap_types[0]);
            type_B = hoomd.context.current.system_definition.getParticleData().getTypeByName(my_swap_types[1]);
            self.cpp_updater.setSwapTypePair(type_A, type_B)

    def get_pivot_acceptance(self):
        R""" Get the average acceptance ratio for pivot moves

        Returns:
            The average acceptance rate for pivot moves during the last run
        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getPivotAcceptance();

    def get_reflection_acceptance(self):
        R""" Get the average acceptance ratio for reflection moves

        Returns:
            The average acceptance rate for reflection moves during the last run
        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getReflectionAcceptance();

    def get_swap_acceptance(self):
        R""" Get the average acceptance ratio for swap moves

        Returns:
            The average acceptance rate for type swap moves during the last run
        """
        counters = self.cpp_updater.getCounters(1);
        return counters.getSwapAcceptance();
