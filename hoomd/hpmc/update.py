# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" HPMC updaters.
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

    Pressure inputs to update.boxmc are defined as :math:`\beta P`. Conversions from a specific definition of reduced
    pressure :math:`P^*` are left for the user to perform.

    Note:
        All *delta* and *weight* values for all move types default to 0.

    Example::

        mc = hpmc.integrate.sphere(seed=415236, d=0.3)
        boxMC = hpmc.update.boxmc(mc, betaP=1.0, seed=9876)
        boxMC.set_betap(2.0)
        boxMC.volume(delta=0.01, weight=2.0)
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
        walls (:py:class:`hoomd.hpmc.compute.wall`): the wall class instance to be updated
        py_updater (callable): the python callback that performs the update moves. This must be a python method that is a function of the timestep of the simulation.
               It must actually update the :py:class:`hoomd.hpmc.compute.wall`) managed object.
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
            cls = integrate._get_sized_entry('UpdaterExternalFieldWallConvexPolyhedron', mc.max_verts);
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
        transfer_types (list): List of type names that are being transfered from/to the reservoir or between boxes (if *None*, all types)
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

    def set_fugacity(self, type, fugacity):
        R""" Change muVT fugacities.

        Args:
            type (str): Particle type to set parameters for
            fugacity (float): Fugacity of this particle type (dimension of volume^-1)

        Example:

            muvt = hpmc.update.muvt(mc, period = 10)
            muvt.set_fugacity(type='A',fugacity=1.23)
            variant = hoomd.variant.linear_interp(points= [(0,1e1), (1e5, 4.56)])
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
        external_lattice (:py:class:`hoomd.hpmc.compute.lattice_field`): lattice field where the lattice is defined.
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
        #initiliaze base class
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

class shape_update(_updater):
    R""" Apply shape updates to the shape definitions defined in the integrator.
         (mainly for internal use)

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
        setup_callback (function): will override the default pos callback. will be called everytime the pos file is written
        nselect (int): number of types to change every time the updater is called.

    Example::
        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.5)
        shape_up = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)

    """
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
                    nselect=1):
        util.print_status_line();
        _updater.__init__(self);

        cls = None;
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.UpdaterShapeSphere;
        elif isinstance(mc, integrate.convex_polygon):
            cls = _hpmc.UpdaterShapeConvexPolygon;
        elif isinstance(mc, integrate.simple_polygon):
            cls = _hpmc.UpdaterShapeSimplePolygon;
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = integrate._get_sized_entry('UpdaterShapeConvexPolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = integrate._get_sized_entry('UpdaterShapeSpheroPolyhedron', mc.max_verts);
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
            globals.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        self.cpp_updater = cls(globals.system_definition,
                                mc.cpp_integrator,
                                move_ratio,
                                seed,
                                nselect,
                                pretend);
        self.move_cpp = None;
        self.boltzmann_function = None;
        self.seed = seed;
        self.mc = mc;
        self.pos = pos;

        ntypes = globals.system_definition.getParticleData().getNTypes();
        self.tunables = ["stepsize-{}".format(i) for i in range(ntypes)];
        self.tunable_map = {};
        for i in range(ntypes):
            self.tunable_map.update({'stepsize-{}'.format(i) : (
                                    lambda obj: getattr(obj, 'get_step_size')(i),
                                    lambda obj: getattr(obj, 'get_move_acceptance')(i),
                                    1.0
                                    )})
        if pos and setup_pos:
            if pos_callback is None:
                pos.set_info(self.pos_callback);
            else:
                pos.set_info(pos_callback);

        self.setupUpdater(period, phase);

    def python_shape_move(self, callback, params, stepsize, param_ratio):
        util.print_status_line();
        if(self.move_cpp):
            globals.msg.error("update.shape_update.python_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            move_cls = _hpmc.PythonShapeMoveSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            move_cls = _hpmc.PythonShapeMoveConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            move_cls = _hpmc.PythonShapeMoveSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = integrate._get_sized_entry('PythonShapeMoveConvexPolyhedron', self.mc.max_verts);
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            move_cls = integrate._get_sized_entry('PythonShapeMoveSpheropolyhedron', self.mc.max_verts);
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
            globals.msg.error("update.shape_update.python_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            globals.msg.error("update.shape_update.python_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = globals.system_definition.getParticleData().getNTypes();
        param_list = self.mc.shape_class.ensure_list(params);
        self.move_cpp = move_cls(ntypes, callback, param_list, float(stepsize), float(param_ratio));
        self.cpp_updater.registerShapeMove(self.move_cpp);

    def vertex_shape_move(self, stepsize=0.01, mixratio=0.25, volume=1.0):
        util.print_status_line();
        if(self.move_cpp):
            globals.msg.error("update.shape_update.vertex_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            pass;
        elif isinstance(self.mc, integrate.convex_polygon):
            pass;
        elif isinstance(self.mc, integrate.simple_polygon):
            pass;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = integrate._get_sized_entry('GeneralizedShapeMoveConvexPolyhedron', self.mc.max_verts);
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
            globals.msg.error("update.shape_update.vertex_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            globals.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = globals.system_definition.getParticleData().getNTypes();
        self.cpp_updater.registerShapeMove(move_cls(ntypes, stepsize, mixratio, volume));

    def constant_shape_move(self, shape_params):
        util.print_status_line();
        if(self.move_cpp):
            globals.msg.error("update.shape_update.constant_shape_move: Cannot change the move once initialized.\n");
            raise RuntimeError("Error initializing update.shape_update");
        move_cls = None;
        if isinstance(self.mc, integrate.sphere):
            move_cls = _hpmc.ConstantShapeMoveSphere;
        elif isinstance(self.mc, integrate.convex_polygon):
            move_cls = _hpmc.ConstantShapeMoveConvexPolygon;
        elif isinstance(self.mc, integrate.simple_polygon):
            move_cls = _hpmc.ConstantShapeMoveSimplePolygon;
        elif isinstance(self.mc, integrate.convex_polyhedron):
            move_cls = integrate._get_sized_entry('ConstantShapeMoveConvexPolyhedron', self.mc.max_verts);
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            move_cls = integrate._get_sized_entry('ConstantShapeMoveSpheropolyhedron', self.mc.max_verts);
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
            globals.msg.error("update.shape_update.constant_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            globals.msg.error("update.shape_update.constant_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = globals.system_definition.getParticleData().getNTypes();
        self.cpp_updater.registerShapeMove(move_cls(ntypes, self.mc.shape_class.make_param(**shape_params)));

    def scale_shear_shape_move(self, scale_max, shear_max, move_ratio=0.5):
        util.print_status_line();
        if(self.move_cpp):
            globals.msg.error("update.shape_update.scale_shear_shape_move: Cannot change the move once initialized.\n");
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
            move_cls = integrate._get_sized_entry('ScaleShearShapeMoveConvexPolyhedron', self.mc.max_verts);
        elif isinstance(self.mc, integrate.convex_spheropolyhedron):
            pass;
            # move_cls = integrate._get_sized_entry('ScaleShearShapeMoveSpheropolyhedron', self.mc.max_verts);
        elif isinstance(self.mc, integrate.ellipsoid):
            # move_cls = _hpmc.ScaleShearShapeMoveEllipsoid;
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
            globals.msg.error("update.shape_update.scale_shear_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        if not move_cls:
            globals.msg.error("update.shape_update.scale_shear_shape_move: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        ntypes = globals.system_definition.getParticleData().getNTypes();
        self.cpp_updater.registerShapeMove(move_cls(ntypes, callback, params, stepsize, param_ratio));


    def get_total_count(self, idx=0):
        R""" Get the total number of moves attempted by the updater
        Args:
            idx (int): the typeid of the particle type
        Returns:
            The total number of moves attempted by the updater

        Example::
            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            total = shape_updater.get_total_count(0)

        """
        util.print_status_line();
        return self.cpp_updater.getTotalCount(idx);

    def get_accepted_count(self, idx=0):
        R""" Get the total number of moves accepted by the updater
        Args:
            idx (int): the typeid of the particle type
        Returns:
            The total number of moves accepted by the updater

        Example::
            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            accepted = shape_updater.get_accepted_count(0)
        """
        util.print_status_line();
        return self.cpp_updater.getAcceptedCount(idx);

    def get_move_acceptance(self, idx=0):
        R""" Get the acceptance ratio for a particle type
        Args:
            idx (int): the typeid of the particle type
        Returns:
            The acceptance ratio for a particle type

        Example::
            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            ratio = shape_updater.get_move_acceptance(0)

        """
        util.print_status_line();
        if self.get_total_count(idx):
            return float(self.get_accepted_count(idx))/float(self.get_total_count(idx));
        return 0.0;

    def get_step_size(self, idx=0):
        R""" Get the shape move stepsize for a particle type
        Args:
            idx (int): the typeid of the particle type
        Returns:
            The shape move stepsize for a particle type

        Example::
            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            stepsize = shape_updater.get_step_size(0)

        """
        util.print_status_line();
        return self.cpp_updater.getStepSize(idx);

    def reset_statistics(self):
        R""" Reset the acceptance statistics for the updater
        Example::
            mc = hpmc.integrate.shape(..);
            mc.shape_param[name].set(....);
            shape_updater = hpmc.update.shape_update(mc, move_ratio=0.25, seed=9876)
            run(100)
            shape_updater.reset_statistics()

        """
        util.print_status_line();
        self.cpp_updater.resetStatistics();

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
        util.print_status_line();
        if isinstance(types, str):
            types = [types];
        for names in types:
            typ = globals.system_definition.getParticleData().getTypeByName(name);
            if not stepsize is None:
                self.cpp_updater.setStepSize(typ, step_size);


class alchemy(shape_update):
    R""" Apply shape updates to the shape definitions defined in the integrator.
         (mainly for internal use)

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): HPMC integrator object for system on which to apply box updates
        move_ratio (:py:class:`float` or :py:mod:`hoomd.variant`): fraction of steps to run the updater.
        seed (int): random number seed for shape move generators
        stepsize (float): initial stepsize for all particle types.
        param_ratio (float): average fraction of shape parameters to change each update.
        volume (float): volume of the particles to hold constant (depricated)
        move (callable object): If supplied the updater will use this callback to update the shape definitions see below for more details.
                                If None then the default shape move will be applied to each shape type.
        params (dict): any of the other keyword arguments from :py:mod:`hoomd.hpmc.update.shape_update`

    Additional comments here. what enseble are we simulating etc.

    Note on move function:
        explain how to write the function here.

    Example::
        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.5)
        alchem = hpmc.update.alchemy(mc, move_ratio=0.25, seed=9876)

    """
    def __init__(   self,
                    # mc,
                    # move_ratio,
                    # seed,
                    # stepsize=0.01,
                    # param_ratio=0.25,
                    # volume=1.0,
                    # move = None,
                    **params):
        util.print_status_line();
        # initialize base class
        shape_update.__init__(self, **params);
        boltzmann_cls = None;
        if isinstance(mc, integrate.sphere):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannSphere;
        elif isinstance(mc, integrate.convex_polygon):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannConvexPolygon;
        elif isinstance(mc, integrate.simple_polygon):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannSimplePolygon;
        elif isinstance(mc, integrate.convex_polyhedron):
            boltzmann_cls = integrate._get_sized_entry('AlchemyLogBotzmannConvexPolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            boltzmann_cls = integrate._get_sized_entry('AlchemyLogBoltzmannSpheroPolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.ellipsoid):
            boltzmann_cls = _hpmc.AlchemyLogBoltzmannEllipsoid;
        elif isinstance(mc, integrate.convex_spheropolygon):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSpheropolygon;
        elif isinstance(mc, integrate.patchy_sphere):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannPatchySphere;
        elif isinstance(mc, integrate.polyhedron):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannPolyhedron;
        elif isinstance(mc, integrate.sphinx):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSphinx;
        elif isinstance(mc, integrate.sphere_union):
            boltzmann_cls =_hpmc.AlchemyLogBoltzmannSphereUnion;
        else:
            globals.msg.error("update.shape_update: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.shape_update");

        self.boltzmann_function = boltzmann_cls();
        self.cpp_updater.registerLogBoltzmannFunction(self.boltzmann_function);

class elastic_shape(shape_update):
    R""" Apply shape updates to the shape definitions defined in the integrator.
         (mainly for internal use)

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): HPMC integrator object for system on which to apply box updates
        move_ratio (:py:class:`float` or :py:mod:`hoomd.variant`): fraction of steps to run the updater.
        seed (int): random number seed for shape move generators
        stepsize (float): initial stepsize for all particle types.
        ref_shape (:py:mod:`hoomd.hpmc.data.shape_param`): describe
        stiffness (float): stiffness of the particle spring
        params (dict): any of the other keyword arguments from :py:mod:`hoomd.hpmc.update.shape_update`

    Additional comments here. what enseble are we simulating etc.

    Note on move functions and acceptance criterion:
        explain how to write the function here.

    Example::
        mc = hpmc.integrate.convex_polyhedron(seed=415236, d=0.3, a=0.5)
        elastic = hpmc.update.elastic_shape(mc, move_ratio=0.25, seed=9876, stiffness=10.0, reference=dict(vertices=[(1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)]))
        # Add a shape move.
        elastic.scale_shear_move(scale_max=0.1, shear_max=0.1, move_ratio=0.5);
    """
    def __init__(   self,
                    # mc,
                    # move_ratio,
                    # seed,
                    stiffness,
                    reference,
                    # stepsize=0.1,
                    **params):
        util.print_status_line();
        # initialize base class
        shape_update.__init__(self, **params); # mc, move_ratio, seed,
        if globals.exec_conf.isCUDAEnabled():
            globals.msg.warning("update.elastic_shape: GPU is not implemented defaulting to CPU implementation.\n");

        ref_shape = self.mc.shape_class.make_param(**reference);
        if isinstance(mc, integrate.convex_polyhedron):
            clss = integrate._get_sized_entry('ShapeSpringLogBoltzmannConvexPolyhedron', mc.max_verts);
        elif isinstance(mc, integrate.ellipsoid):
            clss = _hpmc.ShapeSpringLogBoltzmannEllipsoid(stiffness, ref_shape);
        else:
            globals.msg.error("update.elastic_shape: Unsupported integrator.\n");
            raise RuntimeError("Error initializing compute.elastic_shape");
        self.boltzmann_function = clss(stiffness, ref_shape);
        self.cpp_updater.registerLogBoltzmannFunction(self.boltzmann_function);
