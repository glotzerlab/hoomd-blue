# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" HPMC updaters.
"""

from . import _hpmc
from . import integrate
from . import compute
from hoomd import _hoomd

import math

from hoomd.logger import Loggable
from hoomd.update import _updater
from hoomd.meta import _Updater
from hoomd.parameterdicts import ParameterDict
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
        # initialize base class
        _updater.__init__(self);

        # Updater gets called at every timestep. Whether to perform a move is determined independently
        # according to frequency parameter.
        period = 1

        if not isinstance(mc, integrate._HPMCIntegrator):
            hoomd.context.current.device.cpp_msg.warning("update.boxmc: Must have a handle to an HPMC integrator.\n");
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
            hoomd.context.current.device.cpp_msg.error("update.wall: Unsupported integrator.\n");
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

        if not isinstance(mc, integrate._HPMCIntegrator):
            hoomd.context.current.device.cpp_msg.warning("update.muvt: Must have a handle to an HPMC integrator.\n");
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
        elif isinstance(mc, integrate.convex_spheropolyhedron_union):
            cls = _hpmc.UpdaterMuVTConvexPolyhedronUnion;
        elif isinstance(mc, integrate.faceted_ellipsoid_union):
            cls = _hpmc.UpdaterMuVTFacetedEllipsoidUnion;
        elif isinstance(mc, integrate.polyhedron):
            cls =_hpmc.UpdaterMuVTPolyhedron;
        else:
            hoomd.context.current.device.cpp_msg.error("update.muvt: Unsupported integrator.\n");
            raise RuntimeError("Error initializing update.muvt");

        self.cpp_updater = cls(hoomd.context.current.system_definition,
                               mc.cpp_integrator,
                               int(seed),
                               ngibbs);

        # register the muvt updater
        self.setupUpdater(period);

        # set the list of transferred types
        if not isinstance(transfer_types,list):
            hoomd.context.current.device.cpp_msg.error("update.muvt: Need list of types to transfer.\n");
            raise RuntimeError("Error initializing update.muvt");

        cpp_transfer_types = _hoomd.std_vector_uint();
        for t in transfer_types:
            if t not in type_list:
                hoomd.context.current.device.cpp_msg.error("Trying to transfer unknown type " + str(t) + "\n");
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
        self.check_initialization();

        if self.gibbs:
            raise RuntimeError("Gibbs ensemble does not support setting the fugacity.\n");

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if type not in type_list:
            hoomd.context.current.device.cpp_msg.error("Trying to set fugacity for unknown type " + str(type) + "\n");
            raise RuntimeError("Error setting muVT parameters");
        else:
            type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        fugacity_variant = hoomd.variant._setup_variant_input(fugacity);
        self.cpp_updater.setFugacity(type_id, fugacity_variant.cpp_variant);

    def set_params(self, dV=None, move_ratio=None, n_trial=None):
        R""" Set muVT parameters.

        Args:
            dV (float): (if set) Set volume rescaling factor (dimensionless)
            move_ratio (float): (if set) Set the ratio between volume and exchange/transfer moves (applies to Gibbs ensemble)
            n_trial (int): (if set) Number of re-insertion attempts per depletant

        Example::

            muvt = hpmc.update.muvt(mc, period = 10)
            muvt.set_params(dV=0.1)
            muvt.set_params(n_trial=2)
            muvt.set_params(move_ratio=0.05)

        """
        self.check_initialization();

        if move_ratio is not None:
            if not self.gibbs:
                hoomd.context.current.device.cpp_msg.warning("Move ratio only used in Gibbs ensemble.\n");
            self.cpp_updater.setMoveRatio(float(move_ratio))

        if dV is not None:
            if not self.gibbs:
                hoomd.context.current.device.cpp_msg.warning("Parameter dV only available for Gibbs ensemble.\n");
            self.cpp_updater.setMaxVolumeRescale(float(dV))

        if n_trial is not None:
            self.cpp_updater.setNTrial(int(n_trial))


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
        #initialize base class
        _updater.__init__(self);
        cls = None;
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
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
            elif isinstance(mc, integrate.convex_spheropolyhedron_union):
                cls = _hpmc.RemoveDriftUpdaterConvexPolyhedronUnion;
            elif isinstance(mc, integrate.faceted_ellipsoid_union):
                cls = _hpmc.RemoveDriftUpdaterFacetedEllipsoidUnion;
            else:
                hoomd.context.current.device.cpp_msg.error("update.remove_drift: Unsupported integrator.\n");
                raise RuntimeError("Error initializing update.remove_drift");
        else:
            raise RuntimeError("update.remove_drift: Error! GPU not implemented.");

        self.cpp_updater = cls(hoomd.context.current.system_definition, external_lattice.cpp_compute, mc.cpp_integrator);
        self.setupUpdater(period);


class Clusters(_Updater):
    R""" Equilibrate the system according to the geometric cluster algorithm (GCA).

    The GCA as described in Liu and Lujten (2004),
    http://doi.org/10.1103/PhysRevLett.92.035504 is used for hard shape, patch
    interactions and depletants.

    With depletants, Clusters are defined by a simple distance cut-off
    criterion. Two particles belong to the same cluster if the circumspheres of
    the depletant-excluded volumes overlap.

    Supported moves include pivot moves (point reflection), line reflections
    (pi rotation around an axis), and type swaps.  Only the pivot move is
    rejection free. With anisotropic particles, the pivot move cannot be used
    because it would create a chiral mirror image of the particle, and only
    line reflections are employed. Line reflections are not rejection free
    because of periodic boundary conditions, as discussed in Sinkovits et al.
    (2012), http://doi.org/10.1063/1.3694271 .

    The type swap move works between two types of spherical particles and
    exchanges their identities.

    The :py:class:`Clusters` updater support TBB execution on multiple CPU
    cores. See :doc:`installation` for more information on how to compile HOOMD
    with TBB support.

    Args:
        seed (int): Random number seed.
        swap_types(list): A pair of two types whose identities may be swapped.
        move_ratio(float): Set the ratio between pivot and reflection moves.
        flip_probability(float): Set the probability for transforming an
                                 individual cluster.
        swap_move_ratio(float): Set the ratio between type swap moves and
                                geometric moves.
        delta_mu(float): The chemical potential difference between types to
                         be swapped.
        trigger (int): Number of timesteps between histogram evaluations.

    Examples::

        TODO: link to example notebooks

    """

    def __init__(self, seed, swap_types, move_ratio=0.5,
                 flip_probability=0.5, swap_move_ratio=0.5, trigger=1):
        super().__init__(trigger)
        try:
            if len(swap_types) != 2 and len(swap_types) != 0:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("swap_types must be an iterable of length "
                             "2 or 0.")

        self._param_dict = ParameterDict(seed=int, swap_types=list,
                                         move_ratio=float,
                                         flip_probability=float,
                                         swap_move_ratio=float)
        self._param_dict.update(dict(seed=seed, swap_types=swap_types,
                                     move_ratio=move_ratio,
                                     flip_probability=flip_probability,
                                     swap_move_ratio=swap_move_ratio))

    def attach(self, simulation):
        integrator = simulation.operations.integrator
        if not isinstance(integrator, integrate._HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        integrator_pairs = [
                (integrate.Sphere,
                    _hpmc.UpdaterClustersSphere),
                (integrate.convex_polygon,
                    _hpmc.UpdaterClustersConvexPolygon),
                (integrate.simple_polygon,
                    _hpmc.UpdaterClustersConvexPolygon),
                (integrate.convex_polyhedron,
                    _hpmc.UpdaterClustersConvexPolyhedron),
                (integrate.convex_spheropolyhedron,
                    _hpmc.UpdaterClustersSpheropolyhedron),
                (integrate.ellipsoid,
                    _hpmc.UpdaterClustersEllipsoid),
                (integrate.convex_spheropolygon,
                    _hpmc.UpdaterClustersSpheropolygon),
                (integrate.faceted_sphere,
                    _hpmc.UpdaterClustersFacetedEllipsoid),
                (integrate.sphere_union,
                    _hpmc.UpdaterClustersSphereUnion),
                (integrate.convex_spheropolyhedron_union,
                    _hpmc.UpdaterClustersConvexPolyhedronUnion),
                (integrate.faceted_ellipsoid_union,
                    _hpmc.UpdaterClustersFacetedEllipsoidUnion),
                (integrate.polyhedron,
                    _hpmc.UpdaterClustersPolyhedron),
                (integrate.sphinx,
                    _hpmc.UpdaterClustersSphinx)
                ]

        cpp_cls = None
        for python_integrator, cpp_updater in integrator_pairs:
            if isinstance(integrator, python_integrator):
                cpp_cls = cpp_updater
        if cpp_cls is None:
            raise RuntimeError("Unsupported integrator.\n")

        if not integrator.is_attached:
            raise RuntimeError("Integrator is not attached yet.")
        self._cpp_obj = cpp_cls(simulation.state._cpp_sys_def,
                                integrator._cpp_obj,
                                int(self.seed))
        super().attach(simulation)

    @property
    def counter(self):
        R""" Get the number of accepted and rejected cluster moves.

        Returns:
            A counter object with pivot, reflection, and swap properties. Each
            property is a list of accepted moves and rejected moves since the
            last run.

        Note::
            if the updater is not attached None will be returned.
        """
        if not self.is_attached:
            return None
        else:
            return self._cpp_obj.getCounters(1)

    @Loggable.log(flag='multi')
    def pivot_moves(self):
        R""" Get a tuple with the accepted and rejected pivot moves.

        Returns:
            A tuple of (accepted moves, rejected moves) since the last run.
            Returns (0, 0) if not attached.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            return counter.pivot

    @Loggable.log(flag='multi')
    def reflection_moves(self):
        R""" Get a tuple with the accepted and rejected reflection moves.

        Returns:
            A tuple of (accepted moves, rejected moves) since the last run.
            Returns (0, 0) if not attached.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            return counter.reflection

    @Loggable.log(flag='multi')
    def swap_moves(self):
        R""" Get a tuple with the accepted and rejected swap moves.

        Returns:
            A tuple of (accepted moves, rejected moves) since the last run.
            Returns (0, 0) if not attached.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            return counter.swap
