# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""HPMC updaters."""

from . import _hpmc
from . import integrate
from hoomd import _hoomd
from hoomd.logging import log
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
import hoomd.data.typeconverter
from hoomd.operation import Updater
import hoomd


class BoxMC(Updater):
    r"""Apply box updates to sample isobaric and related ensembles.

    Args:
        betaP (`float` or :py:mod:`hoomd.variant.Variant`):
            :math:`\frac{p}{k_{\mathrm{B}}T}` :math:`[\mathrm{length}^{-2}]`
            in 2D or :math:`[\mathrm{length}^{-3}]` in 3D.
        trigger (hoomd.trigger.Trigger): Select the timesteps to perform box
            trial moves.

    Use `BoxMC` in conjunction with an HPMC integrator to allow the simulation
    box to undergo random fluctuations at constant pressure. `BoxMC` supports
    both isotropic (all box sides changed equally) and anisotropic volume change
    moves as well as shearing of the simulation box. Multiple types of box moves
    can be applied simultaneously during a simulation. For this purpose, each
    type of box move has an associated weight that determines the relative
    frequency of a box move happening relative to the others. By default, no
    moves are applied (*weight* values for all move types default to 0). After
    a box trial move is proposed, all the particle positions are scaled into the
    new box. Trial moves are then accepted, if they do not produce an overlap,
    according to standard Metropolis criterion and rejected otherwise.

    .. rubric:: Mixed precision

    `BoxMC` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    Attributes:
        volume (dict):
            Parameters for isobaric volume moves that scale the box lengths
            uniformly. The dictionary has the following keys:

            * ``weight`` (float) - Relative weight of volume box moves.
            * ``mode`` (str) - ``standard`` proposes changes to the box volume
              and ``ln`` proposes changes to the logarithm of the volume.
              Initially starts off in 'standard' mode.
            * ``delta`` (float) - Maximum change in **V** or **ln(V)** where V
              is box area (2D) or volume (3D).

        aspect (dict):
            Parameters for isovolume aspect ratio moves. The dictionary has the
            following keys:

            * ``weight`` (float) - Relative weight of aspect box moves.
            * ``delta`` (float) - Maximum relative change of box aspect ratio.

        length (dict):
            Parameters for isobaric box length moves that change box lengths
            independently. The dictionary has the following keys:

            * ``weight`` (float) - Maximum change of HOOMD-blue box parameters
              Lx, Ly, and Lz.
            * ``delta`` (tuple[float, float, float]) - Maximum change of the
              box lengths ``(Lx, Ly, Lz)`` :math:`[\mathrm{length}]`.

        shear (dict):
            Parameters for isovolume box shear moves. The dictionary has the
            following keys:

            * ``weight`` (float) - Relative weight of shear box moves.
            * ``delta`` (tuple[float, float, float]) -  maximum change of the
              box tilt factor ``(xy, xz, yz)``.
            * ``reduce`` (float) - Maximum number of lattice vectors of shear
              to allow before applying lattice reduction. Values less than 0.5
              disable shear reduction.

        instance (int):
            When using multiple `BoxMC` updaters in a single simulation,
            give each a unique value for `instance` so they generate
            different streams of random numbers.
    """

    def __init__(self, betaP, trigger=1):
        super().__init__(trigger)

        _default_dict = dict(weight=0.0, delta=0.0)
        param_dict = ParameterDict(volume={
            "mode": hoomd.data.typeconverter.OnlyFrom(['standard', 'ln']),
            **_default_dict
        },
                                   aspect=_default_dict,
                                   length=dict(weight=0.0, delta=(0.0,) * 3),
                                   shear=dict(weight=0.0,
                                              delta=(0.0,) * 3,
                                              reduce=0.0),
                                   betaP=hoomd.variant.Variant,
                                   instance=int,
                                   _defaults={'volume': {
                                       'mode': 'standard'
                                   }})
        self._param_dict.update(param_dict)
        self.betaP = betaP
        self.instance = 0

    def _add(self, simulation):
        """Add the operation to a simulation.

        HPMC uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        self._cpp_obj = _hpmc.UpdaterBoxMC(self._simulation.state._cpp_sys_def,
                                           integrator._cpp_obj, self.betaP)
        super()._attach()

    @property
    def counter(self):
        """Trial move counters.

        The counter object has the following attributes:

        * ``volume``: `tuple` [`int`, `int`] - Number of accepted and rejected
          volume and length moves.
        * ``shear``: `tuple` [`int`, `int`] - Number of accepted and rejected
          shear moves.
        * ``aspect``: `tuple` [`int`, `int`] - Number of accepted and rejected
          aspect moves.

        Note:
            The counts are reset to 0 at the start of each call to
            `hoomd.Simulation.run`. Before the first call to `Simulation.run`,
            `counter` is `None`.
        """
        if not self._attached:
            return None
        else:
            return self._cpp_obj.getCounters(1)

    @log(category="sequence")
    def volume_moves(self):
        """tuple[int, int]: The accepted and rejected volume and length moves.

        (0, 0) before the first call to `Simulation.run`.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            if self.volume["mode"] == "standard":
                attr = "volume"
            else:
                attr = "ln_volume"
            return getattr(counter, attr)

    @log(category="sequence")
    def shear_moves(self):
        """tuple[int, int]: The accepted and rejected shear moves.

        (0, 0) before the first call to `Simulation.run`.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            return counter.shear

    @log(category="sequence")
    def aspect_moves(self):
        """tuple[int, int]: The accepted and rejected aspect moves.

        (0, 0) before the first call to `Simulation.run`.
        """
        counter = self.counter
        if counter is None:
            return (0, 0)
        else:
            return counter.aspect


class wall:  # noqa - will be rewritten for v3
    """Apply wall updates with a user-provided python callback.

    Args:
        mc (:py:mod:`hoomd.hpmc.integrate`): MC integrator.
        walls (:py:class:`hoomd.hpmc.field.wall`): the wall class instance to be
          updated
        py_updater (`callable`): the python callback that performs the update
          moves. This must be a python method that is a function of the
          timestep of the simulation. It must actually update the
          :py:class:`hoomd.hpmc.field.wall`) managed object.
        move_probability (float): the probability with which an update move is
          attempted
        seed (int): the seed of the pseudo-random number generator that
          determines whether or not an update move is attempted
        period (int): the number of timesteps between update move attempt
          attempts Every *period* steps, a walls update move is tried with
          probability *move_probability*. This update move is provided by the
          *py_updater* callback. Then, update.wall only accepts an update move
          provided by the python callback if it maintains confinement conditions
          associated with all walls. Otherwise, it reverts back to a non-updated
          copy of the walls.

    Once initialized, the update provides the following log quantities that can
    be logged via ``hoomd.analyze.log``:

    * **hpmc_wall_acceptance_ratio** - the acceptance ratio for wall update
      moves

    Example::

        mc = hpmc.integrate.sphere(seed = 415236);
        ext_wall = hpmc.compute.wall(mc);
        ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0],
                                 inside = True);
        def perturb(timestep):
          r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0,
                                                     param = "rsq"));
          ext_wall.set_sphere_wall(index = 0, radius = 1.5*r,
                                   origin = [0, 0, 0], inside = True);
        wall_updater = hpmc.update.wall(mc, ext_wall, perturb,
                                        move_probability = 0.5, seed = 27,
                                        period = 50);
        log = analyze.log(quantities=['hpmc_wall_acceptance_ratio'],
                          period=100, filename='log.dat', overwrite=True);

    Example::

        mc = hpmc.integrate.sphere(seed = 415236);
        ext_wall = hpmc.compute.wall(mc);
        ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0],
                                 inside = True);
        def perturb(timestep):
          r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0,
                                                     param = "rsq"));
          ext_wall.set_sphere_wall(index = 0, radius = 1.5*r,
                                   origin = [0, 0, 0], inside = True);
        wall_updater = hpmc.update.wall(mc, ext_wall, perturb,
                                        move_probability = 0.5,
                                        seed = 27, period = 50);

    """

    def __init__(self, mc, walls, py_updater, move_probability, seed, period=1):

        # initialize base class
        # _updater.__init__(self)

        cls = None
        if isinstance(mc, integrate.sphere):
            cls = _hpmc.UpdaterExternalFieldWallSphere
        elif isinstance(mc, integrate.convex_polyhedron):
            cls = _hpmc.UpdaterExternalFieldWallConvexPolyhedron
        elif isinstance(mc, integrate.convex_spheropolyhedron):
            cls = _hpmc.UpdaterExternalFieldWallSpheropolyhedron
        else:
            hoomd.context.current.device.cpp_msg.error(
                "update.wall: Unsupported integrator.\n")
            raise RuntimeError("Error initializing update.wall")

        self.cpp_updater = cls(hoomd.context.current.system_definition,
                               mc.cpp_integrator, walls.cpp_compute, py_updater,
                               move_probability, seed)
        self.setupUpdater(period)

    def get_accepted_count(self, mode=0):
        r"""Get the number of accepted wall update moves.

        Args:
            mode (int): specify the type of count to return. If mode!=0,
              return absolute quantities. If mode=0, return quantities relative
              to the start of the run. DEFAULTS to 0.

        Returns:
           the number of accepted wall update moves

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0],
                                     inside = True);
            def perturb(timestep):
              r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0,
                                                         param = "rsq"));
              ext_wall.set_sphere_wall(index = 0, radius = 1.5*r,
                                       origin = [0, 0, 0], inside = True);
            wall_updater = hpmc.update.wall(mc, ext_wall, perturb,
                                            move_probability = 0.5, seed = 27,
                                            period = 50);
            run(100);
            acc_count = wall_updater.get_accepted_count(mode = 0);
        """
        return self.cpp_updater.getAcceptedCount(mode)

    def get_total_count(self, mode=0):
        r"""Get the number of attempted wall update moves.

        Args:
            mode (int): specify the type of count to return. If mode!=0,
              return absolute quantities. If mode=0, return quantities relative
              to the start of the run. DEFAULTS to 0.

        Returns:
           the number of attempted wall update moves

        Example::

            mc = hpmc.integrate.sphere(seed = 415236);
            ext_wall = hpmc.compute.wall(mc);
            ext_wall.add_sphere_wall(radius = 1.0, origin = [0, 0, 0],
                                     inside = True);
            def perturb(timestep):
              r = np.sqrt(ext_wall.get_sphere_wall_param(index = 0,
                                                         param = "rsq"));
              ext_wall.set_sphere_wall(index = 0, radius = 1.5*r,
                                       origin = [0, 0, 0], inside = True);
            wall_updater = hpmc.update.wall(mc, ext_wall, perturb,
                                            move_probability = 0.5, seed = 27,
                                            period = 50);
            run(100);
            tot_count = wall_updater.get_total_count(mode = 0);
        """
        return self.cpp_updater.getTotalCount(mode)


class MuVT(Updater):
    r"""Insert and remove particles in the muVT ensemble.

    Args:
        trigger (int): Number of timesteps between grand canonical insertions
        transfer_types (list): List of type names that are being transferred
          from/to the reservoir or between boxes
        ngibbs (int): The number of partitions to use in Gibbs ensemble
          simulations (if == 1, perform grand canonical muVT)
        max_volume_rescale (float): maximum step size in ln(V) (applies to Gibbs
          ensemble)
        move_ratio (float): (if set) Set the ratio between volume and
          exchange/transfer moves (applies to Gibbs ensemble)

    The muVT (or grand-canonical) ensemble simulates a system at constant
    fugacity.

    Gibbs ensemble simulations are also supported, where particles and volumeare
    swapped between two or more boxes.  Every box correspond to one MPI
    partition, and can therefore run on multiple ranks. Use the
    ``ranks_per_partition`` argument of `hoomd.communicator.Communicator` to
    enable partitioned simulations.

    Note:
        Multiple Gibbs ensembles are also supported in a single parallel job,
        with the ngibbs option to update.muvt(), where the number of partitions
        can be a multiple of ngibbs.

    Attributes:
        trigger (int): Select the timesteps on which to perform cluster moves.
        transfer_types (list): List of type names that are being transferred
          from/to the reservoir or between boxes
        max_volume_rescale (float): Maximum step size in ln(V) (applies to
          Gibbs ensemble)
        move_ratio (float): The ratio between volume and exchange/transfer moves
          (applies to Gibbs ensemble)
        ntrial (float): (**default**: 1) Number of configurational bias attempts
          to swap depletants
    """

    def __init__(self,
                 transfer_types,
                 ngibbs=1,
                 max_volume_rescale=0.1,
                 volume_move_probability=0.5,
                 trigger=1):
        super().__init__(trigger)

        self.ngibbs = int(ngibbs)

        _default_dict = dict(ntrial=1)
        param_dict = ParameterDict(
            transfer_types=list(transfer_types),
            max_volume_rescale=float(max_volume_rescale),
            volume_move_probability=float(volume_move_probability),
            **_default_dict)
        self._param_dict.update(param_dict)

        typeparam_fugacity = TypeParameter(
            'fugacity',
            type_kind='particle_types',
            param_dict=TypeParameterDict(hoomd.variant.Variant,
                                         len_keys=1,
                                         _defaults=hoomd.variant.Constant(0.0)))
        self._extend_typeparam([typeparam_fugacity])

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        cpp_cls_name = "UpdaterMuVT"
        cpp_cls_name += integrator.__class__.__name__
        cpp_cls = getattr(_hpmc, cpp_cls_name)

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                integrator._cpp_obj, self.ngibbs)
        super()._attach()

    @log(category='sequence', requires_run=True)
    def insert_moves(self):
        """tuple[int, int]: Count of the accepted and rejected paricle \
        insertion moves.

        None when not attached
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.insert

    @log(category='sequence', requires_run=True)
    def remove_moves(self):
        """tuple[int, int]: Count of the accepted and rejected paricle removal \
        moves.

        None when not attached
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.remove

    @log(category='sequence', requires_run=True)
    def exchange_moves(self):
        """tuple[int, int]: Count of the accepted and rejected paricle \
        exchange moves.

        None when not attached
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.exchange

    @log(category='sequence', requires_run=True)
    def volume_moves(self):
        """tuple[int, int]: Count of the accepted and rejected paricle volume \
        moves.

        None when not attached
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.volume

    @log(category='object')
    def N(self):  # noqa: N802 - allow N as a function name
        """dict: Map of number of particles per type.

        None when not attached
        """
        N_dict = None
        if self._attached:
            N_dict = self._cpp_obj.N

        return N_dict


class Clusters(Updater):
    """Apply geometric cluster algorithm (GCA) moves.

    Args:
        pivot_move_probability (float): Set the probability for attempting a
                                        pivot move.
        flip_probability (float): Set the probability for transforming an
                                 individual cluster.
        trigger (Trigger): Select the timesteps on which to perform cluster
            moves.

    The GCA as described in Liu and Lujten (2004),
    http://doi.org/10.1103/PhysRevLett.92.035504 is used for hard shape, patch
    interactions and depletants.

    Implicit depletants are supported and simulated on-the-fly, as if they were
    present in the actual system.

    Supported moves include pivot moves (point reflection) and line reflections
    (pi rotation around an axis).  With anisotropic particles, the pivot move
    cannot be used because it would create a chiral mirror image of the
    particle, and only line reflections are employed. In general, line
    reflections are not rejection free because of periodic boundary conditions,
    as discussed in Sinkovits et al. (2012), http://doi.org/10.1063/1.3694271 .
    However, we restrict the line reflections to axes parallel to the box axis,
    which makes those moves rejection-free for anisotropic particles, but the
    algorithm is then no longer ergodic for those and needs to be combined with
    local moves.


    .. rubric:: Threading

    The `Clusters` updater support threaded execution on multiple CPU cores.

    .. rubric:: Mixed precision

    `Clusters` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    Attributes:
        pivot_move_probability (float): Set the probability for attempting a
                                        pivot move.
        flip_probability (float): Set the probability for transforming an
                                 individual cluster.
        trigger (Trigger): Select the timesteps on which to perform cluster
            moves.
    """
    _remove_for_pickling = Updater._remove_for_pickling + ('_cpp_cell',)
    _skip_for_equality = Updater._skip_for_equality | {'_cpp_cell'}

    def __init__(self,
                 pivot_move_probability=0.5,
                 flip_probability=0.5,
                 trigger=1):
        super().__init__(trigger)

        param_dict = ParameterDict(
            pivot_move_probability=float(pivot_move_probability),
            flip_probability=float(flip_probability))

        self._param_dict.update(param_dict)
        self.instance = 0

    def _add(self, simulation):
        """Add the operation to a simulation.

        HPMC uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        cpp_cls_name = "UpdaterClusters"
        cpp_cls_name += integrator.__class__.__name__
        cpp_cls = getattr(_hpmc, cpp_cls_name)
        use_gpu = (isinstance(self._simulation.device, hoomd.device.GPU)
                   and (cpp_cls_name + 'GPU') in _hpmc.__dict__)
        if use_gpu:
            cpp_cls_name += "GPU"
        cpp_cls = getattr(_hpmc, cpp_cls_name)

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        if use_gpu:
            sys_def = self._simulation.state._cpp_sys_def
            self._cpp_cell = _hoomd.CellListGPU(sys_def)
            self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                    integrator._cpp_obj, self._cpp_cell)
        else:
            self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                    integrator._cpp_obj)
        super()._attach()

    @log(requires_run=True)
    def avg_cluster_size(self):
        """float: the typical size of clusters.

        None when not attached.
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.average_cluster_size


def _box_getter(param_dict, attr):
    return param_dict._dict[attr]


def _box_setter(param_dict, attr, value):
    param_dict._dict[attr] = param_dict._type_converter[attr](value)
    setattr(param_dict._cpp_obj, attr, param_dict._dict[attr])


class QuickCompress(Updater):
    """Quickly compress a hard particle system to a target box.

    Args:
        trigger (Trigger): Update the box dimensions on triggered time steps.

        target_box (Box): Dimensions of the target box.

        max_overlaps_per_particle (float): The maximum number of overlaps to
            allow per particle (may be less than 1 - e.g.
            up to 250 overlaps would be allowed when in a system of 1000
            particles when max_overlaps_per_particle=0.25).

        min_scale (float): The minimum scale factor to apply to box dimensions.

    Use `QuickCompress` in conjunction with an HPMC integrator to scale the
    system to a target box size. `QuickCompress` can typically compress dilute
    systems to near random close packing densities in tens of thousands of time
    steps.

    It operates by making small changes toward the `target_box`: ``L_new = scale
    * L_current`` for each box parameter and then scaling the particle positions
    into the new box. If there are more than ``max_overlaps_per_particle *
    N_particles`` hard particle overlaps in the system, the box move is
    rejected. Otherwise, the small number of overlaps remain. `QuickCompress`
    then waits until local MC trial moves provided by the HPMC integrator
    remove all overlaps before it makes another box change.

    Note:
        The target box size may be larger or smaller than the current system
        box, and also may have different tilt factors. When the target box
        parameter is larger than the current, it scales by ``L_new = 1/scale *
        L_current``

    `QuickCompress` adjusts the value of ``scale`` based on the particle and
    translational trial move sizes to ensure that the trial moves will be able
    to remove the overlaps. It chooses a value of ``scale`` randomly between
    ``max(min_scale, 1.0 - min_move_size / max_diameter)`` and 1.0 where
    ``min_move_size`` is the smallest MC translational move size adjusted
    by the acceptance ratio and ``max_diameter`` is the circumsphere diameter
    of the largest particle type.

    Tip:
        Use the `hoomd.hpmc.tune.MoveSize` in conjunction with
        `QuickCompress` to adjust the move sizes to maintain a constant
        acceptance ratio as the density of the system increases.

    Warning:
        When the smallest MC translational move size is 0, `QuickCompress`
        will scale the box by 1.0 and not progress toward the target box.

    .. rubric:: Mixed precision

    `QuickCompress` uses reduced precision floating point arithmetic when
    checking for particle overlaps in the local particle reference frame.

    Attributes:
        trigger (Trigger): Update the box dimensions on triggered time steps.

        target_box (Box): Dimensions of the target box.

        max_overlaps_per_particle (float): The maximum number of overlaps to
            allow per particle (may be less than 1 - e.g.
            up to 250 overlaps would be allowed when in a system of 1000
            particles when max_overlaps_per_particle=0.25).

        min_scale (float): The minimum scale factor to apply to box dimensions.

        instance (int):
            When using multiple `QuickCompress` updaters in a single simulation,
            give each a unique value for `instance` so that they generate
            different streams of random numbers.
    """

    def __init__(self,
                 trigger,
                 target_box,
                 max_overlaps_per_particle=0.25,
                 min_scale=0.99):
        super().__init__(trigger)

        param_dict = ParameterDict(max_overlaps_per_particle=float,
                                   min_scale=float,
                                   target_box=hoomd.Box,
                                   instance=int)
        param_dict['max_overlaps_per_particle'] = max_overlaps_per_particle
        param_dict['min_scale'] = min_scale
        param_dict['target_box'] = target_box
        param_dict._set_special_getset("target_box", _box_getter, _box_setter)

        self._param_dict.update(param_dict)

        self.instance = 0

    def _add(self, simulation):
        """Add the operation to a simulation.

        HPMC uses RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        self._cpp_obj = _hpmc.UpdaterQuickCompress(
            self._simulation.state._cpp_sys_def, integrator._cpp_obj,
            self.max_overlaps_per_particle, self.min_scale, self.target_box)
        super()._attach()

    @property
    def complete(self):
        """True when the box has achieved the target."""
        if not self._attached:
            return False

        return self._cpp_obj.isComplete()
