# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""HPMC updaters.

HPMC updaters work with the `hpmc.integrate.HPMCIntegrator` to apply changes to
the system consistent with the particle shape and defined interaction energies.
The `BoxMC`, `Clusters`, and `MuVT` updaters apply trial moves that enable
enhanced sampling or the equilibration of different ensembles. `QuickCompress`
helps prepare non-overlapping configurations of particles in a given box shape.
"""

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
        betaP (hoomd.variant.variant_like): :math:`\frac{p}{k_{\mathrm{B}}T}`
            :math:`[\mathrm{length}^{-2}]` in 2D or
            :math:`[\mathrm{length}^{-3}]` in 3D.
        trigger (hoomd.trigger.trigger_like): Select the timesteps to perform
            box trial moves.

    Use `BoxMC` in conjunction with an HPMC integrator to allow the simulation
    box to undergo random fluctuations at constant pressure, or random
    deformations at constant volume. `BoxMC` supports both isotropic and
    anisotropic volume change moves as well as shearing of the simulation box. A
    single `BoxMC` instance may apply multiple types of box moves during a
    simulation run.

    .. rubric:: Box move types

    By default, no moves are applied (the *weight* values for
    all move types default to 0). In a given timestep, the type of move is
    selected randomly with probability:

    .. math::

        p = \frac{w_k}{\sum_k w_k}

    where :math:`w_k` is the weight of the move type.

    A given box move proposes a trial simulation box :math:`(L_x^t, L_y^t,
    L_z^t, xy^t, xz^t, yz^t)` as a change from the current box: :math:`(L_x,
    L_y, L_z, xy, xz, yz)`. The form of the change depends on the selected
    move type:

    * `volume` (``mode='standard'``): Change the volume (or area in 2D) of the
      simulation box while maining fixed aspect ratios :math:`Lx/Ly`,
      :math:`Lx/Lz`. In 3D:

      .. math::

          V^t &= V + u \\
          L_x^t &= \left( \frac{Lx}{Ly} \frac{Lx}{Lz} V^t \right)^{1/3} \\
          L_y^t &= L_x^t \frac{Ly}{Lx} \\
          L_z^t &= L_x^t \frac{Lz}{Lx} \\
          xy^t &= xy \\
          xz^t &= xz \\
          yz^t &= yz \\

      where :math:`u` is a random value uniformly distributed in the interval
      :math:`[-\delta_\mathrm{volume}, \delta_\mathrm{volume}]`.

      In 2D:

      .. math::

          V^t &= V + u \\
          L_x^t &= \left( \frac{Lx}{Ly} V^t \right)^{1/2} \\
          L_y^t &= L_x^t \frac{Ly}{Lx} \\
          xy^t &= xy \\

    * `volume` (``mode='ln'``): Change the volume (or area in 2D) of the
      simulation box while maining fixed aspect ratios :math:`Lx/Ly`,
      :math:`Lx/Lz`. In 3D:

      .. math::

          V^t &= V e^u \\
          L_x^t &= \left( \frac{Lx}{Ly} \frac{Lx}{Lz} V^t \right)^{1/3} \\
          L_y^t &= L_x^t \frac{Ly}{Lx} \\
          L_z^t &= L_x^t \frac{Lz}{Lx} \\
          xy^t &= xy \\
          xz^t &= xz \\
          yz^t &= yz \\

      where :math:`u` is a random value uniformly distributed in the interval
      :math:`[-\delta_\mathrm{volume}, \delta_\mathrm{volume}]`.

      In 2D:

      .. math::

          V^t &= V e^u \\
          L_x^t &= \left( \frac{Lx}{Ly} V^t \right)^{1/2} \\
          L_y^t &= L_x^t \frac{Ly}{Lx} \\
          xy^t &= xy \\
    * `aspect`: Change the aspect ratio of the simulation box while maintaining
      a fixed volume. In 3D:

      .. math::

          L_k^t & = \begin{cases} L_k(1 + a) & u < 0.5 \\
                                L_k \frac{1}{1+a}  & u \ge 0.5
                  \end{cases} \\
          L_{m \ne k}^t & = L_m \sqrt{\frac{L_k}{L_k^t}} &
          xy^t &= xy \\
          xz^t &= xz \\
          yz^t &= yz \\

      where :math:`u` is a random value uniformly distributed in the interval
      :math:`[0, 1]`, :math:`a` is a random value uniformly distributed in the
      interval :math:`[0, \delta_\mathrm{aspect}]` and :math:`k` is randomly
      chosen uniformly from the set :math:`\{x, y, z\}`.

      In 2D:

      .. math::

          L_k^t & = \begin{cases} L_k(1 + a) & u < 0.5 \\
                                L_k \frac{1}{1+a}  & u \ge 0.5
                    \end{cases} \\
          L_{m \ne k}^t & = L_m \frac{L_k}{L_k^t} \\
          xy^t &= xy \\
    * `length`: Change the box lengths:

      .. math::

          L_k^t =  L_k + u

      where :math:`u` is a random value uniformly distributed in the interval
      :math:`[-\delta_{\mathrm{length},k}, -\delta_{\mathrm{length},k}]`,
      and :math:`k` is randomly chosen uniformly from the set
      :math:`\{a : a \in \{x, y, z\}, \delta_{\mathrm{length},a} \ne 0 \}`.
    * `shear`: Change the box shear parameters. In 3D:

      .. math::

          (xy^t, xz^t, yz^t) =
          \begin{cases}
          \left(xy + s_{xy},
                \enspace xz,
                \enspace yz \right) & u < \frac{1}{3} \\
          \left( xy^t = xy,
                \enspace xz + s_{xz},
                \enspace yz \right) & \frac{1}{3} \le u < \frac{2}{3} \\
          \left( xy^t = xy,
                \enspace xz,
                \enspace yz + s_{yz} \right) & \frac{2}{3} \le u \le 1 \\
          \end{cases} \\

      where :math:`u` is a random value uniformly distributed in the interval
      :math:`[0, 1]` and :math:`s_k` is a random value uniformly distributed in
      the interval :math:`[-\delta_{\mathrm{shear},k},
      \delta_{\mathrm{shear},k}]`. `BoxMC` attempts and records trial moves for
      shear parameters even when :math:`\delta_{\mathrm{shear},k}=0`.

      In 2D:

      .. math::

         xy^t = xy + s_{xy}

    .. rubric:: Acceptance

    All particle particle positions are scaled into the trial box to form the
    trial configuration :math:`C^t`:

    .. math::

        \vec{r}_i^t = s_x \vec{a}_1^t + s_y \vec{a}_2^t +
                               s_z \vec{a}_3^t -
                    \frac{\vec{a}_1^t + \vec{a}_2^t + \vec{a}_3^t}{2}

    where :math:`\vec{a}_k^t` are the new box vectors determined by
    :math:`(L_x^t, L_y^t, L_z^t, xy^t, xz^t, yz^t)` and the scale factors are
    determined by the current particle position :math:`\vec{r}_i` and the box
    vectors :math:`\vec{a}_k`:

    .. math::

        \vec{r}_i = s_x \vec{a}_1 + s_y \vec{a}_2 + s_z \vec{a}_3 -
                    \frac{\vec{a}_1 + \vec{a}_2 + \vec{a}_3}{2}

    The trial move is accepted with the probability:

    .. math::

        p_\mathrm{accept} =
        \begin{cases}
        \exp(-(\beta \Delta H + \beta \Delta U)) &
        \beta \Delta H + \beta \Delta U > 0 \\
        1 & \beta \Delta H + \beta \Delta U \le 0 \\
        \end{cases}

    where :math:`\Delta U = U^t - U` is the difference in potential energy,
    :math:`\beta \Delta H = \beta P (V^t - V) - N_\mathrm{particles} \cdot
    \ln(V^t / V)` for most move types. It is :math:`\beta P (V^t - V) -
    (N_\mathrm{particles}+1) \cdot \ln(V^t / V)` for ln volume moves.

    When the trial move is accepted, the system state is set to the the trial
    configuration. When it is not accepted, the move is rejected and the state
    is not modified.

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
              is box area (2D) or volume (3D) :math:`\delta_\mathrm{volume}`.

        aspect (dict):
            Parameters for isovolume aspect ratio moves. The dictionary has the
            following keys:

            * ``weight`` (float) - Relative weight of aspect box moves.
            * ``delta`` (float) - Maximum relative change of box aspect ratio
              :math:`\delta_\mathrm{aspect} [\mathrm{dimensionless}]`.

        length (dict):
            Parameters for isobaric box length moves that change box lengths
            independently. The dictionary has the following keys:

            * ``weight`` (float) - Maximum change of HOOMD-blue box parameters
              Lx, Ly, and Lz.
            * ``delta`` (tuple[float, float, float]) - Maximum change of the
              box lengths :math:`(\delta_{\mathrm{length},x},
              \delta_{\mathrm{length},y}, \delta_{\mathrm{length},z})
              [\mathrm{length}]`.

        shear (dict):
            Parameters for isovolume box shear moves. The dictionary has the
            following keys:

            * ``weight`` (float) - Relative weight of shear box moves.
            * ``delta`` (tuple[float, float, float]) -  maximum change of the
              box tilt factor :math:`(\delta_{\mathrm{shear},xy},
              \delta_{\mathrm{shear},xz}, \delta_{\mathrm{shear},yz})
              [\mathrm{dimensionless}]`.
            * ``reduce`` (float) - Maximum number of lattice vectors of shear
              to allow before applying lattice reduction. Values less than 0.5
              disable shear reduction.

        instance (int):
            When using multiple `BoxMC` updaters in a single simulation,
            give each a unique value for `instance` so they generate
            different streams of random numbers.
    """

    def __init__(self, trigger, betaP):
        super().__init__(trigger)

        _default_dict = dict(weight=0.0, delta=0.0)
        param_dict = ParameterDict(
            volume={
                "mode": hoomd.data.typeconverter.OnlyFrom(['standard', 'ln']),
                **_default_dict
            },
            aspect=_default_dict,
            length=dict(weight=0.0, delta=(0.0,) * 3),
            shear=dict(weight=0.0, delta=(0.0,) * 3, reduce=0.0),
            betaP=hoomd.variant.Variant,
            instance=int,
        )
        self._param_dict.update(param_dict)
        self.volume["mode"] = "standard"
        self.betaP = betaP
        self.instance = 0

    def _attach_hook(self):
        # HPMC uses RNGs. Warn the user if they did not set the seed.
        self._simulation._warn_if_seed_unset()
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        self._cpp_obj = _hpmc.UpdaterBoxMC(self._simulation.state._cpp_sys_def,
                                           self.trigger, integrator._cpp_obj,
                                           self.betaP)

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

    Gibbs ensemble simulations are also supported, where particles and volume
    are swapped between two or more boxes.  Every box correspond to one MPI
    partition, and can therefore run on multiple ranks. Use the
    ``ranks_per_partition`` argument of `hoomd.communicator.Communicator` to
    enable partitioned simulations.

    .. rubric:: Mixed precision

    `MuVT` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    Note:
        Multiple Gibbs ensembles are also supported in a single parallel job,
        with the ``ngibbs`` option to update.muvt(), where the number of
        partitions can be a multiple of ``ngibbs``.

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
        fugacity (`TypeParameter` [ ``particle type``, `float`]):
            Particle fugacity
            :math:`[\mathrm{volume}^{-1}]` (**default:** 0).
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
        self._append_typeparam(typeparam_fugacity)

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        cpp_cls_name = "UpdaterMuVT"
        cpp_cls_name += integrator.__class__.__name__
        cpp_cls = getattr(_hpmc, cpp_cls_name)

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self.trigger, integrator._cpp_obj, self.ngibbs)

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

        None when not attached.
        """
        N_dict = None
        if self._attached:
            N_dict = self._cpp_obj.N

        return N_dict


class Shape(Updater):
    """Apply shape updates to the shape definitions defined in the integrator.

    See Also:
        `hoomd.hpmc.shape_move` describes the shape alchemy algorithm.

    Args:
        trigger (hoomd.trigger.trigger_like): Call the updater on triggered time
            steps.

        shape_move (ShapeMove): Type of shape move to apply when updating shape
            definitions

        pretend (`bool`, optional): When True the updater will not
            actually update the shape definitions. Instead, moves will be
            proposed and the acceptance statistics will be updated correctly
            (**default**: `False`).

        type_select (`int`, optional): Number of types to change every time
            the updater is called (**default**: 1).

        nsweeps (`int`, optional): Number of times to update shape
            definitions during each triggered timesteps (**default**: 1).

    .. rubric:: Shape support.

    See `hoomd.hpmc.shape_move` for supported shapes.

    Example::

        mc = hoomd.hpmc.integrate.ConvexPolyhedron()
        mc.shape["A"] = dict(vertices=numpy.asarray([(1, 1, 1), (-1, -1, 1),
                                                    (1, -1, -1), (-1, 1, -1)]))
        vertex_move = hoomd.hpmc.shape_move.Vertex(stepsize={'A': 0.01},
                                                   param_ratio=0.2,
                                                   volume=1.0)
        updater = hoomd.hpmc.update.Shape(shape_move=vertex_move,
                                          trigger=hoomd.trigger.Periodic(1),
                                          type_select=1,
                                          nsweeps=1)

    Attributes:
        trigger (Trigger): Call the updater on triggered time steps.

        shape_move (ShapeMove): Type of shape move to apply when updating shape
            definitions

        pretend (bool): When True the updater will not actually update the shape
            definitions, instead moves will be proposed and the acceptance
            statistics will be updated correctly.

        type_select (int): Number of types to change every time the updater is
            called.

        nsweeps (int): Number of times to update shape definitions during each
            triggered timesteps.
    """

    def __init__(self,
                 trigger,
                 shape_move,
                 pretend=False,
                 type_select=1,
                 nsweeps=1):
        super().__init__(trigger)
        param_dict = ParameterDict(shape_move=hoomd.hpmc.shape_move.ShapeMove,
                                   pretend=bool(pretend),
                                   type_select=int(type_select),
                                   nsweeps=int(nsweeps))
        param_dict["shape_move"] = shape_move
        self._param_dict.update(param_dict)

    def _setattr_param(self, attr, value):
        if attr == "shape_move":
            self._set_shape_move(value)
            return
        super()._setattr_param(attr, value)

    def _set_shape_move(self, new_move):
        """Handles the adding and detaching of shape_move objects."""
        if new_move is self.shape_move:
            return

        old_move = self.shape_move

        if old_move is not None:
            if self._attached:
                old_move._detach()

        if new_move is None:
            self._param_dict["shape_move"] = None
            return

        if self._attached:
            new_move._attach(self._simulation)
        self._param_dict["shape_move"] = new_move

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        # check for supported shapes is done in the shape move classes
        integrator_name = integrator.__class__.__name__
        updater_cls = getattr(_hpmc, 'UpdaterShape' + integrator_name)

        self.shape_move._attach(self._simulation)
        self._cpp_obj = updater_cls(self._simulation.state._cpp_sys_def,
                                    self.trigger, integrator._cpp_obj,
                                    self.shape_move._cpp_obj)

    @log(category='sequence', requires_run=True)
    def shape_moves(self):
        """tuple[int, int]: Count of the accepted and rejected shape moves."""
        return self._cpp_obj.getShapeMovesCount()

    @log(category='scalar', requires_run=True)
    def particle_volumes(self):
        """list[float]: Volume of a single particle for each type."""
        return self._cpp_obj.particle_volumes

    @log(category="scalar", requires_run=True)
    def shape_move_energy(self):
        """float: Energy penalty due to shape changes."""
        return self._cpp_obj.getShapeMoveEnergy(self._simulation.timestep)


class VirtualClusterMoves(Updater):
    """Apply virtual move Monte Carlo moves.

    See Whitelam and Geissler (2007).
    """

    def __init__(self, trigger=1, attempts_per_particle=1, beta_ficticious=1.0):
        super().__init__(trigger)
        param_dict = ParameterDict(attempts_per_particle=float(attempts_per_particle),
                                   beta_ficticious=float(beta_ficticious))
        self._param_dict.update(param_dict)
        self.instance = 0

    def _attach_hook(self):
        self._simulation._warn_if_seed_unset()
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")
        cpp_cls_name = "UpdaterVMMC"
        cpp_cls_name += integrator.__class__.__name__
        cpp_cls = getattr(_hpmc, cpp_cls_name)
        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def, self.trigger, integrator._cpp_obj
        )

    @log(category='sequence', requires_run=True)
    def move_counts(self):
        return self._cpp_obj.getCounters(1).counts

    @log(category='scalar', requires_run=True)
    def acceptance_rate(self):
        acc, rej = self._cpp_obj.getCounters(1).counts
        if acc + rej == 0:
            return 0.0
        else:
            return acc / (acc + rej)

    @log(category='scalar', requires_run=True)
    def average_cluster_size(self):
        return self._cpp_obj.getCounters(1).average_cluster_size


class Clusters(Updater):
    """Apply geometric cluster algorithm (GCA) moves.

    Args:
        pivot_move_probability (float): Set the probability for attempting a
                                        pivot move.
        flip_probability (float): Set the probability for transforming an
                                 individual cluster.
        trigger (hoomd.trigger.trigger_like): Select the timesteps on which to
            perform cluster moves.

    The GCA as described in Liu and Lujten (2004),
    http://doi.org/10.1103/PhysRevLett.92.035504 is used for hard shape, patch
    interactions and depletants. Implicit depletants are supported and simulated
    on-the-fly, as if they were present in the actual system.

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

    def _attach_hook(self):
        # HPMC uses RNGs. Warn the user if they did not set the seed.
        self._simulation._warn_if_seed_unset()
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
                                    self.trigger, integrator._cpp_obj,
                                    self._cpp_cell)
        else:
            self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                    self.trigger, integrator._cpp_obj)

    @log(requires_run=True)
    def avg_cluster_size(self):
        """float: the typical size of clusters.

        None when not attached.
        """
        counter = self._cpp_obj.getCounters(1)
        return counter.average_cluster_size


class QuickCompress(Updater):
    r"""Quickly compress a hard particle system to a target box.

    Args:
        trigger (hoomd.trigger.trigger_like): Update the box dimensions on
            triggered time steps.

        target_box (`hoomd.box.box_like` or `hoomd.variant.box.BoxVariant`):
            Dimensions of the target box.

        max_overlaps_per_particle (float): The maximum number of overlaps to
            allow per particle (may be less than 1 - e.g.
            up to 250 overlaps would be allowed when in a system of 1000
            particles when max_overlaps_per_particle=0.25).

        min_scale (float): The minimum scale factor to apply to box dimensions.

        allow_unsafe_resize (bool): When `True`, box moves are proposed
            independent of particle translational move sizes.

    Use `QuickCompress` in conjunction with an HPMC integrator to scale the
    system to a target box size. `QuickCompress` can typically compress dilute
    systems to near random close packing densities in tens of thousands of time
    steps. For more control over the rate of compression, use a `BoxVariant` for
    `target_box`.

    `QuickCompress` operates by making small changes toward the `target_box`,
    but only when there are no particle overlaps in the current simulation
    state. In 3D:

    .. math::

          L_x' &= \begin{cases}
          \max( L_x \cdot s, L_{\mathrm{target},x} )
          & L_{\mathrm{target},x} < L_x \\
          \min( L_x / s, L_{\mathrm{target},x} )
          & L_{\mathrm{target},x} \ge L_x
          \end{cases} \\
          L_y' &= \begin{cases}
          \max( L_y \cdot s, L_{\mathrm{target},y} )
          & L_{\mathrm{target},y} < L_y \\
          \min( L_y / s, L_{\mathrm{target},y} )
          & L_{\mathrm{target},y} \ge L_y
          \end{cases} \\
          L_z' &= \begin{cases}
          \max( L_z \cdot s, L_{\mathrm{target},z} )
          & L_{\mathrm{target},z} < L_z \\
          \min( L_z / s, L_{\mathrm{target},z} )
          & L_{\mathrm{target},z} \ge L_z
          \end{cases} \\
          xy' &= \begin{cases}
          \min( xy + (1-s) \cdot xy_\mathrm{target}, xy_\mathrm{target} )
          & xy_\mathrm{target} < xy \\
          \max( xy + (1-s) \cdot xy_\mathrm{target}, xy_\mathrm{target} )
          & xy_\mathrm{target} \ge xy
          \end{cases} \\
          xz' &= \begin{cases}
          \min( xz + (1-s) \cdot xz_\mathrm{target}, xz_\mathrm{target} )
          & xz_\mathrm{target} < xz \\
          \max( xz + (1-s) \cdot xz_\mathrm{target}, xz_\mathrm{target} )
          & xz_\mathrm{target} \ge xz
          \end{cases} \\
          yz' &= \begin{cases}
          \min( yz + (1-s) \cdot yz_\mathrm{target}, yz_\mathrm{target} )
          & yz_\mathrm{target} < yz \\
          \max( yz + (1-s) \cdot yz_\mathrm{target}, yz_\mathrm{target} )
          & yz_\mathrm{target} \ge yz
          \end{cases} \\

    and in 2D:

    .. math::

          L_x' &= \begin{cases}
          \max( L_x \cdot s, L_{\mathrm{target},x} )
          & L_{\mathrm{target},x} < L_x \\
          \min( L_x / s, L_{\mathrm{target},x} )
          & L_{\mathrm{target},x} \ge L_x
          \end{cases} \\
          L_y' &= \begin{cases}
          \max( L_y \cdot s, L_{\mathrm{target},y} )
          & L_{\mathrm{target},y} < L_y \\
          \min( L_y / s, L_{\mathrm{target},y} )
          & L_{\mathrm{target},y} \ge L_y
          \end{cases} \\
          L_z' &= L_z \\
          xy' &= \begin{cases}
          \min( xy + (1-s) \cdot xy_\mathrm{target}, xy_\mathrm{target} )
          & xy_\mathrm{target} < xy \\
          \max( xy + (1-s) \cdot xy_\mathrm{target}, xy_\mathrm{target} )
          & xy_\mathrm{target} \ge xy
          \end{cases} \\
          xz' &= xz \\
          yz' &= yz \\

    where the current simulation box is :math:`(L_x, L_y, L_z, xy, xz, yz)`,
    the target is :math:`(L_{\mathrm{target},x}, L_{\mathrm{target},y},
    L_{\mathrm{target},z}, xy_\mathrm{target}, xz_\mathrm{target},
    yz_\mathrm{target})`, the new simulation box set is
    :math:`(L_x', L_y', L_z', xy', xz', yz')` and :math:`s` is the scale factor
    chosen for this step (see below). `QuickCompress` scales particle
    coordinates (see `BoxMC` for details) when it sets a new box.

    When there are more than ``max_overlaps_per_particle * N_particles`` hard
    particle overlaps in the system in the new box, the box move is rejected.
    Otherwise, the small number of overlaps remain when the new box is set.
    `QuickCompress` then waits until `hoomd.hpmc.integrate.HPMCIntegrator` makes
    local MC trial moves that remove all overlaps.

    `QuickCompress` adjusts the value of :math:`s` based on the particle and
    translational trial move sizes to ensure that the trial moves will be able
    to remove the overlaps. It randomly chooses a value of :math:`s` uniformly
    distributed between ``max(min_scale, 1.0 - min_move_size / max_diameter)``
    and 1.0 where ``min_move_size`` is the smallest MC translational move size
    adjusted by the acceptance ratio and ``max_diameter`` is the circumsphere
    diameter of the largest particle type. If `allow_unsafe_resize` is `True`,
    box move sizes will be uniformly distributed between ``min_scale`` and 1.0
    (with no consideration of ``min_move_size``).

    When using a `BoxVariant` for `target_box`, `complete` returns `True` if the
    current simulation box is equal to the box corresponding to `target_box`
    evaluated at the current timestep and there are no overlaps in the system.
    To ensure the updater has compressed the system to the final target box, use
    a condition that checks both the `complete` attribute of the updater and the
    simulation timestep. For example::

        target_box = hoomd.variant.box.InverseVolumeRamp(
            sim.state.box, sim.state.box.volume / 2, 0, 1_000)
        qc = hoomd.hpmc.update.QuickCompress(10, target_box)
        while (
            sim.timestep < target_box.t_ramp + target_box.t_start or
            not qc.complete):
            sim.run(100)

    Tip:
        Use the `hoomd.hpmc.tune.MoveSize` in conjunction with
        `QuickCompress` to adjust the move sizes to maintain a constant
        acceptance ratio as the density of the system increases.


    Warning:
        When the smallest MC translational move size is 0, `allow_unsafe_resize`
        must be set to `True` to progress toward the target box. Decrease
        `max_overlaps_per_particle` when using this setting to prevent
        unresolvable overlaps.

    Warning:
        Use `QuickCompress` *OR* `BoxMC`. Do not use both at the same time.

    .. rubric:: Mixed precision

    `QuickCompress` uses reduced precision floating point arithmetic when
    checking for particle overlaps in the local particle reference frame.

    Attributes:
        trigger (Trigger): Update the box dimensions on triggered time steps.

        target_box (BoxVariant): The variant for the dimensions of the target
            box.

        max_overlaps_per_particle (float): The maximum number of overlaps to
            allow per particle (may be less than 1 - e.g.
            up to 250 overlaps would be allowed when in a system of 1000
            particles when max_overlaps_per_particle=0.25).

        min_scale (float): The minimum scale factor to apply to box dimensions.

        instance (int):
            When using multiple `QuickCompress` updaters in a single simulation,
            give each a unique value for `instance` so that they generate
            different streams of random numbers.

        allow_unsafe_resize (bool): Flag that determines whether
    """

    def __init__(self,
                 trigger,
                 target_box,
                 max_overlaps_per_particle=0.25,
                 min_scale=0.99,
                 allow_unsafe_resize=False):
        super().__init__(trigger)

        param_dict = ParameterDict(max_overlaps_per_particle=float,
                                   min_scale=float,
                                   target_box=hoomd.variant.box.BoxVariant,
                                   instance=int,
                                   allow_unsafe_resize=bool)
        if isinstance(target_box, hoomd.Box):
            target_box = hoomd.variant.box.Constant(target_box)
        param_dict['max_overlaps_per_particle'] = max_overlaps_per_particle
        param_dict['min_scale'] = min_scale
        param_dict['target_box'] = target_box
        param_dict['allow_unsafe_resize'] = allow_unsafe_resize

        self._param_dict.update(param_dict)

        self.instance = 0

    def _attach_hook(self):
        # HPMC uses RNGs. Warn the user if they did not set the seed.
        self._simulation._warn_if_seed_unset()
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        self._cpp_obj = _hpmc.UpdaterQuickCompress(
            self._simulation.state._cpp_sys_def, self.trigger,
            integrator._cpp_obj, self.max_overlaps_per_particle, self.min_scale,
            self.target_box)

    @property
    def complete(self):
        """True when the box has achieved the target."""
        if not self._attached:
            return False

        return self._cpp_obj.isComplete()
