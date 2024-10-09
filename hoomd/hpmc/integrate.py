# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Hard particle Monte Carlo integrators.

.. rubric:: Metropolis Monte Carlo

The hard particle Monte Carlo (HPMC) integrator `HPMCIntegrator` samples
equilibrium system states using the Metropolis Monte Carlo method. In this
method, `HPMCIntegrator` takes the existing system state in the configuration
:math:`C = (\vec{r}_0, \vec{r}_1, \ldots \vec{r}_{N_\mathrm{particles}-1},
\mathbf{q}_0, \mathbf{q}_2, \ldots \mathbf{q}_{N_\mathrm{particles}-1})` with
potential energy :math:`U` and perturbs it to a trial configuration :math:`C^t`
with potential energy :math:`U^t` leading to an energy difference :math:`\Delta
U = U^t - U`. The trial move is accepted with the probability:

.. math::

    p_\mathrm{accept} =
    \begin{cases}
      \exp(-\beta \Delta U) & \Delta U > 0 \\
      1 & \Delta U \le 0 \\
    \end{cases}

When the trial move is accepted, the system state is set to the the trial
configuration. When it is not accepted, the move is rejected and the state is
not modified.

.. rubric:: Temperature

HPMC assumes that :math:`\beta = \frac{1}{kT} = 1`. This is not relevant to
systems of purely hard particles where :math:`\Delta U` is either 0 or
:math:`\infty`. To adjust the effective temperature in systems with finite
interactions (see *Energy evaluation* below), scale the magnitude of the
energetic interactions accordingly.

.. rubric:: Local trial moves

`HPMCIntegrator` generates local trial moves for a single particle :math:`i` at
a time. The move is either a translation move or a rotation move, selected
randomly with the probability of a translation move set by
`HPMCIntegrator.translation_move_probability` (:math:`p_\mathrm{translation}`).

The form of the trial move depends on the dimensionality of the system.
Let :math:`u` be a random value in the interval :math:`[0,1]`, :math:`\vec{v}`
be a random vector uniformly distributed within the ball of radius 1, and
:math:`\mathbf{w}` be a random unit quaternion from the set of uniformly
distributed rotations. Then the 3D trial move for particle :math:`i` is:

.. math::

    \begin{cases}
      \left( \begin{array}{l}
      \vec{r}^t_i = \vec{r}_i + d_i \vec{v}, \\
      \mathbf{q}^t_i = \mathbf{q}_i
      \end{array} \right) & u \le p_\mathrm{translation} \\
      \left( \begin{array}{l}
      \vec{r}^t_i = \vec{r}_i, \\
      \mathbf{q}^t_i = \frac{\mathbf{q}_i + a_i \mathbf{w}}
        {\vert \mathbf{q}_i + a_i \mathbf{w} \vert}
      \end{array} \right) & u > p_\mathrm{translation} \\
    \end{cases}

where :math:`d_i` is the translation move size for particle :math:`i` (set by
particle type with `HPMCIntegrator.d`) and :math:`a_i` is the rotation move size
(set by particle type with `HPMCIntegrator.a`).

In 2D boxes, let :math:`\vec{v}` be a random vector uniformly distributed within
the disk of radius 1 in the x,y plane and :math:`\alpha` be a random angle in
radians in the interval :math:`[-a_i,a_i]`. Form a quaternion that rotates about
the z axis by :math:`\alpha`: :math:`\mathbf{w} = (\cos(\alpha/2), 0, 0,
\sin(\alpha/2))`. The 2D trial move for particle :math:`i` is:

.. math::

    \begin{cases}
      \left( \begin{array}{l}
      \vec{r}^t_i = \vec{r}_i + d_i \vec{v}, \\
      \mathbf{q}^t_i = \mathbf{q}_i
      \end{array} \right) & u \le p_\mathrm{translation} \\
      \left( \begin{array}{l}
      \vec{r}^t_i = \vec{r}_i, \\
      \mathbf{q}^t_i = \frac{\mathbf{q}_i \cdot \mathbf{w}}
        {\vert \mathbf{q}_i \cdot \mathbf{w} \vert}
      \end{array} \right) & u > p_\mathrm{translation} \\
    \end{cases}

Note:
    For non-orientable spheres, :math:`p_\mathrm{translation} = 1`.

.. rubric:: Timesteps

In the serial CPU implementation, `HPMCIntegrator` performs `nselect
<HPMCIntegrator.nselect>` trial moves per particle in each timestep (which
defaults to 4). To achieve detailed balance at the level of a timestep,
`HPMCIntegrator` randomly chooses with equal probability to loop through
particles in forward index or reverse index order (random selection severely
degrades performance due to cache incoherency). In the GPU and MPI
implementations, trial moves are performed in parallel for particles in active
domains while leaving particles on the border fixed (see `Anderson 2016
<https://dx.doi.org/10.1016/j.cpc.2016.02.024>`_ for a full description). As a
consequence, a single timestep may perform more or less than ``nselect`` trial
moves per particle when using the parallel code paths. Monitor the number of
trial moves performed with `HPMCIntegrator.translate_moves` and
`HPMCIntegrator.rotate_moves`.

.. rubric:: Random numbers

`HPMCIntegrator` uses a pseudorandom number stream to generate the trial moves.
Set the seed using `hoomd.Simulation.seed`. Given the same seed, the same
initial configuration, and the same execution configuration (device and MPI
configuration), `HPMCIntegrator`, will produce exactly the same trajectory.

Note:
    Due to limited floating point precision, full trajectory reproducibility
    is only possible with the same binary installation running on the same
    hardware device. Compiler optimizations, changes to the HOOMD source code,
    and machine specific code paths may lead to different trajectories.

.. rubric:: Energy evaluation

`HPMCIntegrator` evaluates the energy of a configuration from a number of terms:

.. math::

    U = U_{\mathrm{pair}} + U_{\mathrm{shape}} + U_{\mathrm{external}}

To enable simulations of small systems, the pair and shape energies evaluate
interactions between pairs of particles in multiple box images:

.. math::

    U_{\mathrm{pair}} = &
            \sum_{i=0}^{N_\mathrm{particles}-1}
            \sum_{j=i+1}^{N_\mathrm{particles}-1}
            U_{\mathrm{pair},ij}(\vec{r}_j - \vec{r}_i,
                                 \mathbf{q}_i,
                                 \mathbf{q}_j) \\
            + & \sum_{i=0}^{N_\mathrm{particles}-1}
            \sum_{j=i}^{N_\mathrm{particles}-1}
            \sum_{\vec{A} \in B_\mathrm{images}, \vec{A} \ne \vec{0}}
            U_{\mathrm{pair},ij}(\vec{r}_j - (\vec{r}_i + \vec{A}),
                                 \mathbf{q}_i,
                                 \mathbf{q}_j)

where :math:`\vec{A} = h\vec{a}_1 + k\vec{a}_2 + l\vec{a}_3` is a vector that
translates by periodic box images and the set of box images includes all image
vectors necessary to find interactions between particles in the primary image
with particles in periodic images The first sum evaluates interactions between
particle :math:`i` with other particles (not itself) in the primary box image.
The second sum evaluates interactions between particle :math:`i` and all
potentially interacting periodic images of all particles (including itself).
`HPMCIntegrator` computes :math:`U_{\mathrm{shape}}` similarly (see below).

External potentials apply to each particle individually:

.. math::

    U_\mathrm{external} =
        \sum_{i=0}^\mathrm{N_particles-1} U_{\mathrm{external},i}(\vec{r}_i,
                                                                 \mathbf{q}_i)

Potential classes in :doc:`module-hpmc-pair` evaluate
:math:`U_{\mathrm{pair},ij}`. HPMC sums all the `Pair <hoomd.hpmc.pair.Pair>`
potentials in `pair_potentials <HPMCIntegrator.pair_potentials>` during
integration.

Similarly, potential classes in :doc:`module-hpmc-external` evaluate
:math:`U_{\mathrm{external},i}`. Assign a class instance to
`external_potential <HPMCIntegrator.external_potential>` or add instances to
`external_potentials <HPMCIntegrator.external_potentials>` to apply during
integration.

.. rubric:: Shape overlap tests

`HPMCIntegrator` performs shape overlap tests to evaluate
:math:`U_{\mathrm{shape}}`. Let :math:`S` be the set of all points inside the
shape in the local coordinate system of the shape:

.. math::

    S = \{ \vec{a} \in \mathbb{R}^3 :
           \vec{a} \enspace \mathrm{inside\ the\ shape} \}

See the subclasses of `HPMCIntegrator` for formal definitions of the shapes,
whose parameters are set by particle type. Let :math:`S_i` refer specifically
to the shape for particle :math:`i`.

The quaternion :math:`\mathbf{q}` represents a rotation of the shape from its
local coordinate system to the given orientation:

.. math::

    S(\mathbf{q}) = \{ \mathbf{q}\vec{a}\mathbf{q}^* : \vec{a} \in S \}

The full transformation from the local shape coordinate system to the simulation
box coordinate system includes a rotation and translation:

.. math::

    S(\mathbf{q}, \vec{r}) = \{ \mathbf{q}\vec{a}\mathbf{q}^* + \vec{r} :
                                \vec{a} \in S \}

`HPMCIntegrator` defines the shape overlap test for two shapes:

.. math::

    \mathrm{overlap}(S_1, S_2) = S_1 \bigcap S_2

To check for overlaps between two particles in the box, rotating both shapes
from their local frame to the box frame, and translate :math:`S_2` relative to
particle 1:

.. math::

    \mathrm{overlap}(S_1(\mathbf{q}_1), S_2(\mathbf{q}_2,
                                            \vec{r}_2 - \vec{r}_1))

The complete hard shape interaction energy for a given configuration is:

.. math::

    U_\mathrm{shape} = \quad & \infty
            \cdot
            \sum_{i=0}^{N_\mathrm{particles}-1}
            \sum_{j=i+1}^{N_\mathrm{particles}-1}
            \left[
            \mathrm{overlap}\left(
            S_i(\mathbf{q}_i),
            S_j(\mathbf{q}_j, \vec{r}_j - \vec{r}_i)
            \right) \ne \emptyset
            \right]
            \\
            + & \infty \cdot \sum_{i=0}^{N_\mathrm{particles}-1}
            \sum_{j=i}^{N_\mathrm{particles}-1}
            \sum_{\vec{A} \in B_\mathrm{images}, \vec{A} \ne \vec{0}}
            \left[
            \mathrm{overlap}\left(
            S_i(\mathbf{q}_i),
            S_j(\mathbf{q}_j, \vec{r}_j - (\vec{r}_i + \vec{A}))
            \right) \ne \emptyset
            \right]

where the square brackets denote the Iverson bracket.

Note:
    While this notation is written in as sums over all particles
    `HPMCIntegrator` uses spatial data structures to evaluate these calculations
    efficiently. Similarly, while the overlap test is notated as a set
    intersection, `HPMCIntegrator` employs efficient computational geometry
    algorithms to determine whether there is or is not an overlap.

.. rubric:: Implicit depletants

Set `HPMCIntegrator.depletant_fugacity` to activate the implicit depletant code
path. This inerts depletant particles during every trial move and modifies the
acceptance criterion accordingly. See `Glaser 2015
<https://dx.doi.org/10.1063/1.4935175>`_ for details.

.. deprecated:: 4.4.0

    ``depletant_fugacity > 0`` is deprecated.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere
"""

from hoomd import _hoomd
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from hoomd.data.typeparam import TypeParameter
from hoomd.error import DataAccessError
from hoomd.hpmc import _hpmc
from hoomd.operation import Integrator
from hoomd.logging import log
import hoomd
import json
import warnings


class HPMCIntegrator(Integrator):
    """Base class hard particle Monte Carlo integrator.

    `HPMCIntegrator` is the base class for all HPMC integrators. The attributes
    documented here are available to all HPMC integrators.

    See Also:
        The module level documentation `hoomd.hpmc.integrate` describes the
        hard particle Monte Carlo algorithm.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    .. rubric:: Ignoring overlap checks

    Set elements of `interaction_matrix` to `False` to disable overlap checks
    between specific pairs of particle types.

    .. rubric:: Writing type_shapes to GSD files.

    Use a Logger in combination with a HPMC integrator and a GSD writer to write
    ``type_shapes`` to the GSD file for use with OVITO. For example::

        mc = hoomd.hpmc.integrate.Sphere()
        log = hoomd.logging.Logger()
        log.add(mc, quantities=['type_shapes'])
        gsd = hoomd.write.GSD(
            'trajectory.gsd', hoomd.trigger.Periodic(1000), log=log)

    .. rubric:: Threading

    HPMC integrators use threaded execution on multiple CPU cores only when
    placing implicit depletants (``depletant_fugacity != 0``).

    .. deprecated:: 4.4.0

        ``num_cpu_threads >= 1`` is deprecated. Set ``num_cpu_threads = 1``.

    .. rubric:: Mixed precision

    All HPMC integrators use reduced precision floating point arithmetic when
    checking for particle overlaps in the local particle reference frame.

    .. rubric:: Parameters

    Attributes:
        a (`TypeParameter` [``particle type``, `float`]):
            Maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.

        d (`TypeParameter` [``particle type``, `float`]):
            Maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.

        depletant_fugacity (`TypeParameter` [ ``particle type``, `float`]):
            Depletant fugacity
            :math:`[\\mathrm{volume}^{-1}]` (**default:** 0)

            Allows setting the fugacity per particle type, e.g. ``'A'``
            refers to a depletant of type **A**.

        depletant_ntrial (`TypeParameter` [``particle type``, `int`]):
            Multiplicative factor for the number of times a depletant is
            inserted. This factor is accounted for in the acceptance criterion
            so that detailed balance is unchanged. Higher values of ntrial (than
            one) can be used to reduce the variance of the free energy estimate
            and improve the acceptance rate of the Markov chain.

        interaction_matrix (`TypeParameter` [\
                            `tuple` [``particle type``, ``particle type``],\
                            `bool`]):
            Set to `False` for a pair of particle types to disable
            overlap checks between particles of those types (**default:**
            `True`).

        translation_move_probability (float): Fraction of moves to be selected
            as translation moves.

        nselect (int): Number of trial moves to perform per particle per
            timestep.

    .. rubric:: Attributes
    """
    _ext_module = _hpmc
    _remove_for_pickling = Integrator._remove_for_pickling + ('_cpp_cell',)
    _skip_for_equality = Integrator._skip_for_equality | {'_cpp_cell'}
    _cpp_cls = None

    def __init__(self, default_d, default_a, translation_move_probability,
                 nselect, kT):
        super().__init__()

        # Set base parameter dict for hpmc integrators
        param_dict = ParameterDict(
            translation_move_probability=float(translation_move_probability),
            nselect=int(nselect),
            kT=hoomd.variant.Variant)
        self._param_dict.update(param_dict)
        self.kT = kT

        self._external_potential = None

        # Set standard typeparameters for hpmc integrators
        typeparam_d = TypeParameter('d',
                                    type_kind='particle_types',
                                    param_dict=TypeParameterDict(
                                        float(default_d), len_keys=1))
        typeparam_a = TypeParameter('a',
                                    type_kind='particle_types',
                                    param_dict=TypeParameterDict(
                                        float(default_a), len_keys=1))

        typeparam_fugacity = TypeParameter('depletant_fugacity',
                                           type_kind='particle_types',
                                           param_dict=TypeParameterDict(
                                               0., len_keys=1))

        typeparam_ntrial = TypeParameter('depletant_ntrial',
                                         type_kind='particle_types',
                                         param_dict=TypeParameterDict(
                                             1, len_keys=1))

        typeparam_inter_matrix = TypeParameter('interaction_matrix',
                                               type_kind='particle_types',
                                               param_dict=TypeParameterDict(
                                                   True, len_keys=2))

        self._extend_typeparam([
            typeparam_d, typeparam_a, typeparam_fugacity, typeparam_ntrial,
            typeparam_inter_matrix
        ])

        self._pair_potentials = hoomd.data.syncedlist.SyncedList(
            hoomd.hpmc.pair.Pair,
            hoomd.data.syncedlist._PartialGetAttr('_cpp_obj'))

        self._external_potentials = hoomd.data.syncedlist.SyncedList(
            hoomd.hpmc.external.External,
            hoomd.data.syncedlist._PartialGetAttr('_cpp_obj'))

    def _attach_hook(self):
        """Initialize the reflected c++ class.

        HPMC uses RNGs. Warn the user if they did not set the seed.
        """
        if any([f != 0 for f in self.depletant_fugacity.values()]):
            warnings.warn("depletant_fugacity > 0 is deprecated since 4.4.0.",
                          FutureWarning,
                          stacklevel=1)

            if (isinstance(self._simulation.device, hoomd.device.CPU)
                    and self._simulation.device.num_cpu_threads > 1):
                warnings.warn(
                    "num_cpu_threads > 1 is deprecated since 4.4.0. "
                    "Use num_cpu_threads=1.",
                    FutureWarning,
                    stacklevel=1)

        self._simulation._warn_if_seed_unset()
        sys_def = self._simulation.state._cpp_sys_def
        if (isinstance(self._simulation.device, hoomd.device.GPU)
                and (self._cpp_cls + 'GPU') in _hpmc.__dict__):
            self._cpp_cell = _hoomd.CellListGPU(sys_def)
            self._cpp_obj = getattr(self._ext_module,
                                    self._cpp_cls + 'GPU')(sys_def,
                                                           self._cpp_cell)
        else:
            if isinstance(self._simulation.device, hoomd.device.GPU):
                self._simulation.device._cpp_msg.warning(
                    "Falling back on CPU. No GPU implementation for shape.\n")
            self._cpp_obj = getattr(self._ext_module, self._cpp_cls)(sys_def)
            self._cpp_cell = None

        if self._external_potential is not None:
            self._external_potential._attach(self._simulation)
            self._cpp_obj.setExternalField(self._external_potential._cpp_obj)

        self._pair_potentials._sync(self._simulation,
                                    self._cpp_obj.pair_potentials)

        self._external_potentials._sync(self._simulation,
                                        self._cpp_obj.external_potentials)

        super()._attach_hook()

    def _detach_hook(self):
        if self._external_potential is not None:
            self._external_potential._detach()
        self._pair_potentials._unsync()
        self._external_potentials._unsync()

    # TODO need to validate somewhere that quaternions are normalized

    def _return_type_shapes(self):
        type_shapes = self._cpp_obj.getTypeShapesPy()
        ret = [json.loads(json_string) for json_string in type_shapes]
        return ret

    @property
    def pair_potentials(self):
        """list[hoomd.hpmc.pair.Pair]: Pair potentials to apply.

        Defines the pairwise particle interaction energy
        :math:`U_{\\mathrm{pair},ij}` as the sum over all potentials in the
        list.

        .. rubric:: Example

        .. invisible-code-block: python

            lennard_jones = hoomd.hpmc.pair.LennardJones()
            lennard_jones.params[('A', 'A')] = dict(
                epsilon=1, sigma=1, r_cut=2.5)

        .. code-block:: python

            simulation.operations.integrator.pair_potentials = [lennard_jones]
        """
        return self._pair_potentials

    @pair_potentials.setter
    def pair_potentials(self, value):
        self._pair_potentials.clear()
        self._pair_potentials.extend(value)

    @property
    def external_potentials(self):
        """list[hoomd.hpmc.external.External]: External potentials to apply.

        Defines the external energy :math:`U_{\\mathrm{external},i}` as the
        sum over all potentials in the list.

        .. rubric:: Example

        .. invisible-code-block: python

            linear = hoomd.hpmc.external.Linear(
                plane_origin=(0, 0, 0),
                plane_normal=(0, 0, -1))
            linear.alpha['A'] = 1.234

        .. code-block:: python

            simulation.operations.integrator.external_potentials = [linear]
        """
        return self._external_potentials

    @external_potentials.setter
    def external_potentials(self, value):
        self._external_potentials.clear()
        self._external_potentials.extend(value)

    @log(category='sequence', requires_run=True)
    def map_overlaps(self):
        """list[tuple[int, int]]: List of overlapping particles.

        The list contains one entry for each overlapping pair of particles. When
        a tuple ``(i,j)`` is present in the list, there is an overlap between
        the particles with tags ``i`` and ``j``.

        Attention:
            `map_overlaps` does not support MPI parallel simulations. It returns
            `None` when there is more than one MPI rank.
        """
        if self._simulation.device.communicator.num_ranks > 1:
            return None

        return self._cpp_obj.mapOverlaps()

    @log(requires_run=True)
    def overlaps(self):
        """int: Number of overlapping particle pairs."""
        self._cpp_obj.communicate(True)
        return self._cpp_obj.countOverlaps(False)

    @log(category='sequence', requires_run=True)
    def translate_moves(self):
        """tuple[int, int]: Count of the accepted and rejected translate moves.

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getCounters(1).translate

    @log(category='sequence', requires_run=True)
    def rotate_moves(self):
        """tuple[int, int]: Count of the accepted and rejected rotate moves.

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getCounters(1).rotate

    @log(requires_run=True)
    def mps(self):
        """float: Number of trial moves performed per second.

        Note:
            The count is reset at the start of each `hoomd.Simulation.run`.
        """
        return self._cpp_obj.getMPS()

    @property
    def counters(self):
        """dict: Trial move counters.

        The counter object has the following attributes:

        * ``translate``: `tuple` [`int`, `int`] - Number of accepted and
          rejected translate trial moves.
        * ``rotate``: `tuple` [`int`, `int`] - Number of accepted and rejected
          rotate trial moves.
        * ``overlap_checks``: `int` - Number of overlap checks performed.
        * ``overlap_errors``: `int` - Number of overlap checks that were too
          close to resolve.

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.
        """
        if self._attached:
            return self._cpp_obj.getCounters(1)
        else:
            raise DataAccessError("counters")

    @property
    def external_potential(self):
        r"""The user-defined potential energy field integrator.

        Defines the external energy :math:`U_{\mathrm{external},i}`. Defaults to
        `None`. May be set to an object from :doc:`module-hpmc-external`.

        .. deprecated:: 4.8.0

            Use `external_potentials`.
        """
        warnings.warn(
            "external_potential is deprecated since 4.8.0. "
            "Use external_potentials (when possible).",
            FutureWarning,
            stacklevel=2)
        return self._external_potential

    @external_potential.setter
    def external_potential(self, new_external_potential):
        warnings.warn(
            "external_potential is deprecated since 4.8.0. "
            "Use external_potentials (when possible).",
            FutureWarning,
            stacklevel=4)
        if not isinstance(new_external_potential,
                          hoomd.hpmc.external.field.ExternalField):
            msg = 'External potentials should be an instance of '
            msg += 'hoomd.hpmc.field.external.ExternalField.'
            raise TypeError(msg)
        if self._attached:
            new_external_potential._attach(self._simulation)
            self._cpp_obj.setExternalField(new_external_potential._cpp_obj)
            if self._external_potential is not None:
                self._external_potential._detach()
        self._external_potential = new_external_potential

    @log(requires_run=True)
    def pair_energy(self):
        """float: Total potential energy contributed by all pair potentials \
        :math:`[\\mathrm{energy}]`."""
        timestep = self._simulation.timestep
        return self._cpp_obj.computeTotalPairEnergy(timestep)

    @log(requires_run=True)
    def external_energy(self):
        """float: Total external energy contributed by all external potentials \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.computeTotalExternalEnergy(False)


class Sphere(HPMCIntegrator):
    """Sphere hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of spheres.
    The shape :math:`S` includes all points inside and on the surface of a
    sphere:

    .. math::

        S = \\left \\{ \\vec{r} : \\frac{\\vec{r}\\cdot\\vec{r}}{(d/2)^2}
                                  \\le 1 \\right\\}

    where :math:`d`, is the diameter set in `shape`. When the shape parameter
    ``orientable`` is `False` (the default), `Sphere` only applies translation
    trial moves and ignores ``translation_move_probability``.

    Tip:
        Use spheres with ``diameter=0`` in conjunction with pair potentials
        for Monte Carlo simulations of particles with no hard core.

    Tip:
        Use `Sphere` in a 2D simulation to perform Monte Carlo on hard disks.

    .. rubric:: Wall support.

    `Sphere` supports all `hoomd.wall` geometries.

    Examples::

        mc = hoomd.hpmc.integrate.Sphere(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(diameter=1.0)
        mc.shape["B"] = dict(diameter=2.0)
        mc.shape["C"] = dict(diameter=1.0, orientable=True)
        print('diameter = ', mc.shape["A"]["diameter"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``diameter`` (`float`, **required**) - Sphere diameter
              :math:`[\\mathrm{length}]`.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``orientable`` (`bool`, **default:** `False`) - set to `True` to
              allow rotation moves on this particle type.
    """
    _cpp_cls = 'IntegratorHPMCMonoSphere'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameter=float,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Examples:
            The types will be 'Sphere' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'Sphere', 'diameter': 1},
             {'type': 'Sphere', 'diameter': 2}]
        """
        return super()._return_type_shapes()


class ConvexPolygon(HPMCIntegrator):
    """Convex polygon hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of convex polygons. The shape :math:`S`
    of a convex polygon includes the points inside and on the surface of the
    convex hull of the vertices (see `shape`). For example:

    .. image:: convex-polygon.svg
       :alt: Example of a convex polygon with vertex labels.

    Important:
        `ConvexPolygon` simulations must be performed in 2D systems.

    See Also:
        Use `SimplePolygon` for concave polygons.

    .. rubric:: Wall support.

    `ConvexPolygon` supports no `hoomd.wall` geometries.

    Examples::

        mc = hoomd.hpmc.integrate.ConvexPolygon(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5),
                                       (0.5, -0.5),
                                       (0.5, 0.5),
                                       (-0.5, 0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):

            The shape parameters for each particle type. The dictionary has the
            following keys.

            * ``vertices`` (`list` [`tuple` [`float`, `float`]], **required**)
              - vertices of the polygon :math:`[\\mathrm{length}]`.

              * Vertices **MUST** be specified in a *counter-clockwise* order.
              * The origin **MUST** be contained within the polygon.
              * Points inside the polygon **MUST NOT** be included.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `ConvexPolygon` shares data structures with
              `ConvexSpheropolygon` :math:`[\\mathrm{length}]`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior **will result** when they are violated.

    """
    _cpp_cls = 'IntegratorHPMCMonoConvexPolygon'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            ignore_statistics=False,
                                            sweep_radius=0.0,
                                            len_keys=1,
                                        ))

        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(ConvexPolygon, self)._return_type_shapes()


class ConvexSpheropolygon(HPMCIntegrator):
    """Convex spheropolygon hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of convex spheropolygons. The shape
    :math:`S` of a convex spheropolygon includes the points inside and on the
    surface of the convex hull of the vertices plus a disk (with radius
    ``sweep_radius``)swept along the perimeter (see `shape`). For example:

    .. image:: convex-spheropolygon.svg
       :alt: Example of a convex spheropolygon with vertex and sweep labels.

    Important:
        `ConvexSpheropolygon` simulations must be performed in 2D systems.

    Tip:
        To model mixtures of convex polygons and convex spheropolygons, use
        `ConvexSpheropolygon` and set the sweep radius to 0 for some shape
        types.

    Tip:
        A 1-vertex spheropolygon is a disk and a 2-vertex spheropolygon is a
        rectangle with half disk caps.

    .. rubric:: Wall support.

    `ConvexSpheropolygon` supports no `hoomd.wall` geometries.

    Examples::

        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(default_d=0.3,
                                                     default_a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5),
                                       (0.5, -0.5),
                                       (0.5, 0.5),
                                       (-0.5, 0.5)],
                             sweep_radius=0.1);

        mc.shape["A"] = dict(vertices=[(0,0)],
                             sweep_radius=0.5,
                             ignore_statistics=True);

        print('vertices = ', mc.shape["A"]["vertices"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (`list` [`tuple` [`float`, `float`]], **required**)
              - vertices of the polygon  :math:`[\\mathrm{length}]`.

              * The origin **MUST** be contained within the spheropolygon.
              * Points inside the polygon should not be included.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (**default:** 0.0) - radius of the disk swept
              around the edges of the polygon :math:`[\\mathrm{length}]`.
              Set a non-zero ``sweep_radius`` to create a spheropolygon.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoSpheropolygon'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))

        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0.1,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(ConvexSpheropolygon, self)._return_type_shapes()


class SimplePolygon(HPMCIntegrator):
    """Simple polygon hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of simple polygons. The shape :math:`S` of
    a simple polygon includes the points inside and on the surface of the simple
    polygon defined by the vertices (see `shape`). For example:

    .. image:: simple-polygon.svg
       :alt: Example of a simple polygon with vertex labels.

    Important:
        `SimplePolygon` simulations must be performed in 2D systems.

    See Also:
        Use `ConvexPolygon` for faster performance with convex polygons.

    .. rubric:: Wall support.

    `SimplePolygon` supports no `hoomd.wall` geometries.

    Examples::

        mc = hpmc.integrate.SimplePolygon(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(vertices=[(0, 0.5),
                                       (-0.5, -0.5),
                                       (0, 0),
                                       (0.5, -0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])


    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):

            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (`list` [`tuple` [`float`, `float`]], **required**) -
              vertices of the polygon :math:`[\\mathrm{length}]`.

              * Vertices **MUST** be specified in a *counter-clockwise* order.
              * The polygon may be concave, but edges must not cross.
              * The origin may be inside or outside the shape.
              * The origin centered circle that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `SimplePolygon` shares data structures with
              `ConvexSpheropolygon` :math:`[\\mathrm{length}]`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoSimplePolygon'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float)],
                                            ignore_statistics=False,
                                            sweep_radius=0,
                                            len_keys=1))

        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Polygon', 'sweep_radius': 0,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]
        """
        return super(SimplePolygon, self)._return_type_shapes()


class Polyhedron(HPMCIntegrator):
    """Polyhedron hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of general polyhedra. The shape :math:`S`
    contains the points inside the polyhedron defined by vertices and faces (see
    `shape`). `Polyhedron` supports triangle meshes and spheres only. The mesh
    must be free of self-intersections.

    See Also:
        Use `ConvexPolyhedron` for faster performance with convex polyhedra.

    Note:
        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent faces in the tree, different
        values of the number of faces per leaf node may yield different
        optimal performance. The capacity of leaf nodes is configurable.

    .. rubric:: Wall support.

    `Polyhedron` supports no `hoomd.wall` geometries.

    Example::

        mc = hpmc.integrate.Polyhedron(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(vertices=[(-0.5, -0.5, -0.5),
                                       (-0.5, -0.5, 0.5),
                                       (-0.5, 0.5, -0.5),
                                       (-0.5, 0.5, 0.5),
                                       (0.5, -0.5, -0.5),
                                       (0.5, -0.5, 0.5),
                                       (0.5, 0.5, -0.5),
                                       (0.5, 0.5, 0.5)],
                            faces=[[0, 2, 6],
                                   [6, 4, 0],
                                   [5, 0, 4],
                                   [5, 1, 0],
                                   [5, 4, 6],
                                   [5, 6, 7],
                                   [3, 2, 0],
                                   [3, 0, 1],
                                   [3, 6, 2],
                                   [3, 7, 6],
                                   [3, 1, 5],
                                   [3, 5, 7]])
        print('vertices = ', mc.shape["A"]["vertices"])
        print('faces = ', mc.shape["A"]["faces"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - vertices of the polyhedron
              :math:`[\\mathrm{length}]`.

              * The origin **MUST** strictly be contained in the generally
                nonconvex volume defined by the vertices and faces.
              * The origin centered sphere that encloses all vertices should
                be of minimal size for optimal performance.

            * ``faces`` (`list` [`tuple` [`int`, `int`, `int`], **required**) -
              Vertex indices for every triangle in the mesh.

              * For visualization purposes, the faces **MUST** be defined with
                a counterclockwise winding order to produce an outward normal.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - radius of the
              sphere swept around the surface of the polyhedron
              :math:`[\\mathrm{length}]`. Set a non-zero sweep_radius to create
              a spheropolyhedron.
            * ``overlap`` (`list` [`int`], **default:** None) - Check for
              overlaps between faces when ``overlap [i] & overlap[j]`` is
              nonzero (``&`` is the bitwise AND operator). When not `None`,
              ``overlap`` must have a length equal to that of ``faces``. When
              `None` (the default), ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``origin`` (`tuple` [`float`, `float`, `float`],
              **default:** (0,0,0)) - a point strictly inside the shape,
              needed for correctness of overlap checks.
            * ``hull_only`` (`bool`, **default:** `False`) - When `True`, only
              check for intersections between the convex hulls.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoPolyhedron'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            faces=[(int, int, int)],
                                            sweep_radius=0.0,
                                            capacity=4,
                                            origin=(0., 0., 0.),
                                            hull_only=False,
                                            overlap=OnlyIf(to_type_converter(
                                                [bool]),
                                                           allow_none=True),
                                            ignore_statistics=False,
                                            len_keys=1,
                                            _defaults={'overlap': None}))

        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Mesh', 'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
              [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]],
              'faces': [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]}]
        """
        return super(Polyhedron, self)._return_type_shapes()


class ConvexPolyhedron(HPMCIntegrator):
    """Convex polyhedron hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of convex polyhedra. The shape :math:`S`
    of a convex polyhedron includes the points inside and on the surface of the
    convex hull of the vertices (see `shape`). For example:

    .. image:: convex-polyhedron.svg
       :alt: Example of a convex polyhedron with vertex labels.

    See Also:
        Use `Polyhedron` for concave polyhedra.

    .. rubric:: Wall support.

    `ConvexPolyhedron` supports all `hoomd.wall` geometries.

    Example::

        mc = hpmc.integrate.ConvexPolyhedron(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(vertices=[(0.5, 0.5, 0.5),
                                       (0.5, -0.5, -0.5),
                                       (-0.5, 0.5, -0.5),
                                       (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape["A"]["vertices"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys.

            * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - vertices of the polyhedron
              :math:`[\\mathrm{length}]`.

              * The origin **MUST** be contained within the polyhedron.
              * The origin centered sphere that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `ConvexPolyhedron` shares data structures with
              `ConvexSpheropolyhedron` :math:`[\\mathrm{length}]`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedron'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexPolyhedron, self)._return_type_shapes()


class FacetedEllipsoid(HPMCIntegrator):
    r"""Faceted ellipsoid hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of faceted ellipsoids. The shape :math:`S`
    of a faceted ellipsoid is the intersection of an ellipsoid with a convex
    polyhedron defined through halfspaces (see `shape`). The equation defining
    each halfspace is given by:

    .. math::
        \vec{n}_i\cdot \vec{r} + b_i \le 0

    where :math:`\vec{n}_i` is the face normal, and :math:`b_i` is  the offset.

    .. rubric:: Wall support.

    `FacetedEllipsoid` supports no `hoomd.wall` geometries.

    Example::

        mc = hpmc.integrate.FacetedEllipsoid(default_d=0.3, default_a=0.4)

        # half-space intersection
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5]

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],
                      [-.1,-.5,.5],
                      [-.1,.5,.5],
                      [-.1,.5,-.5],
                      [1,-.5,-.5],
                      [1,-.5,.5],
                      [1,.5,.5],
                      [1,.5,-.5]]

        mc.shape["A"] = dict(normals=slab_normals,
                             offsets=slab_offsets,
                             vertices=slab_verts,
                             a=1.0,
                             b=0.5,
                             c=0.5);
        print('a = {}, b = {}, c = {}',
              mc.shape["A"]["a"], mc.shape["A"]["b"], mc.shape["A"]["c"])

    Attributes:
        shape (TypeParameter[``particle type``, dict]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``normals`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - facet normals :math:`\\vec{n}_i`.
            * ``offsets`` (`list` [`float`], **required**) - list of offsets
              :math:`b_i` :math:`[\mathrm{length}^2]`
            * ``a`` (`float`, **required**) - half axis of ellipsoid in the *x*
              direction :math:`[\mathrm{length}]`
            * ``b`` (`float`, **required**) - half axis of ellipsoid in the *y*
              direction :math:`[\mathrm{length}]`
            * ``c`` (`float`, **required**) - half axis of ellipsoid in the *z*
              direction :math:`[\mathrm{length}]`
            * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
              **default:** []) - list of vertices for intersection polyhedron
              (see note below). :math:`[\mathrm{length}]`
            * ``origin`` (`tuple` [`float`, `float`, `float`],
              **default:** (0,0,0)) - A point inside the shape.
              :math:`[\mathrm{length}]`
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.

            Important:
                The origin must be chosen so as to lie **inside the shape**, or
                the overlap check will not work. This condition is not checked.

            Warning:
                Planes must not be coplanar.

            Note:
                For simple intersections with planes that do not intersect
                within the sphere, the vertices list can be left empty. When
                specified, the half-space intersection of the normals must match
                the convex polyhedron defined by the vertices (if non-empty),
                the half-space intersection is **not** calculated automatically.
    """

    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoid'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            normals=[(float, float, float)],
                                            offsets=[float],
                                            a=float,
                                            b=float,
                                            c=float,
                                            vertices=OnlyIf(to_type_converter([
                                                (float, float, float)
                                            ]),
                                                            allow_none=True),
                                            origin=(0.0, 0.0, 0.0),
                                            ignore_statistics=False,
                                            len_keys=1,
                                            _defaults={'vertices': None}))
        self._add_typeparam(typeparam_shape)


class Sphinx(HPMCIntegrator):
    """Sphinx hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of sphere unions and differences,
    depending on the sign of the diameter. The shape :math:`S` is:

    .. math::

        S = \\left(\\bigcup_{k,d_k\\ge 0} S_k((1, 0, 0, 0), \\vec{r}_k) \\right)
            \\setminus \\left(\\bigcup_{k,d_k < 0} S_k((1, 0, 0, 0), \\vec{r}_k)
            \\right)

    Where :math:`d_k` is the diameter given in `shape`, :math:`\\vec{r}_k` is
    the center given in `shape` and :math:`S_k` is the set of points in a sphere
    or diameter :math:`|d_k|`.

    .. rubric:: Wall support.

    `Sphinx` supports no `hoomd.wall` geometries.

    Example::

        mc = hpmc.integrate.Sphinx(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(centers=[(0,0,0),(1,0,0)], diameters=[1,.25])
        print('diameters = ', mc.shape["A"]["diameters"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``diameters`` (`list` [`float`], **required**) -
              diameters of spheres (positive OR negative real numbers)
              :math:`[\\mathrm{length}]`.
            * ``centers`` (`list` [`tuple` [`float`, `float`, `float`],
              **required**) - centers of spheres in local coordinate frame
              :math:`[\\mathrm{length}]`.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoSphinx'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameters=[float],
                                            centers=[(float, float, float)],
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)


class ConvexSpheropolyhedron(HPMCIntegrator):
    """Convex spheropolyhedron hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of convex spheropolyhedra. The shape
    :math:`S` of a convex spheropolyhedron includes the points inside and on the
    surface of the convex hull of the vertices plus a sphere (with radius
    ``sweep_radius``) swept along the perimeter (see `shape`).
    See `ConvexSpheropolygon` for a visual example in 2D.

    Tip:
        A 1-vertex spheropolygon is a sphere and a 2-vertex spheropolygon is a
        spherocylinder.

    .. rubric:: Wall support.

    `ConvexSpheropolyhedron` supports the `hoomd.wall.Sphere` and
    `hoomd.wall.Plane` geometries.

    Example::

        mc = hpmc.integrate.ConvexSpheropolyhedron(default_d=0.3, default_a=0.4)
        mc.shape['tetrahedron'] = dict(vertices=[(0.5, 0.5, 0.5),
                                                 (0.5, -0.5, -0.5),
                                                 (-0.5, 0.5, -0.5),
                                                 (-0.5, -0.5, 0.5)]);
        print('vertices = ', mc.shape['tetrahedron']["vertices"])

        mc.shape['SphericalDepletant'] = dict(vertices=[],
                                              sweep_radius=0.1,
                                              ignore_statistics=True);

    Depletants example::

        mc = hpmc.integrate.ConvexSpheropolyhedron(default_d=0.3, default_a=0.4)
        mc.shape["tetrahedron"] = dict(vertices=[(0.5, 0.5, 0.5),
                                                 (0.5, -0.5, -0.5),
                                                 (-0.5, 0.5, -0.5),
                                                 (-0.5, -0.5, 0.5)]);
        mc.shape["SphericalDepletant"] = dict(vertices=[], sweep_radius=0.1);
        mc.depletant_fugacity["SphericalDepletant"] = 3.0

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - vertices of the polyhedron
              :math:`[\\mathrm{length}]`.

              * The origin **MUST** be contained within the polyhedron.
              * The origin centered sphere that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - radius of the
              sphere swept around the surface of the polyhedron
              :math:`[\\mathrm{length}]`. Set a non-zero sweep_radius to
              create a spheropolyhedron.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """

    _cpp_cls = 'IntegratorHPMCMonoSpheropolyhedron'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0.1,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexSpheropolyhedron, self)._return_type_shapes()


class Ellipsoid(HPMCIntegrator):
    """Ellipsoid hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of ellipsoids. The shape :math:`S`
    includes all points inside and on the surface of an ellipsoid:

    .. math::

        S = \\left \\{ \\vec{r} : \\frac{r_x^2}{a^2}
                           + \\frac{r_y^2}{b^2}
                           + \\frac{r_z^2}{c^2} \\le 1 \\right\\}

    where :math:`r_x`, :math:`r_y`, :math:`r_z` are the components of
    :math:`\\vec{r}`, and the parameters :math:`a`, :math:`b`, and
    :math:`c` are the half axes of the ellipsoid set in `shape`.

    .. rubric:: Wall support.

    `Ellipsoid` supports no `hoomd.wall`  geometries.

    Example::

        mc = hpmc.integrate.Ellipsoid(default_d=0.3, default_a=0.4)
        mc.shape["A"] = dict(a=0.5, b=0.25, c=0.125);
        print('ellipsoids parameters (a,b,c) = ',
              mc.shape["A"]["a"],
              mc.shape["A"]["b"],
              mc.shape["A"]["c"])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``a`` (`float`, **required**) - half axis of ellipsoid in the *x*
              direction :math:`[\\mathrm{length}]`
            * ``b`` (`float`, **required**) - half axis of ellipsoid in the *y*
              direction :math:`[\\mathrm{length}]`
            * ``c`` (`float`, **required**) - half axis of ellipsoid in the *z*
              direction :math:`[\\mathrm{length}]`
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoEllipsoid'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            a=float,
                                            b=float,
                                            c=float,
                                            ignore_statistics=False,
                                            len_keys=1))

        self._extend_typeparam([typeparam_shape])

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'Ellipsoid', 'a': 1.0, 'b': 1.5, 'c': 1}]
        """
        return super()._return_type_shapes()


class SphereUnion(HPMCIntegrator):
    """Sphere union hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of unions of spheres. The union shape
    :math:`S` is the set union of the given spheres:

    .. math::

        S = \\bigcup_k S_k(\\mathbf{q}_k, \\vec{r}_k)

    Each constituent shape in the union has its own shape parameters
    :math:`S_k`, position :math:`\\vec{r}_k`, and orientation
    :math:`\\mathbf{q}_k` (see `shape`).

    Note:
        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent spheres in the tree, different
        values of the number of spheres per leaf node may yield different
        performance. The capacity of leaf nodes is configurable.

    .. rubric:: Wall support.

    `SphereUnion` supports no `hoomd.wall`  geometries.

    Example::

        mc = hpmc.integrate.SphereUnion(default_d=0.3, default_a=0.4)
        sphere1 = dict(diameter=1)
        sphere2 = dict(diameter=2)
        mc.shape["A"] = dict(shapes=[sphere1, sphere2],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1])
        print('diameter of the first sphere = ',
              mc.shape["A"]["shapes"][0]["diameter"])
        print('center of the first sphere = ', mc.shape["A"]["positions"][0])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (`list` [`dict`], **required**) -
              Shape parameters for each sphere in the union. See `Sphere.shape`
              for the accepted parameters.
            * ``positions`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - Position of each sphere in the union.
              :math:`[\\mathrm{length}]`
            * ``orientations`` (`list` [`tuple` [`float`, `float`, `float`,\
              `float`]], **default:** `None`) -
              Orientation of each sphere in the union. When not `None`,
              ``orientations`` must have a length equal to that of
              ``positions``. When `None` (the default), ``orientations`` is
              initialized with all [1,0,0,0]'s.
            * ``overlap`` (`list` [`int`], **default:** `None`) - Check for
              overlaps between constituent particles when
              ``overlap [i] & overlap[j]`` is nonzero (``&`` is the bitwise AND
              operator). When not `None`, ``overlap`` must have a length equal
              to that of ``positions``. When `None` (the default), ``overlap``
              is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoSphereUnion'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(shapes=[
                dict(diameter=float, ignore_statistics=False, orientable=False)
            ],
                                         positions=[(float, float, float)],
                                         orientations=OnlyIf(to_type_converter([
                                             (float, float, float, float)
                                         ]),
                                                             allow_none=True),
                                         capacity=4,
                                         overlap=OnlyIf(to_type_converter([int
                                                                           ]),
                                                        allow_none=True),
                                         ignore_statistics=False,
                                         len_keys=1,
                                         _defaults={
                                             'orientations': None,
                                             'overlap': None
                                         }))
        self._add_typeparam(typeparam_shape)

    @log(category='object', requires_run=True)
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Examples:
            The type will be 'SphereUnion' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'SphereUnion',
              'centers': [[0, 0, 0], [0, 0, 1]],
              'diameters': [1, 0.5]},
             {'type': 'SphereUnion',
              'centers': [[1, 2, 3], [4, 5, 6]],
              'diameters': [0.5, 1]}]
        """
        return super()._return_type_shapes()


class ConvexSpheropolyhedronUnion(HPMCIntegrator):
    """Convex spheropolyhedron union hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of unions of convex sphereopolyhedra. The
    union shape :math:`S` is the set union of the given convex spheropolyhedra:

    .. math::

        S = \\bigcup_k S_k(\\mathbf{q}_k, \\vec{r}_k)

    Each constituent shape in the union has its own shape parameters
    :math:`S_k`, position :math:`\\vec{r}_k`, and orientation
    :math:`\\mathbf{q}_k` (see `shape`).

    Note:
        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent spheropolyhedra in the tree,
        different values of the number of spheropolyhedra per leaf node may
        yield different performance. The capacity of leaf nodes is configurable.

    .. rubric:: Wall support.

    `ConvexSpheropolyhedronUnion` supports no `hoomd.wall`
    geometries.

    Example::

        mc = hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion(default_d=0.3,
                                                              default_a=0.4)
        cube_verts = [[-1,-1,-1],
                      [-1,-1,1],
                      [-1,1,1],
                      [-1,1,-1],
                      [1,-1,-1],
                      [1,-1,1],
                      [1,1,1],
                      [1,1,-1]]
        mc.shape["A"] = dict(shapes=[cube_verts, cube_verts],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1]);
        print('vertices of the first cube = ',
              mc.shape["A"]["shapes"][0]["vertices"])
        print('center of the first cube = ', mc.shape["A"]["positions"][0])
        print('orientation of the first cube = ',
              mc.shape_param["A"]["orientations"][0])

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (`list` [`dict`], **required**) -
              Shape parameters for each spheropolyhedron in the union. See
              `ConvexSpheropolyhedron.shape` for the accepted parameters.
            * ``positions`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - Position of each spheropolyhedron in the union.
              :math:`[\\mathrm{length}]`
            * ``orientations`` (`list` [ `tuple` [`float`, `float`, `float`,\
              `float`]], **default:** None) - Orientation of each
              spheropolyhedron in the union. When not `None`,
              ``orientations`` must have a length equal to that of
              ``positions``. When `None` (the default), ``orientations`` is
              initialized with all [1,0,0,0]'s.
            * ``overlap`` (`list` [`int`], **default:** `None`) - Check for
              overlaps between constituent particles when
              ``overlap [i] & overlap[j]`` is nonzero (``&`` is the bitwise
              AND operator). When not `None`, ``overlap`` must have a length
              equal to that of ``positions``. When `None` (the default),
              ``overlap`` is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoConvexPolyhedronUnion'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(shapes=[
                dict(vertices=[(float, float, float)],
                     sweep_radius=0.0,
                     ignore_statistics=False)
            ],
                                         positions=[(float, float, float)],
                                         orientations=OnlyIf(to_type_converter([
                                             (float, float, float, float)
                                         ]),
                                                             allow_none=True),
                                         overlap=OnlyIf(to_type_converter([int
                                                                           ]),
                                                        allow_none=True),
                                         ignore_statistics=False,
                                         capacity=4,
                                         len_keys=1,
                                         _defaults={
                                             'orientations': None,
                                             'overlap': None
                                         }))

        self._add_typeparam(typeparam_shape)
        # meta data
        self.metadata_fields = ['capacity']


class FacetedEllipsoidUnion(HPMCIntegrator):
    """Faceted ellispod union hard particle Monte Carlo integrator.

    Args:
        default_d (float): Default maximum size of displacement trial moves
            :math:`[\\mathrm{length}]`.
        default_a (float): Default maximum size of rotation trial moves
            :math:`[\\mathrm{dimensionless}]`.
        translation_move_probability (float): Fraction of moves that are
            translation moves.
        nselect (int): Number of trial moves to perform per particle per
            timestep.

    Perform hard particle Monte Carlo of unions of faceted ellipsoids. The union
    shape :math:`S` is the set union of the given faceted ellipsoids:

    .. math::

        S = \\bigcup_k S_k(\\mathbf{q}_k, \\vec{r}_k)

    Each constituent shape in the union has its own shape parameters
    :math:`S_k`, position :math:`\\vec{r}_k`, and orientation
    :math:`\\mathbf{q}_k` (see `shape`).

    Note:
        This shape uses an internal OBB tree for fast collision queries.
        Depending on the number of constituent faceted ellipsoids in the tree,
        different values of the number of faceted ellipsoids per leaf node may
        yield different performance. The capacity of leaf nodes is configurable.

    .. rubric:: Wall support.

    `FacetedEllipsoidUnion` supports no `hoomd.wall` geometries.

    Example::

        mc = hpmc.integrate.FacetedEllipsoidUnion(default_d=0.3, default_a=0.4)

        # make a prolate Janus ellipsoid
        # cut away -x halfspace
        normals = [(-1,0,0)]
        offsets = [0]
        slab_normals = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        slab_offsets = [-0.1,-1,-.5,-.5,-.5,-.5)

        # polyedron vertices
        slab_verts = [[-.1,-.5,-.5],
                      [-.1,-.5,.5],
                      [-.1,.5,.5],
                      [-.1,.5,-.5],
                      [1,-.5,-.5],
                      [1,-.5,.5],
                      [1,.5,.5],
                      [1,.5,-.5]]

        faceted_ellipsoid1 = dict(normals=slab_normals,
                                  offsets=slab_offsets,
                                  vertices=slab_verts,
                                  a=1.0,
                                  b=0.5,
                                  c=0.5);
        faceted_ellipsoid2 = dict(normals=slab_normals,
                                  offsets=slab_offsets,
                                  vertices=slab_verts,
                                  a=0.5,
                                  b=1,
                                  c=1);

        mc.shape["A"] = dict(shapes=[faceted_ellipsoid1, faceted_ellipsoid2],
                             positions=[(0, 0, 0), (0, 0, 1)],
                             orientations=[(1, 0, 0, 0), (1, 0, 0, 0)],
                             overlap=[1, 1]);

        print('offsets of the first faceted ellipsoid = ',
              mc.shape["A"]["shapes"][0]["offsets"])
        print('normals of the first faceted ellipsoid = ',
              mc.shape["A"]["shapes"][0]["normals"])
        print('vertices of the first faceted ellipsoid = ',
              mc.shape["A"]["shapes"][0]["vertices"]

    Attributes:
        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``shapes`` (`list` [ `dict`], **required**) -
              Shape parameters for each faceted ellipsoid in the union. See
              `FacetedEllipsoid.shape` for the accepted parameters.
            * ``positions`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - Position of each faceted ellipsoid in the union.
              :math:`[\\mathrm{length}]`
            * ``orientations`` (`list` [`tuple` [`float`, `float`, `float`,
              `float`]], **default:** `None`) - Orientation of each faceted
              ellipsoid in the union. When not `None`, ``orientations``
              must have a length equal to that of ``positions``. When `None`
              (the default), ``orientations`` is initialized with all
              [1,0,0,0]'s.
            * ``overlap`` (`list` [`int`], **default:** `None`) - Check for
              overlaps between constituent particles when
              ``overlap [i] & overlap[j]`` is nonzero (``&`` is the bitwise AND
              operator). When not `None`, ``overlap`` must have a length equal
              to that of ``positions``. When `None` (the default), ``overlap``
              is initialized with all 1's.
            * ``capacity`` (`int`, **default:** 4) - set the maximum number of
              particles per leaf node to adjust performance.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
    """

    _cpp_cls = 'IntegratorHPMCMonoFacetedEllipsoidUnion'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 translation_move_probability=0.5,
                 nselect=4,
                 kT=1.0):

        # initialize base class
        super().__init__(default_d, default_a, translation_move_probability,
                         nselect, kT)

        typeparam_shape = TypeParameter(
            'shape',
            type_kind='particle_types',
            param_dict=TypeParameterDict(shapes=[
                dict(a=float,
                     b=float,
                     c=float,
                     normals=[(float, float, float)],
                     offsets=[float],
                     vertices=[(float, float, float)],
                     origin=(float, float, float),
                     ignore_statistics=False)
            ],
                                         positions=[(float, float, float)],
                                         orientations=OnlyIf(to_type_converter([
                                             (float, float, float, float)
                                         ]),
                                                             allow_none=True),
                                         overlap=OnlyIf(to_type_converter([int
                                                                           ]),
                                                        allow_none=True),
                                         ignore_statistics=False,
                                         capacity=4,
                                         len_keys=1,
                                         _defaults={
                                             'orientations': None,
                                             'overlap': None
                                         }))
        self._add_typeparam(typeparam_shape)
