# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Apply forces to particles."""

from abc import abstractmethod

import hoomd
from hoomd.md import _md
from hoomd.operation import Compute
from hoomd.logging import log
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import ParticleFilter
from hoomd.md.manifold import Manifold
import numpy


class Force(Compute):
    r"""Defines a force for molecular dynamics simulations.

    `Force` is the base class for all molecular dynamics forces and provides
    common methods.

    A `Force` class computes the force and torque on each particle in the
    simulation state :math:`\vec{F}_i` and :math:`\vec{\tau}_i`. With a few
    exceptions (noted in the documentation of the specific force classes),
    `Force` subclasses also compute the contribution to the system's potential
    energy :math:`U` and the the virial tensor :math:`W`. `Force` breaks the
    computation of the total system :math:`U` and :math:`W` into per-particle
    and additional terms as detailed in the documentation for each specific
    `Force` subclass.

    .. math::

        U & = U_\mathrm{additional} + \sum_{i=0}^{N_\mathrm{particles}-1} U_i \\
        W & = W_\mathrm{additional} + \sum_{i=0}^{N_\mathrm{particles}-1} W_i

    `Force` represents virial tensors as six element arrays listing the
    components of the tensor in this order:

    .. math::

        (W^{xx}, W^{xy}, W^{xz}, W^{yy}, W^{yz}, W^{zz}).

    The components of the virial tensor for a force on a single particle are:

    .. math::

        W^{kl}_i = F^k \cdot r_i^l

    where the superscripts select the x,y, and z components of the vectors.
    To properly account for periodic boundary conditions, pairwise interactions
    evaluate the virial:

    .. math::

        W^{kl}_i = \frac{1}{2} \sum_j F^k_{ij} \cdot
        \mathrm{minimum\_image}(\vec{r}_j - \vec{r}_i)^l

    Tip:
        Add a `Force` to your integrator's `forces <hoomd.md.Integrator.forces>`
        list to include it in the equations of motion of your system. Add a
        `Force` to your simulation's `operations.computes
        <hoomd.Operations.computes>` list to compute the forces and energy
        without influencing the system dynamics.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    def __init__(self):
        self._in_context_manager = False

    @log(requires_run=True)
    def energy(self):
        """float: The potential energy :math:`U` of the system from this force \
        :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.calcEnergySum()

    @log(category="particle", requires_run=True)
    def energies(self):
        """(*N_particles*, ) `numpy.ndarray` of ``float``: Energy \
        contribution :math:`U_i` from each particle :math:`[\\mathrm{energy}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `energies` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getEnergies()

    @log(requires_run=True)
    def additional_energy(self):
        """float: Additional energy term :math:`U_\\mathrm{additional}` \
        :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getExternalEnergy()

    @log(category="particle", requires_run=True)
    def forces(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``float``: The \
        force :math:`\\vec{F}_i` applied to each particle \
        :math:`[\\mathrm{force}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `forces` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getForces()

    @log(category="particle", requires_run=True)
    def torques(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``float``: The torque \
        :math:`\\vec{\\tau}_i` applied to each particle \
        :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `torques` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getTorques()

    @log(category="particle", requires_run=True)
    def virials(self):
        """(*N_particles*, 6) `numpy.ndarray` of ``float``: Virial tensor \
        contribution :math:`W_i` from each particle :math:`[\\mathrm{energy}]`.

        Attention:
            To improve performance `Force` objects only compute virials when
            needed. When not computed, `virials` is `None`. Virials are computed
            on every step when using a `md.methods.ConstantPressure`
            integrator, on steps where a writer is triggered (such as
            `write.GSD` which may log pressure or virials), or when
            `Simulation.always_compute_pressure` is `True`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `virials` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getVirials()

    @log(category="sequence", requires_run=True)
    def additional_virial(self):
        """(1, 6) `numpy.ndarray` of ``float``: Additional virial tensor \
        term :math:`W_\\mathrm{additional}` :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        virial = []
        for i in range(6):
            virial.append(self._cpp_obj.getExternalVirial(i))
        return numpy.array(virial, dtype=numpy.float64)

    @property
    def cpu_local_force_arrays(self):
        """hoomd.md.data.ForceLocalAccess: Expose force arrays on the CPU.

        Provides direct access to the force, potential energy, torque, and
        virial data of the particles in the system on the cpu through a context
        manager. All data is MPI rank-local.

        The `hoomd.md.data.ForceLocalAccess` object returned by this property
        has four arrays through which one can modify the force data:

        Note:
            The local arrays are read only for built-in forces. Use `Custom` to
            implement custom forces.

        Examples::

            with self.cpu_local_force_arrays as arrays:
                arrays.force[:] = ...
                arrays.potential_energy[:] = ...
                arrays.torque[:] = ...
                arrays.virial[:] = ...
        """
        if self._in_context_manager:
            raise RuntimeError("Cannot enter cpu_local_force_arrays context "
                               "manager inside another local_force_arrays "
                               "context manager")
        if not self._attached:
            raise hoomd.error.DataAccessError("cpu_local_force_arrays")
        return hoomd.md.data.ForceLocalAccess(self, self._simulation.state)

    @property
    def gpu_local_force_arrays(self):
        """hoomd.md.data.ForceLocalAccessGPU: Expose force arrays on the GPU.

        Provides direct access to the force, potential energy, torque, and
        virial data of the particles in the system on the gpu through a context
        manager. All data is MPI rank-local.

        The `hoomd.md.data.ForceLocalAccessGPU` object returned by this property
        has four arrays through which one can modify the force data:

        Note:
            The local arrays are read only for built-in forces. Use `Custom` to
            implement custom forces.

        Examples::

            with self.gpu_local_force_arrays as arrays:
                arrays.force[:] = ...
                arrays.potential_energy[:] = ...
                arrays.torque[:] = ...
                arrays.virial[:] = ...

        Note:
            GPU local force data is not available if the chosen device for the
            simulation is `hoomd.device.CPU`.
        """
        if not isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError(
                "Cannot access gpu_local_force_arrays without a GPU device")
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_force_arrays context manager inside "
                "another local_force_arrays context manager")
        if not self._attached:
            raise hoomd.error.DataAccessError("gpu_local_force_arrays")
        return hoomd.md.data.ForceLocalAccessGPU(self, self._simulation.state)


class Custom(Force):
    """Custom forces implemented in python.

    Derive a custom force class from `Custom`, and override the `set_forces`
    method to compute forces on particles. Users have direct, zero-copy access
    to the C++ managed buffers via either the `cpu_local_force_arrays` or
    `gpu_local_force_arrays` property. Choose the property that corresponds to
    the device you wish to alter the data on. In addition to zero-copy access to
    force buffers, custom forces have access to the local snapshot API via the
    ``_state.cpu_local_snapshot`` or the ``_state.gpu_local_snapshot`` property.

    See Also:
      See the documentation in `hoomd.State` for more information on the local
      snapshot API.

    .. rubric:: Examples:

    .. code-block:: python

        class MyCustomForce(hoomd.md.force.Custom):
            def __init__(self):
                super().__init__(aniso=True)

            def set_forces(self, timestep):
                with self.cpu_local_force_arrays as arrays:
                    arrays.force[:] = -5
                    arrays.torque[:] = 3
                    arrays.potential_energy[:] = 27
                    arrays.virial[:] = np.arange(6)[None, :]

    In addition, since data is MPI rank-local, there may be ghost particle data
    associated with each rank. To access this read-only ghost data, access the
    property name with either the prefix ``ghost_`` of the suffix
    ``_with_ghost``.

    Note:
        Pass ``aniso=True`` to the `md.force.Custom` constructor if your custom
        force produces non-zero torques on particles.

    .. code-block:: python

        class MyCustomForce(hoomd.md.force.Custom):
            def __init__(self):
                super().__init__()

            def set_forces(self, timestep):
                with self.cpu_local_force_arrays as arrays:
                    # access only the ghost particle forces
                    ghost_force_data = arrays.ghost_force

                    # access torque data on this rank and ghost torque data
                    torque_data = arrays.torque_with_ghost

    Note:
        When accessing the local force arrays, always use a context manager.

    Note:
        The shape of the exposed arrays cannot change while in the context
        manager.

    Note:
        All force data buffers are MPI rank local, so in simulations with MPI,
        only the data for a single rank is available.

    Note:
        Access to the force buffers is constant (O(1)) time.

    .. versionchanged:: 3.1.0
        `Custom` zeros the force, torque, energy, and virial arrays before
        calling the user-provided `set_forces`.
    """

    def __init__(self, aniso=False):
        super().__init__()
        self._aniso = aniso

        self._state = None  # to be set on attaching

    def _attach_hook(self):
        self._state = self._simulation.state
        self._cpp_obj = _md.CustomForceCompute(self._state._cpp_sys_def,
                                               self.set_forces, self._aniso)

    @abstractmethod
    def set_forces(self, timestep):
        """Set the forces in the simulation loop.

        Args:
            timestep (int): The current timestep in the simulation.
        """
        pass


class Active(Force):
    r"""Active force.

    Args:
        filter (`hoomd.filter`): Subset of particles on which to
            apply active forces.

    `Active` computes an active force and torque on all particles selected by
    the filter:

    .. math::

        \vec{F}_i = \mathbf{q}_i \vec{f}_i \mathbf{q}_i^* \\
        \vec{\tau}_i = \mathbf{q}_i \vec{u}_i \mathbf{q}_i^*,

    where :math:`\vec{f}_i` is the active force in the local particle
    coordinate system (set by type `active_force`) and :math:`\vec{u}_i`
    is the active torque in the local particle coordinate system (set by type
    in `active_torque`.

    Note:
        To introduce rotational diffusion to the particle orientations, use
        `create_diffusion_updater`.

        .. seealso::

            `hoomd.md.update.ActiveRotationalDiffusion`

    Examples::

        all = hoomd.filter.All()
        active = hoomd.md.force.Active(
            filter=hoomd.filter.All()
            )
        active.active_force['A','B'] = (1,0,0)
        active.active_torque['A','B'] = (0,0,0)
        rotational_diffusion_updater = active.create_diffusion_updater(
            trigger=10)
        sim.operations += rotational_diffusion_updater

    Note:

        The energy and virial associated with the active force are 0.

    Attributes:
        filter (`hoomd.filter`): Subset of particles on which to
            apply active forces.

    .. py:attribute:: active_force

        Active force vector in the local reference frame of the particle
        :math:`[\mathrm{force}]`.  It is defined per particle type and stays
        constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. py:attribute:: active_torque

        Active torque vector in the local reference frame of the particle
        :math:`[\mathrm{force} \cdot \mathrm{length}]`. It is defined per
        particle type and stays constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]
    """

    def __init__(self, filter):
        super().__init__()
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter)
        param_dict["filter"] = filter
        # set defaults
        self._param_dict.update(param_dict)

        active_force = TypeParameter(
            "active_force",
            type_kind="particle_types",
            param_dict=TypeParameterDict((1.0, 0.0, 0.0), len_keys=1),
        )
        active_torque = TypeParameter(
            "active_torque",
            type_kind="particle_types",
            param_dict=TypeParameterDict((0.0, 0.0, 0.0), len_keys=1),
        )

        self._extend_typeparam([active_force, active_torque])

    def _attach_hook(self):
        # Active forces use RNGs. Warn the user if they did not set the seed.
        self._simulation._warn_if_seed_unset()
        # Set C++ class
        self._set_cpp_obj()

    def _set_cpp_obj(self):

        # initialize the reflected c++ class
        sim = self._simulation

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = _md.ActiveForceCompute
        else:
            my_class = _md.ActiveForceComputeGPU

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter))

    def create_diffusion_updater(self, trigger, rotational_diffusion):
        """Create a rotational diffusion updater for this active force.

        Args:
            trigger (hoomd.trigger.trigger_like): Select the timesteps to update
                rotational diffusion.
            rotational_diffusion (hoomd.variant.variant_like): The
                rotational diffusion as a function of time or a constant.

        Returns:
            hoomd.md.update.ActiveRotationalDiffusion:
                The rotational diffusion updater.
        """
        return hoomd.md.update.ActiveRotationalDiffusion(
            trigger, self, rotational_diffusion)


class ActiveOnManifold(Active):
    r"""Active force on a manifold.

    Args:
        filter (`hoomd.filter`): Subset of particles on which to
            apply active forces.
        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint.

    `ActiveOnManifold` computes a constrained active force and torque on all
    particles selected by the filter similar to `Active`. `ActiveOnManifold`
    restricts the forces to the local tangent plane of the manifold constraint.
    For more information see `Active`.

    Hint:
        Use `ActiveOnManifold` with a `md.methods.rattle` integration method
        with the same manifold constraint.

    Note:
        To introduce rotational diffusion to the particle orientations, use
        `create_diffusion_updater`. The rotational diffusion occurs in the local
        tangent plane of the manifold.

        .. seealso::

            `hoomd.md.update.ActiveRotationalDiffusion`

    Examples::

        all = filter.All()
        sphere = hoomd.md.manifold.Sphere(r=10)
        active = hoomd.md.force.ActiveOnManifold(
            filter=hoomd.filter.All(), rotation_diff=0.01,
            manifold_constraint = sphere
            )
        active.active_force['A','B'] = (1,0,0)
        active.active_torque['A','B'] = (0,0,0)

    Attributes:
        filter (`hoomd.filter`): Subset of particles on which to
            apply active forces.
        manifold_constraint (hoomd.md.manifold.Manifold): Manifold constraint.

    .. py:attribute:: active_force

        Active force vector in the local reference frame of the particle
        :math:`[\mathrm{force}]`.  It is defined per particle type and stays
        constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. py:attribute:: active_torque

        Active torque vector in local reference frame of the particle
        :math:`[\mathrm{force} \cdot \mathrm{length}]`. It is defined per
        particle type and stays constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]
    """

    def __init__(self, filter, manifold_constraint):
        # store metadata
        super().__init__(filter)
        param_dict = ParameterDict(
            manifold_constraint=OnlyTypes(Manifold, allow_none=False))
        param_dict["manifold_constraint"] = manifold_constraint
        self._param_dict.update(param_dict)

    def _setattr_param(self, attr, value):
        if attr == "manifold_constraint":
            raise AttributeError(
                "Cannot set manifold_constraint after construction.")
        super()._setattr_param(attr, value)

    def _set_cpp_obj(self):

        # initialize the reflected c++ class
        sim = self._simulation

        if not self.manifold_constraint._attached:
            self.manifold_constraint._attach(sim)

        base_class_str = 'ActiveForceConstraintCompute'
        base_class_str += self.manifold_constraint.__class__.__name__
        if isinstance(sim.device, hoomd.device.GPU):
            base_class_str += "GPU"
        self._cpp_obj = getattr(
            _md, base_class_str)(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj)


class Constant(Force):
    r"""Constant force.

    Args:
        filter (`hoomd.filter`): Subset of particles on which to
            apply constant forces.

    `Constant` applies a type dependent constant force and torque on all
    particles selected by the filter. `Constant` sets the force and torque
    to  ``(0,0,0)`` for particles not selected by the filter.

    Examples::

        constant = hoomd.md.force.Constant(
            filter=hoomd.filter.All()
            )
        constant.constant_force['A'] = (1,0,0)
        constant.constant_torque['A'] = (0,0,0)

    Note:

        The energy and virial associated with the constant force are 0.

    Attributes:
        filter (`hoomd.filter`): Subset of particles on which to
            apply constant forces.

    .. py:attribute:: constant_force

        Constant force vector in the global reference frame of the system
        :math:`[\mathrm{force}]`.  It is defined per particle type and
        defaults to (0.0, 0.0, 0.0) for all types.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. py:attribute:: constant_torque

        Constant torque vector in the global reference frame of the system
        :math:`[\mathrm{force} \cdot \mathrm{length}]`. It is defined per
        particle type and defaults to (0.0, 0.0, 0.0) for all types.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]
    """

    def __init__(self, filter):
        super().__init__()
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter)
        param_dict["filter"] = filter
        # set defaults
        self._param_dict.update(param_dict)

        constant_force = TypeParameter(
            "constant_force",
            type_kind="particle_types",
            param_dict=TypeParameterDict((0.0, 0.0, 0.0), len_keys=1),
        )
        constant_torque = TypeParameter(
            "constant_torque",
            type_kind="particle_types",
            param_dict=TypeParameterDict((0.0, 0.0, 0.0), len_keys=1),
        )

        self._extend_typeparam([constant_force, constant_torque])

    def _attach_hook(self):
        # initialize the reflected c++ class
        sim = self._simulation

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = _md.ConstantForceCompute
        else:
            my_class = _md.ConstantForceComputeGPU

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter))
